import os
from dataclasses import dataclass
from itertools import islice
from typing import Generator, List, Tuple, Optional, Any, Dict
import time

import json
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import PreTrainedTokenizer, PreTrainedModel
from tqdm.auto import tqdm
from contextlib import contextmanager

from eleuther_sae.sae.data import chunk_and_tokenize


########################################
### MODEL ARCHITECTURE CONFIGS ###
########################################

@dataclass
class ModelArchConfig:
    """Configuration for a model architecture"""
    attn_path_template: str
    mlp_path_template: str
    attn_modules: List[str]
    mlp_modules: List[str]
    
class ModelConfigs:
    """Configurations for different model architectures"""
    
    LLAMA = ModelArchConfig(
        attn_path_template="model.layers.{}.self_attn.{}",
        mlp_path_template="model.layers.{}.mlp.{}",
        attn_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        mlp_modules=["gate_proj", "up_proj", "down_proj"]
    )
    
    GEMMA = ModelArchConfig(
        attn_path_template="model.layers.{}.self_attn.{}",
        mlp_path_template="model.layers.{}.mlp.{}",
        attn_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        mlp_modules=["gate_proj", "up_proj", "down_proj"]
    )
    
    GPT2 = ModelArchConfig(
        attn_path_template="transformer.h.{}.attn.{}",
        mlp_path_template="transformer.h.{}.mlp.{}",
        attn_modules=["c_attn", "c_proj"],
        mlp_modules=["c_fc", "c_proj"]
    )
    
    @classmethod
    def get_config(cls, model_name: str) -> ModelArchConfig:
        """Get the configuration for a specific model"""
        if "llama" in model_name.lower():
            return cls.LLAMA
        elif "gemma" in model_name.lower():
            return cls.GEMMA
        elif "gpt2" in model_name.lower():
            return cls.GPT2
        else:
            raise ValueError(f"Unsupported model: {model_name}")

def get_target_modules(
    model_name: str,
    peft_layers: List[int],
    peft_type: str
) -> List[str]:
    """Get target modules for PEFT configuration"""
    
    config = ModelConfigs.get_config(model_name)
    
    def get_attn_modules():
        return [
            config.attn_path_template.format(i, module)
            for i in peft_layers
            for module in config.attn_modules
        ]
    
    def get_mlp_modules():
        return [
            config.mlp_path_template.format(i, module)
            for i in peft_layers
            for module in config.mlp_modules
        ]
    
    def get_pre_mlp_modules():
        return [
            config.mlp_path_template.format(i, module)
            for i in peft_layers
            for module in ["gate_proj", "up_proj"]
        ]
    
    peft_types = {
        "attn": get_attn_modules,
        "mlp": get_mlp_modules,
        "pre-mlp": get_pre_mlp_modules,
        "both": lambda: get_attn_modules() + get_mlp_modules(),
        "gate": lambda: [config.mlp_path_template.format(i, "gate_proj") for i in peft_layers],
        "up": lambda: [config.mlp_path_template.format(i, "up_proj") for i in peft_layers]
    }
    
    if peft_type not in peft_types:
        raise ValueError(f"Invalid peft_type: {peft_type}")
    
    return peft_types[peft_type]()

def get_peft_model_layers(peft_model, model_name: str):
    """Get the appropriate layers attribute based on model architecture.
    
    Args:
        peft_model: The PEFT model instance
        model_name (str): Name of the model architecture
        
    Returns:
        nn.ModuleList: The layers of the model
        
    Raises:
        ValueError: If the model architecture is not supported
    """
    if any(name in model_name.lower() for name in ["gemma", "llama", "mistral"]):
        return peft_model.model.model.layers
    elif "gpt2" in model_name.lower():
        return peft_model.model.transformer.h
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")


def initialize_auth() -> None:
    """Initialize authentication for Hugging Face and Weights & Biases."""
    load_dotenv()
    
    # Hugging Face authentication
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables")
    login(token=hf_token)
    
    # Weights & Biases authentication
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        raise ValueError("WANDB_API_KEY not found in environment variables")
    wandb.login(key=wandb_key)






########################################
### DATA PROCESSING FUNCTIONS ###
########################################

def _tokenized_batch(
    generator: Generator[str, None, None],
    tokenizer: PreTrainedTokenizer,
    args: Any
) -> torch.Tensor:
    """Create a batch of tokenized texts."""
    batch = []
    try:
        while len(batch) < args.batch_size:
            next_text = next(generator)
            tokenized = tokenizer(
                next_text,
                return_tensors="pt",
                max_length=args.ctx_len,
                padding=False,
                truncation=True,
            )
            if tokenized["input_ids"].shape[1] == args.ctx_len:
                batch.append(tokenized)
    except StopIteration:
        if not batch:
            raise RuntimeError("Generator exhausted before creating a batch")
    
    return torch.cat([x["input_ids"] for x in batch], dim=0)

def hf_dataset_to_generator(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    args: Any,
    split: str = "train", 
    streaming: bool = True,
    seed: int = 42
) -> Tuple[Generator, List[torch.Tensor]]:
    """
    Process a Hugging Face dataset into training generator and validation tensors.
    Returns:
        Tuple[Generator, List[torch.Tensor]]: A tuple containing:
            - Generator that yields batched training examples
            - List of validation tensors (batched)
    """
    print(f"Loading dataset: {dataset_name}")
    is_redpajama = dataset_name == "togethercomputer/RedPajama-Data-V2"
    is_pile = dataset_name == "monology/pile-uncopyrighted"
    dataset = load_dataset(
        dataset_name,
        name="sample-10B" if is_redpajama else None,
        data_files=["train/00.jsonl.zst"] if is_pile else None,
        trust_remote_code=True,
    ) 

    dataset = dataset.shuffle(seed=42)
    max_seq_len = args.ctx_len
    tokenized = chunk_and_tokenize(dataset, tokenizer, text_key="raw_content" if is_redpajama else "text", max_seq_len=max_seq_len)['train']

    total_train_tokens = args.num_train_examples * max_seq_len
    val_tokens = args.num_val_tokens
    total_tokens = total_train_tokens + val_tokens

    total_cutoff = (total_tokens + max_seq_len - 1) // max_seq_len
    train_cutoff = (total_train_tokens + max_seq_len - 1) // max_seq_len
    val_cutoff = total_cutoff - train_cutoff

    # Get validation data as tensors (typically smaller)
    subset_val_data = tokenized.select(range(val_cutoff))
    
    # Create generator for training data that yields batches
    def train_gen():
        batch_examples = []
        for i in range(val_cutoff, val_cutoff + train_cutoff):
            batch_examples.append(tokenized[i]["input_ids"])
            if len(batch_examples) == args.batch_size:
                yield torch.stack(batch_examples)
                batch_examples = []
        # Yield any remaining examples in the final batch
        if batch_examples:
            yield torch.stack(batch_examples)
    
    # Convert validation data to batched tensors
    val_tensors = []
    for i in range(0, len(subset_val_data), args.batch_size):
        batch_indices = range(i, min(i + args.batch_size, len(subset_val_data)))
        batch = torch.stack([subset_val_data[j]["input_ids"] for j in batch_indices])
        val_tensors.append(batch)
    
    print(f"Created {len(val_tensors)} validation batches of size up to {args.batch_size}")
    return train_gen(), val_tensors






########################################
### SAVE DATA FUNCTIONS ###
########################################

def _get_data_filenames(
    model_name: str,
    sae_path: Optional[str],
    peft_layers: List[int],
    sae_from_hf: bool,
    peft_rank: int,
    num_train_examples: int,
    use_16_bit: bool
) -> Tuple[str, str, str]:
    """
    Helper function to generate filenames for CE increase and validation losses data.
    """
    num_train_examples_in_thousands = num_train_examples // 1000
    model_name = model_name.split('/')[-1]

    if sae_path:
        if not sae_from_hf:
            sae_path = sae_path.split('/')[-2]
        else:
            parts = sae_path.split('/', 1)
            sae_path = f"{parts[1].replace('/', '_')}"
        
    peft_range = f"{peft_layers[0]}-{peft_layers[-1]}" if len(peft_layers) > 1 else peft_layers[0]
    
    # Build base paths
    base_path = f"data/scaling" if sae_from_hf else f"data/TopK"

    ce_base_path = f"{base_path}/CE_increase/{model_name}/{sae_path}"
    val_base_path = f"{base_path}/val_loss/{model_name}/{sae_path}" 
    time_base_path = f"{base_path}/time/{model_name}/{sae_path}"
    
    CE_increase_filename = f"{ce_base_path}/peft_{peft_range}_rank_{peft_rank}_CE_increase_{num_train_examples_in_thousands}k.json"
    val_losses_filename = f"{val_base_path}/peft_{peft_range}_rank_{peft_rank}_val_losses_{num_train_examples_in_thousands}k.json"
    time_filename = f"{time_base_path}/peft_{peft_range}_rank_{peft_rank}_time_{num_train_examples_in_thousands}k.json"

    return CE_increase_filename, val_losses_filename, time_filename

def save_data(
    CE_increase: dict,
    val_losses_dict: dict,
    total_training_minutes_dict: dict,
    **kwargs
) -> None:
    """
    Save CE increase and validation losses data to json files.
    """
    CE_increase_filename, val_losses_filename, time_filename = _get_data_filenames(
        model_name=kwargs['model_name'],
        sae_path=kwargs['sae_path'],
        peft_layers=kwargs['peft_layers'],
        sae_from_hf=kwargs['sae_from_hf'],
        peft_rank=kwargs['peft_rank'],
        num_train_examples=kwargs['num_train_examples'],
        use_16_bit=kwargs['use_16_bit']
    )
    
    os.makedirs(os.path.dirname(CE_increase_filename), exist_ok=True)
    os.makedirs(os.path.dirname(val_losses_filename), exist_ok=True)
    os.makedirs(os.path.dirname(time_filename), exist_ok=True)

    with open(CE_increase_filename, "w") as f:
        json.dump(CE_increase, f, indent=4)

    with open(val_losses_filename, "w") as f:
        json.dump(val_losses_dict, f, indent=4)
    
    with open(time_filename, "w") as f:
        json.dump(total_training_minutes_dict, f, indent=4)

    print(f"Saved CE increase data to {CE_increase_filename}")
    print(f"Saved validation losses to {val_losses_filename}")
    print(f"Saved total training minutes to {time_filename}")

def load_data(**kwargs) -> Tuple[dict, dict, dict]:
    """
    Load CE increase and validation losses data from json files.
    """
    CE_increase_filename, val_losses_filename, time_filename = _get_data_filenames(
        model_name=kwargs['model_name'],
        sae_path=kwargs['sae_path'],
        peft_layers=kwargs['peft_layers'],
        sae_from_hf=kwargs['sae_from_hf'],
        peft_rank=kwargs['peft_rank'],
        num_train_examples=kwargs['num_train_examples'],
        use_16_bit=kwargs['use_16_bit']
    )
    
    try:
        with open(CE_increase_filename, "r") as f:
            CE_increase = json.load(f)
            
        with open(val_losses_filename, "r") as f:
            val_losses_dict = json.load(f)
        
        with open(time_filename, "r") as f:
            total_training_minutes_dict = json.load(f)
            
        print(f"Loaded CE increase data from {CE_increase_filename}")
        print(f"Loaded validation losses from {val_losses_filename}")
        
        return CE_increase, val_losses_dict, total_training_minutes_dict
    
    except FileNotFoundError:
        print(f"Could not find data files. Creating new files...")
        return {}, {}, {}
    except json.JSONDecodeError:
        print(f"Error parsing JSON files. Creating new files...")
        return {}, {}, {}

def save_model(peft_model, rank, **kwargs) -> None:
    sae_path = kwargs['sae_path']
    model_name = kwargs['model_name']
    peft_layers = kwargs['peft_layers']
    peft_type = kwargs['peft_type']
    sae_from_hf = kwargs['sae_from_hf']

    if sae_path:
        if not sae_from_hf:
            sae_path = sae_path.split('/')[-2]
        else:
            parts = sae_path.split('/', 1)
            sae_path = f"{parts[1].replace('/', '_')}"
    
    model_name = model_name.split('/')[-1]
    
    peft_range = f"{peft_layers[0]}-{peft_layers[-1]}" if len(peft_layers) > 1 else peft_layers[0]
    
    base_path = f"saved_models/{model_name}"
    base_path = f"{base_path}/{sae_path}" if sae_path else f"{base_path}/base"
    save_dir = f"{base_path}/peft_{peft_range}"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"rank_{rank}")
    peft_model.save_pretrained(model_path)
    print(f"Saved model to {model_path}")









###############################################
### TRAIN AND EVALUATION FUNCTIONS ###
###############################################
@contextmanager
def wandb_session(project_name: str, run_name: str, config: Dict[str, Any]):
    """Context manager for wandb session"""
    try:
        wandb.init(project=project_name, name=run_name, config=config)
        yield
    finally:
        wandb.finish()

def evaluate(
    model: PreTrainedModel,
    val_dataset: List[torch.Tensor],
) -> float:
    """Evaluate model on validation dataset"""
    device = model.device
    model.eval()
    
    val_loss = 0
    total_examples = 0
    
    with torch.no_grad():
        val_loop = tqdm(val_dataset, leave=True, desc="Validation")
        try:
            for val_batch in val_loop:
                val_inputs = val_batch.to(device)
                val_targets = val_inputs.clone()
                batch_size = val_inputs.size(0)

                GlobalSAE.current_batch = val_inputs.detach()
                val_outputs = model(val_inputs, labels=val_targets)
                val_loss += val_outputs.loss.item() * batch_size
                total_examples += batch_size
                
                # Cleanup
                del val_inputs, val_targets, val_outputs
                torch.cuda.empty_cache()

                val_loop.set_description_str(f"Validation Loss: {val_loss/total_examples:.4f}")
        finally:
            val_loop.close()
            
    return val_loss / total_examples

def train_model(
    peft_model: PreTrainedModel,
    train_gen: Generator[str, None, None],
    val_dataset: List[torch.Tensor],
    args: Any,
    rank: int,
    project_name: str,
    run_name: str,
    initial_loss: Optional[float] = None,
    base_loss: Optional[float] = None,
    track_evals: bool = True,
) -> Tuple[List[float], List[float]]:
    """Train the model using KL divergence loss"""
    print("Training model with KL divergence loss")
    device = peft_model.device
    
    args_config = {
        attr: getattr(args, attr) 
        for attr in dir(args) 
        if not callable(getattr(args, attr)) and not attr.startswith("__")
    }
    
    train_losses = []
    val_losses = [initial_loss] if initial_loss is not None else []
    total_training_minutes_list = [0] if initial_loss is not None else []
    total_examples = 0
    
    total_training_minutes = 0

    with wandb_session(project_name, run_name, args_config):
        if initial_loss is not None:
            wandb.log({
                "examples_processed": 0,
                "val_loss": initial_loss,
                "base_loss": base_loss,
                "total_training_minutes": 0,
                "training_minutes_between_evals": 0
            })
        
        optimizer = optim.AdamW(peft_model.parameters(), lr=5e-5)
        
        peft_model.train()
        total_loss = 0
        examples_since_last_eval = 0
        
        train_loop = tqdm(desc="Training", total=args.num_train_examples)
        
        training_time_between_evals = 0
        
        try:
            while total_examples < args.num_train_examples:
                train_step_start = time.time()

                inputs = next(train_gen).to(device)
                batch_size = inputs.size(0)
                
                optimizer.zero_grad()
                
                GlobalSAE.current_batch = inputs.detach()
                with torch.no_grad():
                    GlobalSAE.use_sae = False
                    with peft_model.disable_adapter():
                        base_outputs = peft_model(inputs.to(peft_model.device))
                        base_logits = base_outputs.logits
                        base_probs = torch.nn.functional.softmax(base_logits, dim=-1).to(peft_model.device)
                
                GlobalSAE.use_sae = True
                peft_outputs = peft_model(inputs)
                peft_logits = peft_outputs.logits
                peft_log_probs = torch.nn.functional.log_softmax(peft_logits, dim=-1)
                
                # Calculate KL divergence loss
                loss = torch.nn.functional.kl_div(
                    peft_log_probs,
                    base_probs,
                    reduction='batchmean',
                    log_target=False
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item() * batch_size
                total_examples += batch_size
                examples_since_last_eval += batch_size
                
                # Clean up memory
                del inputs, base_outputs, base_logits, base_probs, peft_outputs, peft_logits, peft_log_probs, loss
                torch.cuda.empty_cache()
                
                training_time_between_evals += time.time() - train_step_start
                train_loop.update(batch_size)

                if track_evals and examples_since_last_eval >= args.examples_per_eval:
                    avg_train_loss = total_loss / examples_since_last_eval
                    val_loss = evaluate(peft_model, val_dataset)
                    
                    total_training_minutes += training_time_between_evals / 60
                    
                    train_losses.append(avg_train_loss)
                    val_losses.append(val_loss)
                    total_training_minutes_list.append(total_training_minutes)
                    
                    wandb.log({
                        "examples_processed": total_examples,
                        "train_loss": avg_train_loss,
                        "val_loss": val_loss,
                        "training_minutes_between_evals": training_time_between_evals / 60,
                        "total_training_minutes": total_training_minutes
                    })
                    
                    print(f"\nExamples: {total_examples}, Train Loss: {avg_train_loss:.4f}, "
                            f"Val Loss: {val_loss:.4f}, "
                            f"Training Minutes Between Evals: {training_time_between_evals / 60:.2f}, "
                            f"Total Training Minutes: {total_training_minutes:.2f}")
                    
                    # Reset counters
                    total_loss = 0
                    examples_since_last_eval = 0
                    training_time_between_evals = 0
        
        finally:
            train_loop.close()
        
        # Final evaluation
        if not track_evals:
            val_loss = evaluate(peft_model, val_dataset)
            total_training_minutes += training_time_between_evals / 60
            
            train_losses.append(total_loss / total_examples)
            val_losses.append(val_loss)
            total_training_minutes_list.append(total_training_minutes)
            
            wandb.log({
                "examples_processed": total_examples,
                "train_loss": total_loss / total_examples,
                "val_loss": val_loss,
                "training_minutes_between_evals": training_time_between_evals / 60,
                "total_training_minutes": total_training_minutes
            })
            
            print(f"Final Evaluation: Train Loss: {total_loss / total_examples:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Total Training Minutes: {total_training_minutes:.2f}")
    
    return val_losses, total_training_minutes_list
    

class GlobalSAE:
    current_batch = None
    use_sae = True


def get_sae_hook(sae_module, tokenizer,sae_from_hf=False):
    def sae_reconstruction_hook(module, input, output):
        if not GlobalSAE.use_sae:
            return output

        original_shape = output[0].shape
        output_tensor = output[0]
        
        original_outputs = output[0]

        if sae_from_hf:
            token_is_eos_or_bos = (GlobalSAE.current_batch == tokenizer.eos_token_id) | (GlobalSAE.current_batch == tokenizer.bos_token_id)

            flattened_output = output_tensor.flatten(end_dim=1)
            reconstructed_output = sae_module(flattened_output).to(dtype=flattened_output.dtype)

            reconstructed_output = reconstructed_output.reshape(original_shape)       

            # Create mask for non-special tokens
            token_mask = ~token_is_eos_or_bos.unsqueeze(-1)
            
            # Where mask is True (non-special tokens), use reconstructed output
            # Where mask is False (special tokens), use original output
            reconstructed_output = torch.where(token_mask, reconstructed_output, original_outputs)

        else:
            # Process all tokens through SAE as before
            flat_output = output_tensor.reshape(-1, original_shape[-1])
            reconstructed_output = sae_module(flat_output).sae_out
            reconstructed_output = reconstructed_output.reshape(original_shape)

        return (reconstructed_output,) + output[1:]

    return sae_reconstruction_hook

# Add more utility functions as needed...



