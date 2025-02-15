# %%
from sae_lens import SAE
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import torch

from tqdm import tqdm
import wandb
from eleuther_sae.sae.data import chunk_and_tokenize
from collections import OrderedDict
from typing import Generator, List, Tuple, Any

import torch.optim as optim
import time
from transformers import PreTrainedModel
import argparse


# %%

class Args:
    batch_size = 1
    ctx_len = 1024
    num_val_examples = 1_000
    examples_per_eval = 1000
    peft_rank = 64
    num_train_examples = 30_000
    device = "cuda:0"
    sae_jump = None
args = Args()

# %%


parser = argparse.ArgumentParser()
parser.add_argument("--sae_jump", type=int, default=1, choices=[1, 2, 3, 4, 6, 8, 12, 16])
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--dont_freeze_saes", action="store_true")
parser.add_argument("--dont_use_lora", action="store_true")
args_parsed = parser.parse_args()

sae_jump = args_parsed.sae_jump
device = args_parsed.device
freeze_saes = not args_parsed.dont_freeze_saes
use_lora = not args_parsed.dont_use_lora

if freeze_saes and not use_lora:
    raise ValueError("Cannot freeze SAEs and not use PEFT")

args.device = device


sae_indices = list(range(sae_jump, 32, sae_jump))
if sae_jump == 1:
    sae_indices = [0] + sae_indices

saes = []

for i in tqdm(sae_indices):
    saes.append(SAE.from_pretrained(
        release="llama_scope_lxr_32x",
        sae_id=f"l{i}r_32x",
        device=args.device))

# %%

MODEL = "meta-llama/Llama-3.1-8B"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": args.device},
    torch_dtype=torch.bfloat16,
)

# %%

dataset = load_dataset(
    "togethercomputer/RedPajama-Data-1T-Sample",
    split="train",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenized = chunk_and_tokenize(dataset, tokenizer, max_seq_len=args.ctx_len)

tokenized = tokenized.shuffle(seed=43)

train_data = tokenized.select(range(args.num_train_examples))
val_data = tokenized.select(range(args.num_train_examples, args.num_train_examples + args.num_val_examples))


# Create data generator
def data_generator():
    indices = list(range(0, len(train_data), args.batch_size))
    while True:
        for i in indices:
            batch = train_data[i:i+args.batch_size]
            inputs = torch.stack([batch['input_ids']]).to(args.device, dtype=torch.long)
            yield inputs

train_gen = data_generator()

# %%

eval_base_model = False

if eval_base_model:
    # Evaluate base model
    @torch.no_grad()
    def evaluate_base_model(model, val_dataset, device):
        model.eval()
        total_val_loss = 0
        total_examples = 0
        
        for val_batch in tqdm(val_dataset, desc="Evaluating base model", leave=False):
            val_batch = val_batch['input_ids'].to(device).unsqueeze(0)
            val_targets = val_batch.clone()
            batch_size = val_batch.size(0)
            
            outputs = model(val_batch, labels=val_targets)
            val_loss = outputs.loss
                
            total_val_loss += val_loss.item() * batch_size
            total_examples += batch_size
                
        avg_val_loss = total_val_loss / total_examples
        model.train()
        return avg_val_loss

    base_val_loss = evaluate_base_model(model, val_data, args.device)
    print(f"Base model validation loss: {base_val_loss:.4f}")



# %%

if freeze_saes:
    for sae, config, none in saes:
        for param in sae.parameters():
            param.requires_grad = False

# %%
if use_lora:
    # Add PEFT config
    from utils import get_target_modules
    target_modules = get_target_modules(MODEL, peft_layers=range(32), peft_type="both")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.peft_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
    )

    # Convert to PEFT model
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
else:
    peft_model = model
# %%

def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks = OrderedDict()
            remove_all_forward_hooks(child)

# Clear all hooks
remove_all_forward_hooks(peft_model)

class GlobalUseSae:
    use_sae = False
    current_batch = None

def get_sae_hook(sae):
    def sae_reconstruction_hook(module, input, output):
        if not GlobalUseSae.use_sae:
            return output
        original_shape = output[0].shape
        output_tensor = output[0]
        
        original_outputs = output[0]
        token_is_eos_or_bos = (GlobalUseSae.current_batch == tokenizer.eos_token_id) | (GlobalUseSae.current_batch == tokenizer.bos_token_id)

        flattened_output = output_tensor.flatten(end_dim=1)
        reconstructed_output = sae(flattened_output).to(dtype=flattened_output.dtype)

        reconstructed_output = reconstructed_output.reshape(original_shape)       

        # Create mask for non-special tokens
        token_mask = ~token_is_eos_or_bos.unsqueeze(-1)
        
        # Where mask is True (non-special tokens), use reconstructed output
        # Where mask is False (special tokens), use original output
        reconstructed_output = torch.where(token_mask, reconstructed_output, original_outputs)

        return (reconstructed_output,) + output[1:]
    return sae_reconstruction_hook

# Add SAE hooks
handles = []
for index, layer in enumerate(sae_indices):
    model_to_attach = peft_model
    while hasattr(model_to_attach, "model"):
        model_to_attach = model_to_attach.model
    handles.append(model_to_attach.layers[layer + 1].register_forward_hook(
        get_sae_hook(saes[index][0])
    ))

# %%

class NoopOrDisableAdapter:
    def __init__(self, model):
        self.model = model
    
    def __enter__(self):
        if use_lora:
            return self.model.disable_adapter()
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass



@torch.no_grad()
def evaluate_model(peft_model, val_dataset, device, disable_peft=False):
    """Evaluate model on validation set"""
    peft_model.eval()
    total_val_loss = 0
    total_examples = 0
    
    for val_batch in tqdm(val_dataset, desc="Evaluating", leave=False):
        val_batch = val_batch['input_ids'].to(device).unsqueeze(0)
        val_targets = val_batch.clone()
        GlobalUseSae.current_batch = val_batch.detach()
        batch_size = val_batch.size(0)
        
        GlobalUseSae.use_sae = True

        if disable_peft:
            with NoopOrDisableAdapter(peft_model):
                outputs = peft_model(val_batch, labels=val_targets)
        else:
            outputs = peft_model(val_batch, labels=val_targets)

        val_loss = outputs.loss
            
        total_val_loss += val_loss.item() * batch_size
        total_examples += batch_size
            
    avg_val_loss = total_val_loss / total_examples
    peft_model.train()
    return avg_val_loss

def train_model(
    peft_model: PreTrainedModel,
    train_gen: Generator[str, None, None],
    args: Any,
    rank: int,
    project_name: str,
    run_name: str,
    val_dataset: torch.Tensor,
    initial_val_loss: float
) -> Tuple[List[float], List[float]]:
    """Train the model and return validation losses"""
    # print(f"Training model with KL divergence")

    device = peft_model.device
    
    args_config = {
        attr: getattr(args, attr) 
        for attr in dir(args) 
        if not callable(getattr(args, attr)) and not attr.startswith("__")
    }
    
    total_examples = 0
    eval_every = 1000  # Evaluate every 1000 examples

    with wandb.init(project=project_name, name=f"{run_name}_rank_{rank}", config=args_config):
        wandb.log({"initial_val_loss": initial_val_loss})
        optimizer = optim.AdamW(peft_model.parameters(), lr=5e-5)
        
        peft_model.train()
        total_loss = 0
        examples_since_last_eval = eval_every
        
        train_loop = tqdm(desc="Training", total=args.num_train_examples)
        
        training_time_between_evals = 0
        
        try:
            while total_examples < args.num_train_examples:
                train_step_start = time.time()

                inputs = next(train_gen).to(device)[0]
                batch_size = inputs.size(0)
                
                optimizer.zero_grad()

                GlobalUseSae.current_batch = inputs.detach()
                with NoopOrDisableAdapter(peft_model):
                    with torch.no_grad():
                        GlobalUseSae.use_sae = False
                        base_outputs = peft_model(inputs)
                        base_logits = base_outputs.logits
                        base_probs = torch.nn.functional.softmax(base_logits, dim=-1)
                
                GlobalUseSae.use_sae = True
                peft_outputs = peft_model(inputs)

                peft_logits = peft_outputs.logits
                peft_log_probs = torch.nn.functional.log_softmax(peft_logits, dim=-1)
                
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

                wandb.log({"train_kl_divergence": loss.item()})

                # Clean up memory
                del inputs, base_outputs, base_logits, base_probs, peft_outputs, peft_logits, peft_log_probs, loss
                torch.cuda.empty_cache()
                
                training_time_between_evals += time.time() - train_step_start
                train_loop.update(batch_size)

                # Evaluate on validation set every eval_every examples
                if examples_since_last_eval >= eval_every:
                    val_loss = evaluate_model(peft_model, val_dataset, device)
                    wandb.log({
                        "val_ce_loss": val_loss,
                        "training_time": training_time_between_evals
                    })
                    # print(f"\nValidation loss: {val_loss:.4f}")
                    examples_since_last_eval = 0
                    training_time_between_evals = 0
        
        finally:
            train_loop.close()
            # Remove hooks after training
            for handle in handles:
                handle.remove()
        
    return None

initial_val_loss = evaluate_model(peft_model, val_data, args.device, disable_peft=True)

train_model(
    peft_model=peft_model,
    train_gen=train_gen,
    args=args,
    rank=args.peft_rank,
    project_name="llama_multiple_saes_2",
    run_name=f"llama_{args.peft_rank}_{sae_indices}_{'lora' if use_lora else 'no-lora'}_{'freeze-saes' if freeze_saes else 'no-freeze-saes'}",
    val_dataset=val_data,
    initial_val_loss=initial_val_loss
)

# %%