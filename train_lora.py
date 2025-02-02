"""
Generalized script for training any our experiments.
"""

# Packages
import argparse
from typing import List
from inspect import signature
import os
from utils import get_sae_hook

import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from eleuther_sae.sae import Sae


from utils import (
    get_target_modules,
    get_peft_model_layers,
    initialize_auth, 
    hf_dataset_to_generator,
    save_data,
    load_data,
    save_model,
    train_model,
    evaluate,
    _get_data_filenames
)


# TRAINING ARGS: DO NOT CHANGE
class args:
    batch_size = 1
    ctx_len = 1024
    num_val_tokens = 1_000_000 #1_000_000
    examples_per_eval = 1000 #1000


def main(
    model_name: str, # Model name
    sae_path: str, # Path to pre-trained SAE
    sae_from_hf: bool, # Whether to load SAE from HF
    dataset: str, # HF dataset name
    experiment_name: str, # Name of experiment
    run_name: str, # Name of run
    sae_layer: int, # Layer of SAE to hook into
    peft_layers: List[int], # List of peft layers
    peft_type: str, # "attn", "mlp", "pre-mlp", "both", "gate", "up"
    num_train_examples: int,
    peft_rank: int = 64, # peft rank
    track_evals: bool = False, # Whether to track evals
    use_16_bit: bool = False,
    device: int = 0,
    save_model_file: bool = False,
    args: args = args
):
    kwargs = {
        key: value for key, value in locals().items() 
        if key in signature(main).parameters
    }

    device_name = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch.cuda.set_device(device)

    initialize_auth()

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if use_16_bit else torch.float32,
        attn_implementation='eager'
    ).to(device)
    print("Done loading models")

    print(f"Loaded {model_name} model and tokenizer:")
    # print(model)

    if sae_path:
        if sae_from_hf: # load in trained SAE from HF
            from sae_lens import SAE
            print(f"Loading SAE from HF...")
            release = sae_path.split("/")[0]
            sae_id = "/".join(sae_path.split("/")[1:])
            sae_module = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=device_name
            )[0]
            print(f"Loaded SAE from HF")
        else: # load in trained SAE from disk
            print(f"Loading SAE from disk...")
            sae_module = Sae.load_from_disk(sae_path, device=device_name)
            print(f"Loaded SAE from disk")
        
        sae_module.eval()
        sae_module.requires_grad_(False)
        print("sae module", sae_module)

    model.requires_grad_(False)
    
    _, val_dataset = hf_dataset_to_generator(dataset, tokenizer, args)
    print(f"Loaded {num_train_examples} training examples and {len(val_dataset)} validation examples")

    CE_increase, val_losses_dict, total_training_minutes_dict = load_data(**kwargs)
    print(f"Loaded CE_increase, val_losses, and total training minutes")

    CE_increase_filename = _get_data_filenames(
        model_name=model_name,
        sae_path=sae_path,
        peft_layers=peft_layers,
        peft_rank=peft_rank,
        sae_from_hf=sae_from_hf,
        num_train_examples=num_train_examples,
        use_16_bit=use_16_bit
    )[0]
    print(f"CE_increase_filename: {CE_increase_filename}")
    if os.path.exists(CE_increase_filename):
        print(f"CE_increase_filename already exists: {CE_increase_filename}")
        return
    print(f"CE_increase_filename does not exist: {CE_increase_filename}, starting training")

    train_gen, _ = hf_dataset_to_generator(dataset, tokenizer, args)
    
    target_modules = get_target_modules(model_name, peft_layers, peft_type)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=peft_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=target_modules,
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    peft_model = peft_model.to(device)


    print("BASE MODEL LOSS")
    base_loss = evaluate(model, val_dataset)
    print(f"Base loss: {base_loss:.4f}")

    hook_handle = None
    if sae_path:
        print(f"Registering SAE hook (rank {peft_rank})")
        model_layers = get_peft_model_layers(peft_model, model_name) # varies based on the model architecture
        hook_handle = model_layers[sae_layer].register_forward_hook(get_sae_hook(sae_module, tokenizer, sae_from_hf))

    print("INITIAL PEFT MODEL LOSS")
    initial_loss = evaluate(peft_model, val_dataset)
    print(f"Initial loss: {initial_loss:.4f}")

    val_losses, total_training_minutes = train_model(
        peft_model=peft_model,
        train_gen=train_gen,
        val_dataset=val_dataset,
        args=args,
        rank=peft_rank,
        project_name=experiment_name,
        run_name=run_name,
        initial_loss=initial_loss,
        base_loss=base_loss,
        track_evals=track_evals,
    )
    converged_loss = val_losses[-1]

    if hook_handle:
        hook_handle.remove()
    
    CE_increase = {
        "initial": initial_loss,
        "converged": converged_loss,
        "difference": initial_loss - converged_loss
    }
    val_losses_dict = val_losses
    total_training_minutes_dict = total_training_minutes

    print(f"Rank {peft_rank}:")
    print(f"  - Initial Loss: {initial_loss:.4f}")
    print(f"  - Converged Loss: {converged_loss:.4f}")
    print(f"  - Loss Increase: {CE_increase['difference']:.4f}")
    print(f"Updated and saved CE_increase data")

    save_data(CE_increase, val_losses_dict, total_training_minutes_dict, **kwargs)
    
    if save_model_file:
        save_model(peft_model, peft_rank, **kwargs)
            
    del peft_model, model, tokenizer
    if sae_path:
        del sae_module


if __name__ == "__main__":
    # Run Experiment Args
    parser = argparse.ArgumentParser(description="Arguments related to running experiment")
    parser.add_argument("--device", type=int, required=True, help="CUDA device index")
    parser.add_argument("--model_type", type=str, required=True, choices=["gemma", "llama"], help="Type of model")
    parser.add_argument("--sae_layer", type=int, default=12, help="SAE layer")
    parser.add_argument("--LoRA_layers", type=str, choices=["all", "after"], default="all", help="Which layers to apply LoRA to")
    parser.add_argument("--rank", type=int, required=True, help="LoRA rank")
    parser.add_argument("--save_model", action="store_true", help="Whether to save the model")
    parser.add_argument("--num_train_examples", type=int, help="Number of training examples", choices=[15_000, 30_000, 100_000], required=True)
    parser.add_argument("--checkpoint_percent", type=int, help="Train on a specific checkpoint percent. If None, train on all checkpoints")

    parsed_args = parser.parse_args()
    for arg_name, arg_value in vars(parsed_args).items():
        setattr(args, arg_name, arg_value)

    layer = args.sae_layer
    rank = args.rank
    percents = [args.checkpoint_percent] if args.checkpoint_percent else range(10, 101, 10)
    
    if args.model_type == "gemma":
        model_name = "google/gemma-2-2b"
        sae_path_template = "saved_saes/gemma-2-2b/normal/expansion_8_L0_64-{pct}pct/model.layers.{layer}"
        LoRA_layers = list(range(26)) if args.LoRA_layers == "all" else list(range(layer+1, 26))
    elif args.model_type == "llama":
        model_name = "meta-llama/Llama-3.2-1B"
        sae_path_template = "saved_saes/Llama-3.2-1B/normal/expansion_8_L0_64-{pct}pct/model.layers.{layer}"
        LoRA_layers = list(range(16)) if args.LoRA_layers == "all" else list(range(layer+1, 16))

    for pct in percents:
        sae_path = sae_path_template.format(pct=pct, layer=layer)

        if not os.path.exists(sae_path):
            print(f"No SAE found at {sae_path}")
            continue
            
        main(
            model_name=model_name,
            sae_path=sae_path,
            sae_from_hf=False,
            dataset="togethercomputer/RedPajama-Data-V2", 
            experiment_name=f"{args.model_type}_LoRA",
            run_name=f"layer_{layer}_rank_{rank}",
            sae_layer=layer,
            peft_layers=LoRA_layers,
            peft_type="both",
            peft_rank=rank,
            num_train_examples=args.num_train_examples,
            track_evals=True,
            use_16_bit=False,
            device=args.device,
            save_model_file=args.save_model
        )
