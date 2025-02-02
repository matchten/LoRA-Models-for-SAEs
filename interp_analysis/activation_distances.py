import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from peft import PeftModel, PeftConfig
import numpy as np
from datasets import load_dataset
from itertools import islice
import json

from eleuther_sae.sae import Sae

# Constants and configuration
MODEL_NAME = "google/gemma-2-2b"
DATASET_NAME = "monology/pile-uncopyrighted"
PEFT_MODEL_PATH = "saved_models/gemma-2-2b/expansion_8_L0_64/peft_0-25/rank_64"
SAE_PATH = "saved_saes/gemma-2-2b/normal/expansion_8_L0_64/model.layers.12"
BATCH_SIZE = 1
SAE_LAYER = 12

class ModelConfig:
    batch_size = 1
    ctx_len = 1024
    num_val_examples = 500
    device = "cuda:0"

def hf_dataset_to_generator(dataset_name, split="train", streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    dataset = dataset.shuffle(seed=42, buffer_size=10000)

    if split == "train":
        train_gen = (x["text"] for x in iter(dataset))
        val_gen = (x["text"] for x in iter(dataset))
        val_data = list(islice(val_gen, 5*ModelConfig.num_val_examples))
        
        def train():
            for item in train_gen:
                if item not in val_data:
                    yield item
        
        return train(), val_data
    return (x["text"] for x in iter(dataset))

def tokenized_batch(generator, tokenizer):
    batch = []
    while len(batch) < ModelConfig.batch_size:
        next_text = next(generator)
        tokenized = tokenizer(
            next_text,
            return_tensors="pt",
            max_length=ModelConfig.ctx_len,
            padding=False,
            truncation=True,
        )
        if tokenized["input_ids"].shape[1] == ModelConfig.ctx_len:
            batch.append(tokenized)
    return torch.cat([x["input_ids"] for x in batch], dim=0)

def compute_sae_activation_metrics(input_tensor, model_type, sae, base_model, peft_model):
    """Compute activation distances and cosine similarities between original and SAE-modified models."""
    model = base_model if model_type == "base" else peft_model
    n_layers = model.config.num_hidden_layers
    
    original_activations = []
    sae_activations = []
    
    def get_activation_hook(activation_list):
        def hook(module, input, output):
            activation_list.append(output[0].detach())
            return output
        return hook
    
    def sae_hook(module, input, output):
        with torch.no_grad():
            original_shape = output[0].shape
            output_tensor = output[0]
            flat_output = output_tensor.reshape(-1, original_shape[-1])
            reconstructed = sae(flat_output).sae_out
            reconstructed = reconstructed.reshape(original_shape)
            return (reconstructed.to(output_tensor.dtype),) + output[1:]
    
    # Collect original activations
    original_handles = [
        base_model.model.layers[i].register_forward_hook(get_activation_hook(original_activations))
        for i in range(n_layers)
    ]
    
    with torch.no_grad():
        base_model(input_tensor.to(ModelConfig.device))
    
    for h in original_handles:
        h.remove()
    
    # Apply SAE and collect modified activations
    if model_type == "base":
        sae_hook_handle = model.model.layers[SAE_LAYER].register_forward_hook(sae_hook)
        sae_handles = [
            model.model.layers[i].register_forward_hook(get_activation_hook(sae_activations))
            for i in range(n_layers)
        ]
    else:
        sae_hook_handle = model.base_model.model.model.layers[SAE_LAYER].register_forward_hook(sae_hook)
        sae_handles = [
            model.base_model.model.model.layers[i].register_forward_hook(get_activation_hook(sae_activations))
            for i in range(n_layers)
        ]
    
    with torch.no_grad():
        model(input_tensor.to(ModelConfig.device))
    
    sae_hook_handle.remove()
    for h in sae_handles:
        h.remove()
    
    # Compute metrics
    metrics = []
    for orig_act, sae_act in zip(original_activations, sae_activations):
        l2_dist = torch.norm(orig_act - sae_act, p=2, dim=-1).mean().item()
        orig_flat = orig_act.reshape(-1, orig_act.size(-1))
        sae_flat = sae_act.reshape(-1, sae_act.size(-1))
        cos_sim = torch.nn.functional.cosine_similarity(orig_flat, sae_flat, dim=1).mean().item()
        metrics.append((l2_dist, cos_sim))
    
    distances, cosine_sims = zip(*metrics)
    return list(distances), list(cosine_sims)

def main():
    # Initialize models
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    adapter_config = PeftConfig.from_pretrained(PEFT_MODEL_PATH)
    peft_model = PeftModel.from_pretrained(model, model_id=PEFT_MODEL_PATH, config=adapter_config)
    
    base_model.to(ModelConfig.device)
    peft_model.to(ModelConfig.device)

    sae = Sae.load_from_disk(SAE_PATH, device=ModelConfig.device)
    
    # Prepare validation data
    train_gen, val_data = hf_dataset_to_generator(DATASET_NAME)
    val_dataset = []
    val_data_iter = iter(val_data)
    
    num_batches = (ModelConfig.num_val_examples + ModelConfig.batch_size - 1) // ModelConfig.batch_size
    while len(val_dataset) < num_batches:
        try:
            val_dataset.append(tokenized_batch(val_data_iter, tokenizer))
        except StopIteration:
            break
    
    val_tensor = torch.cat(val_dataset, dim=0).to(ModelConfig.device)
    
    # Collect metrics
    all_metrics = {
        "base": {"distances": [], "cosine_sims": []},
        "peft": {"distances": [], "cosine_sims": []}
    }
    
    for model_type in ["base", "peft"]:
        print(f"Computing metrics for {model_type} model...")
        all_distances = []
        all_cosine_sims = []
        
        for i in range(0, val_tensor.size(0), BATCH_SIZE):
            batch = val_tensor[i:i + BATCH_SIZE]
            distances, cosine_sims = compute_sae_activation_metrics(
                batch, model_type, sae, base_model, peft_model
            )
            all_distances.append(distances)
            all_cosine_sims.append(cosine_sims)
            
            if i % (BATCH_SIZE * 4) == 0:
                print(f"Processed {i}/{val_tensor.size(0)} examples")
        
        all_metrics[model_type]["distances"] = np.array(all_distances)
        all_metrics[model_type]["cosine_sims"] = np.array(all_cosine_sims)
    
    # Save metrics
    metrics_dict = {
        model_type: {
            "distances": metrics["distances"].tolist(),
            "cosine_sims": metrics["cosine_sims"].tolist()
        }
        for model_type, metrics in all_metrics.items()
    }
    
    with open('interp_analysis/activation_metrics.json', 'w') as f:
        json.dump(metrics_dict, f)
    
    plot_metrics(all_metrics)

def plot_metrics(metrics):
    """Plot the activation metrics."""
    # Calculate differences between base and peft
    dist_diff = metrics["peft"]["distances"] - metrics["base"]["distances"]
    cos_diff = metrics["peft"]["cosine_sims"] - metrics["base"]["cosine_sims"]
    
    # Calculate statistics
    dist_means = np.mean(dist_diff, axis=0)[SAE_LAYER:]
    cos_means = np.mean(cos_diff, axis=0)[SAE_LAYER:]
    dist_stds = np.std(dist_diff, axis=0, ddof=1)[SAE_LAYER:]
    cos_stds = np.std(cos_diff, axis=0, ddof=1)[SAE_LAYER:]
    
    dist_se = dist_stds / np.sqrt(dist_diff.shape[0])
    cos_se = cos_stds / np.sqrt(cos_diff.shape[0])
    
    z = 1.645  # 90% confidence interval
    dist_lower = dist_means - z * dist_se
    dist_upper = dist_means + z * dist_se
    cos_lower = cos_means - z * cos_se
    cos_upper = cos_means + z * cos_se
    
    layers = np.arange(len(dist_means))
    
    # Plot settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 5,
        'axes.labelsize': 7.25,
        'axes.titlesize': 7.25,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 5.5,
        'figure.dpi': 300
    })
    
    # Plot L2 distances
    plot_metric(layers, dist_means, dist_lower, dist_upper, 
                'Number of Layers after SAE', 'Change in Distance',
                'plots/activation_interp_l2_diff.pdf')
    
    # Plot cosine similarities
    plot_metric(layers, cos_means, cos_lower, cos_upper,
                'Number of Layers after SAE', 'Change in Cos Similarity',
                'plots/activation_interp_cosine_diff.pdf')

def plot_metric(layers, means, lower, upper, xlabel, ylabel, save_path):
    """Helper function for plotting metrics."""
    plt.figure(figsize=(3.25, 1.8))
    line_color = "green"
    
    plt.plot(layers, means, 'o-', color=line_color,
             label='Rank 64', linewidth=1, markersize=1.5, alpha=0.75)
    plt.fill_between(layers, lower, upper, color=line_color, alpha=0.2)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle=':', alpha=0.65, linewidth=0.35)
    plt.grid(True, which='minor', linestyle=':', alpha=0.65, linewidth=0.35)
    plt.grid(True, which='major', linestyle=':', alpha=0.65, linewidth=0.35)
    
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.3)
    plt.gca().tick_params(axis='x', which='major', length=3, width=0.35, pad=1)
    plt.gca().tick_params(axis='x', which='minor', length=1.5, width=0.3, pad=1)
    plt.gca().tick_params(axis='y', which='major', length=3, width=0.35, pad=1)
    
    plt.legend(frameon=False)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.025, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
