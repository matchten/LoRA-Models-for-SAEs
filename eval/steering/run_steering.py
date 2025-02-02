import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import PeftModel, PeftConfig
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json

from tqdm import tqdm
from sae.sae import Sae
import scipy.stats


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, required=True, help="GPU device ID (0 or 1)")
parser.add_argument("--neuron", type=int, required=True, help="Neuron ID")
parser.add_argument("--tuning", action="store_true", help="Whether to run in tuning mode")
parser.add_argument("--alpha", type=float, required=True, help="Alpha value for steering")
args = parser.parse_args()

neuron = args.neuron
BATCH_SIZE = 1 # keep as 1 to avoid truncating or padding

RESULTS_DIR = f"eval/steering/LL/{neuron}"
LL_PLOT_DIR = f"eval/steering/plots/{neuron}"
DIST_PLOT_DIR = f"eval/steering/distributions/{neuron}"
DATASET_DIR = f"eval/steering/datasets"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LL_PLOT_DIR, exist_ok=True)
os.makedirs(DIST_PLOT_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

DATASETS = {
    "law": f"{DATASET_DIR}/law.json",
    "arabic": f"{DATASET_DIR}/arabic.json",
    "shakespeare": f"{DATASET_DIR}/shakespeare.json",
    "biology": f"{DATASET_DIR}/biology.json",
    "recipes": f"{DATASET_DIR}/recipes.json",
    "positive": f"{DATASET_DIR}/latent_{neuron}.json",
}

def get_dataset(dataset_type):
    dataset_path = DATASETS[dataset_type]
    with open(dataset_path, "r") as f:
        dataset = json.load(f)["examples"]
    
    return dataset


def LL(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=False).to(model.device)
    token_ids = inputs["input_ids"].squeeze(0)

    with torch.no_grad():
        logits = model(**inputs).logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs.squeeze(0) # Remove batch dimension
        
        # Calculate log-likelihood
        token_log_probs = log_probs[1:].gather(dim=-1, index=token_ids[1:].unsqueeze(-1)).squeeze(-1)
        log_likelihood = token_log_probs.mean().item() # normalize for seq length
        perplexity = np.exp(-log_likelihood)

        return log_likelihood

def normalize_likelihoods(base_LL, intervention_LL):
    base_tensor = torch.tensor(base_LL)
    intervention_tensor = torch.tensor(intervention_LL)
    
    shift = base_tensor.min()
    scale = base_tensor.max() - base_tensor.min()

    base_tensor = 100*(base_tensor - shift) / scale
    intervention_tensor = 100*(intervention_tensor - shift) / scale
    
    return base_tensor.tolist(), intervention_tensor.tolist()


def get_log_likelihoods(model, tokenizer, dataset):
    likelihoods = []
    for example in tqdm(dataset, desc="Computing log likelihoods"):
        likelihoods.append(LL(model, tokenizer, example))

    return likelihoods


def main(model_type, dataset_type, percentile, alpha):
    device_name = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch.cuda.set_device(device)

    ds = get_dataset(dataset_type)

    # Load model and tokenizer
    model_name = "google/gemma-2-2b"
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sae_path = "saved_saes/gemma-2-2b/normal/expansion_8_L0_64/model.layers.12"
    sae = Sae.load_from_disk(sae_path, device=device_name)
    sae_feature = sae.W_dec[neuron].detach().clone()
    del sae

    if model_type == "base":
        model = base_model
    elif model_type == "peft":
        peft_model_path = "saved_models/gemma-2-2b/expansion_8_L0_64/peft_13-25/rank_1"
        adapter_config = PeftConfig.from_pretrained(peft_model_path)
        model = PeftModel.from_pretrained(
            model=base_model, 
            model_id=peft_model_path, 
            config=adapter_config,
        )
    
    model.to(device)

    # Get log-likelihoods
    print("Getting Base Model Log-Likelihoods")
    base_LL = get_log_likelihoods(model, tokenizer, ds)

    def intervention_hook(module, inputs, outputs):
        # intervention on the SAE feature
        parallel_comp = torch.sum(outputs[0] * sae_feature, dim=-1, keepdim=True)
        intervention = outputs[0] + alpha * parallel_comp * sae_feature
        return (intervention,) + outputs[1:]

    if model_type == "peft":
        hook_handle = model.base_model.model.model.layers[12].register_forward_hook(intervention_hook)
    else:
        hook_handle = model.model.layers[12].register_forward_hook(intervention_hook)
    
    print("Getting Intervention Model Log-Likelihoods")
    intervention_LL = get_log_likelihoods(model, tokenizer, ds)
    hook_handle.remove()

    # Print worst examples
    print("\nWorst examples by base model likelihood:")
    print("=" * 50)
    pos_examples_with_ll = list(zip(ds, base_LL))
    sorted_pos = sorted(pos_examples_with_ll, key=lambda x: x[1])  # Sort by log likelihood
    
    # Print top 5 worst examples
    for i, (example, ll) in enumerate(sorted_pos[:10]):
        print(f"{i+1}. LL: {ll:.3f}")
        print(f"Text: {example}")
        print("-" * 50)
    
    # Print top 5 best examples
    print("\nBest examples by base model likelihood:")
    print("=" * 50)
    for i, (example, ll) in enumerate(sorted_pos[-10:]):
        print(f"{i+1}. LL: {ll:.3f}")
        print(f"Text: {example}")
        print("-" * 50)

    base_LL, intervention_LL = normalize_likelihoods(base_LL, intervention_LL)

    # Find samples with biggest increase from base to intervention for negative examples
    neg_changes = [(intervention_ll - base_ll, base_ll, intervention_ll, example) 
                   for base_ll, intervention_ll, example 
                   in zip(base_LL, intervention_LL, ds)]
    
    # Sort by change (first element of tuple) in descending order
    neg_changes.sort(reverse=True)
    
    print("\nExamples with largest increase in likelihood after intervention:")
    print("=" * 50)
    for i, (change, base_ll, int_ll, example) in enumerate(neg_changes[:10]):
        print(f"{i+1}. Change in LL: {change:.3f}")
        print(f"Base LL: {base_ll:.3f}")
        print(f"Intervention LL: {int_ll:.3f}") 
        print(f"Text: {example}")
        print("-" * 50)
    
    print("\nExamples with largest decrease in likelihood after intervention:")
    print("=" * 50)
    for i, (change, base_ll, int_ll, example) in enumerate(neg_changes[-10:]):
        print(f"{i+1}. Change in LL: {change:.3f}")
        print(f"Base LL: {base_ll:.3f}")
        print(f"Intervention LL: {int_ll:.3f}") 
        print(f"Text: {example}")
        print("-" * 50)

    LLs = sorted(zip(base_LL, intervention_LL), key=lambda x: x[0])

    # Take upper percentile of negative examples
    if dataset_type != "positive":
        neg_cutoff = int(len(LLs) * (1 - percentile))
        base_LL, intervention_LL = zip(*LLs[neg_cutoff:])
    
    # Take lower percentile of positive examples
    if dataset_type == "positive":
        pos_cutoff = int(len(LLs) * percentile)
        base_LL, intervention_LL = zip(*LLs[:pos_cutoff])

    # Save log-likelihood data to JSON
    ll_data = {
        "base_LL": base_LL,
        "intervention_LL": intervention_LL,
    }
    
    with open(f"{RESULTS_DIR}/{dataset_type}_{model_type}_{percentile}.json", "w") as f:
        json.dump(ll_data, f, indent=4)

    return base_LL, intervention_LL

##########
# plotting
##########
def stat_eval(base_pos_diff, peft_pos_diff, base_neg_diff, peft_neg_diff):
    # Get differences between PEFT and base model
    pos_diff = np.array(peft_pos_diff) - np.array(base_pos_diff) 
    neg_diff = np.array(peft_neg_diff) - np.array(base_neg_diff)
    
    # Calculate 90% confidence intervals
    confidence = 0.90
    pos_mean = np.mean(pos_diff)
    neg_mean = np.mean(neg_diff)
    
    # Calculate confidence intervals using standard error and t-distribution
    pos_ci = scipy.stats.t.interval(confidence, len(pos_diff)-1, 
                                  loc=pos_mean, 
                                  scale=scipy.stats.sem(pos_diff))
    neg_ci = scipy.stats.t.interval(confidence, len(neg_diff)-1, 
                                  loc=neg_mean, 
                                  scale=scipy.stats.sem(neg_diff))
    
    return pos_ci, neg_ci

def plot_distribution(base_pos_diff, peft_pos_diff, base_neg_diff, peft_neg_diff, base_dataset):
    # Define font sizes
    FONT_SIZES = {
        'title': 7.25,
        'axis_label': 7.25, 
        'legend_pos': 5.5,
        'legend_neg': 5.5,
        'ticks': 6,
        'shared_xlabel': 7.25
    }

    # Calculate t statistics and p values
    pos_ci, neg_ci = stat_eval(base_pos_diff, peft_pos_diff, base_neg_diff, peft_neg_diff)

    if base_dataset == "all":
        print("POSITIVE CI:", pos_ci)
        print("NEGATIVE CI:", neg_ci)

    # Create the figure and axis with academic styling
    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.25, 2))

    # Plot positive differences (base and peft)
    BINS = 40
    ax1.hist(base_pos_diff, color='blue', 
             label=f'Base Model', 
             alpha=0.5, bins=BINS, linewidth=0.5)
    ax1.hist(peft_pos_diff, color='green', 
             label='LoRA Model', alpha=0.5, bins=BINS, linewidth=0.5)
    
    ax1.set_title('Positive Examples', fontsize=FONT_SIZES['title'], pad=5)
    ax1.set_ylabel('Count', fontsize=FONT_SIZES['axis_label'])
    pos_legend_loc = 'upper left' if neuron == 13677 else 'upper right'
    ax1.legend(loc=pos_legend_loc,
              frameon=True, fancybox=False, edgecolor='black',
              fontsize=FONT_SIZES['legend_pos'], borderaxespad=0.3, borderpad=0.3,
              handletextpad=0.3)
    ax1.tick_params(labelsize=FONT_SIZES['ticks'], width=0.5, length=3, pad=0.5)

    # Plot negative differences (base and peft)
    ax2.hist(base_neg_diff, color='blue', 
             label=f'Base Model', 
             alpha=0.5, bins=BINS, linewidth=0.5)
    ax2.hist(peft_neg_diff, color='green', 
             label='LoRA Model', alpha=0.5, bins=BINS, linewidth=0.5)
    
    
    ax2.set_title('Negative Examples', fontsize=FONT_SIZES['title'], pad=5)
    ax2.tick_params(labelsize=FONT_SIZES['ticks'], width=0.5, length=3, pad=0.5)

    # Clean up axes
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

    # Add shared x-label
    fig.text(0.5, 0.02, 'Change in Normalized LL after Steering', ha='center', 
            fontsize=FONT_SIZES['shared_xlabel'])

    # Adjust layout
    plt.tight_layout(pad=0.05)
    plt.subplots_adjust(wspace=0.2, bottom=0.15)
    
    plt.savefig(f"{DIST_PLOT_DIR}/{base_dataset}.png", bbox_inches='tight', pad_inches=0.01)
    plt.savefig(f"{DIST_PLOT_DIR}/{base_dataset}.pdf", bbox_inches='tight', pad_inches=0.01)
    plt.close()

def get_diffs(base_LL_data, peft_LL_data):
    # Extract data
    base_pos_LL = np.array(base_LL_data["base_positive_ll"])
    base_neg_LL = np.array(base_LL_data["base_negative_ll"])
    base_intervention_pos_LL = np.array(base_LL_data["intervention_positive_ll"])
    base_intervention_neg_LL = np.array(base_LL_data["intervention_negative_ll"])
    
    peft_pos_LL = np.array(peft_LL_data["base_positive_ll"])
    peft_neg_LL = np.array(peft_LL_data["base_negative_ll"])
    peft_intervention_pos_LL = np.array(peft_LL_data["intervention_positive_ll"])
    peft_intervention_neg_LL = np.array(peft_LL_data["intervention_negative_ll"])

    # Calculate differences
    base_pos_diff = base_intervention_pos_LL - base_pos_LL
    base_neg_diff = base_intervention_neg_LL - base_neg_LL
    peft_pos_diff = peft_intervention_pos_LL - peft_pos_LL
    peft_neg_diff = peft_intervention_neg_LL - peft_neg_LL

    return base_pos_diff.tolist(), peft_pos_diff.tolist(), base_neg_diff.tolist(), peft_neg_diff.tolist()


if __name__ == "__main__":
    PERCENTILE = 1
    print("NEURON ID:", neuron)
    tuning = args.tuning
    alpha = args.alpha

    base_pos_diff = []
    peft_pos_diff = []
    base_neg_diff = []
    peft_neg_diff = []

    base_ds_types = ["law", "arabic", "shakespeare", "recipes"] if not tuning else ["biology"]

    base_pos_LL, base_pos_intervention_LL = main("base", "positive", PERCENTILE, alpha)
    peft_pos_LL, peft_pos_intervention_LL = main("peft", "positive", PERCENTILE, alpha)

    for i, dataset_type in enumerate(base_ds_types):
        print(f"({i+1}/{len(base_ds_types)}) DATASET:", dataset_type)
        print("=" * 50)
        base_neg_LL, base_neg_intervention_LL = main("base", dataset_type, PERCENTILE, alpha)
        peft_neg_LL, peft_neg_intervention_LL = main("peft", dataset_type, PERCENTILE, alpha)

        base_ll_data = {
            "base_positive_ll": base_pos_LL,
            "base_negative_ll": base_neg_LL,
            "intervention_positive_ll": base_pos_intervention_LL,
            "intervention_negative_ll": base_neg_intervention_LL,
        }

        peft_ll_data = {
            "base_positive_ll": peft_pos_LL,
            "base_negative_ll": peft_neg_LL,
            "intervention_positive_ll": peft_pos_intervention_LL,
            "intervention_negative_ll": peft_neg_intervention_LL,
        }

        base_pos_diff, peft_pos_diff, base_neg, peft_neg = get_diffs(base_ll_data, peft_ll_data)
        plot_distribution(base_pos_diff, peft_pos_diff, base_neg, peft_neg, dataset_type)
        base_neg_diff.extend(base_neg)
        peft_neg_diff.extend(peft_neg)

    if not tuning:
        print("ALL DATASETS")
        plot_distribution(base_pos_diff, peft_pos_diff, base_neg_diff, peft_neg_diff, "all")