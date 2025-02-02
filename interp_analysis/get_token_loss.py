# %%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from eleuther_sae.sae import Sae

torch.set_grad_enabled(False)
# %%

device = "cuda:0"
# device = "cpu"

model_name = "google/gemma-2-2b"
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device)
print(base_model.config)

# %%
peft_model_path = "saved_models/gemma-2-2b/expansion_8_L0_64/peft_0-25"

# %%

adapter_config = PeftConfig.from_pretrained(peft_model_path)
print(adapter_config)
peft_model = PeftModel.from_pretrained(
    model=base_model, 
    model_id=peft_model_path, 
    config=adapter_config,
)
peft_model = peft_model.to(device)
print(peft_model)

# %%


sae = Sae.load_from_disk(
    path = "saved_saes/gemma-2-2b/normal/expansion_8_L0_64/model.layers.12",
    device = device,
)
sae.eval()

# %%

from datasets import load_dataset
from itertools import islice

class args:
    batch_size = 1
    ctx_len = 1024
    num_epochs = 1
    batches_train_per_epoch = 15_000 
    batches_val_per_epoch = 400
    device = "cuda:0"
        

dataset_name = "monology/pile-uncopyrighted"

def hf_dataset_to_generator(dataset_name, split="train", streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    dataset = dataset.shuffle(seed=42, buffer_size=10000)

    if split == "train":
        train_gen = (x["text"] for x in iter(dataset))
        val_gen = (x["text"] for x in iter(dataset))
        
        val_data = list(islice(val_gen, 5 * args.batches_val_per_epoch * args.batch_size)) 
        
        def train():
            for item in train_gen:
                if item not in val_data:
                    yield item
        
        return train(), val_data
    else:
        return (x["text"] for x in iter(dataset))

train_gen, val_data = hf_dataset_to_generator(dataset_name)

def tokenized_batch(generator):
    batch = []
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
    return torch.cat([x["input_ids"] for x in batch], dim=0)


val_dataset = []
val_data_iter = iter(val_data)

while len(val_dataset) < args.batches_val_per_epoch:
    try:
        val_dataset.append(tokenized_batch(val_data_iter))
    except StopIteration:
        break
# %%

def calculate_manual_loss(output, batch, device):
    # Calculate loss manually
    logits = output.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = batch[..., 1:].to(device)
    
    # Calculate cross entropy loss
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

# %%

original_losses = []
sae_losses = []
peft_losses = []

def replace_sae_hook(module, input, output):
    flat_output = output[0].squeeze(0)

    # Skip the first token
    flat_output[1:] = sae(flat_output[1:]).sae_out
    reconstructed_output = flat_output.unsqueeze(0)
    return (reconstructed_output,) + output[1:]

# # Original Losses
with peft_model.disable_adapter():
    for batch in tqdm(val_dataset):
        output = peft_model(batch.to(peft_model.device), labels=batch)
        original_losses.append(calculate_manual_loss(output, batch, peft_model.device))


layer_12 = peft_model.model.model.layers[12]
handle = layer_12.register_forward_hook(replace_sae_hook)

# SAE Losses
with peft_model.disable_adapter():
    for batch in tqdm(val_dataset):
        output = peft_model(batch.to(peft_model.device), labels=batch)
        sae_losses.append(calculate_manual_loss(output, batch, peft_model.device))


# PEFT Losses
for batch in tqdm(val_dataset):
    output = peft_model(batch.to(peft_model.device), labels=batch)
    peft_losses.append(calculate_manual_loss(output, batch, peft_model.device))

handle.remove()

# %%

all_original_losses = torch.stack(original_losses)
all_sae_losses = torch.stack(sae_losses)
all_peft_losses = torch.stack(peft_losses)

# %%

os.makedirs("data/losses", exist_ok=True)
torch.save(all_original_losses, "data/losses/all_original_losses.pt")
torch.save(all_sae_losses, "data/losses/all_sae_losses.pt")
torch.save(all_peft_losses, "data/losses/all_peft_losses.pt")

# %%
stacked_tokens = torch.cat(val_dataset, dim=0)
torch.save(stacked_tokens, "data/losses/stacked_tokens.pt")

# %%