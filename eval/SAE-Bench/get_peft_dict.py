from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
import torch

# Load PEFT configuration
RANK = 64
peft_path = f"saved_models/gemma-2-2b/expansion_8_L0_64/peft_0-25/rank_{RANK}"
config = PeftConfig.from_pretrained(peft_path)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

# Load fine-tuned model
peft_model = PeftModel.from_pretrained(base_model, peft_path)

# Merge LoRA weights with base model weights
peft_model.merge_and_unload()

# Get the full state dictionary
base_state_dict = base_model.state_dict()
state_dict = peft_model.state_dict()

adapted_state_dict = {}
for key, value in state_dict.items():
    # Example: modify key names if necessary
    adapted_key = key.replace("base_model.model.", "")
    adapted_state_dict[adapted_key] = value

print(adapted_state_dict.keys())

# Save the adapted state dictionary to a .pt file
file_path = f"eval/SAE-Bench/gemma_peft_0-25_rank_{RANK}.pt"
torch.save(adapted_state_dict, file_path)
print(f"Saved adapted state dictionary to {file_path}")
