import os
from typing import Any, Optional

import evals.core.main as core
import evals.scr_and_tpp.main as scr_and_tpp
import evals.sparse_probing.main as sparse_probing
import sae_bench_utils.general_utils as general_utils
import custom_saes.custom_sae_config as custom_sae_config
import custom_saes.topk_sae as topk_sae
from sae_bench_utils.sae_selection_utils import get_saes_from_regex
import custom_saes.run_all_evals_custom_saes as run_all_evals_custom_saes
import torch
from dotenv import load_dotenv

RANDOM_SEED = 42


# Note: Unlearning is not recommended for models with < 2B parameters and we recommend an instruct tuned model
# Unlearning will also require requesting permission for the WMDP dataset (see unlearning/README.md)
# Absorption not recommended for models < 2B parameters
# asyncio doesn't like notebooks, so autointerp must be ran using a python script



# Select your eval types here.
eval_types = [
    "absorption",
    "core",
    "autointerp",
    "scr",
    "tpp",
    "sparse_probing",
]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, required=True, help='CUDA device ID to use')
parser.add_argument('--MODEL_TYPE', type=str, required=True, choices=["peft", "base"], help='Model type to evaluate')
args = parser.parse_args()

device = general_utils.setup_environment(args.device)

model_name = "google/gemma-2-2b"
peft_state_dict = None
llm_batch_size = 1
dtype = "float32"


# If evaluating multiple SAEs on the same layer, set save_activations to True
# This will require at least 100GB of disk space


def get_sae(path_str: str, new_sae_key: str) -> Any:
    sae = topk_sae.load_topk_sae(path_str)
    sae = sae.to(device, dtype=general_utils.str_to_dtype(dtype))

    print(f"sae dtype: {sae.dtype}, device: {sae.device}")

    d_sae, d_in = sae.W_dec.data.shape

    assert d_sae >= d_in

    print(f"d_in: {d_in}, d_sae: {d_sae}")
    sae.cfg = custom_sae_config.CustomSAEConfig(
        model_name, d_in=d_in, d_sae=d_sae, hook_name=hook_name, hook_layer=hook_layer
    )

    # Core evals require us to specify the dtype. This must be a string that can be converted to a torch dtype using general_utils.str_to_dtype.
    sae.cfg.dtype = dtype

    sae.cfg.architecture = new_sae_key
    sae.cfg.training_tokens = 1_000_000_000
    unique_custom_sae_id = path_str.replace("/", "_").replace(".", "_")
    if peft_state_dict is not None:
        unique_custom_sae_id  = f"peft_{unique_custom_sae_id}"
    
    return unique_custom_sae_id, sae



# The following contains our current defined SAE types and the shapes to plot for each. Add your custom SAE as new_sae_key
new_sae_key = "lora"


MODEL_TYPE = args.MODEL_TYPE # "peft" or "base"
MODEL_NAME = model_name.split("/")[-1]

save_activations = False
hook_layer = 12
hook_name = f"blocks.{hook_layer}.hook_resid_post"

base_path = f"eval/SAE-Bench/{MODEL_NAME}_{MODEL_TYPE}_results"

output_folders = {
    "absorption": f"{base_path}/absorption",
    "autointerp": f"{base_path}/autointerp",
    "core": f"{base_path}/core",
    "scr": f"{base_path}",
    "tpp": f"{base_path}",
    "sparse_probing": f"{base_path}/sparse_probing",
    "unlearning": f"{base_path}/unlearning",
}



# Note: the custom_sae_id should be unique, as it is used for the intermediate results and final results file names
unique_id1, sae1 = get_sae("/home/mattchen/urop_fall_2024/saved_saes/gemma-2b/normal/RedPajama-Data-1T-Sample_gemma_L0_64_expansion_8_layer_12_normal/model.layers.12", new_sae_key)
print("sae1 loaded", unique_id1)

RANK = 64
peft_state_dict = f"eval/SAE-Bench/gemma_peft_0-25_rank_{RANK}.pt" if MODEL_TYPE == "peft" else None

print("peft_state_dict loaded:", peft_state_dict)

load_dotenv('/home/mattchen/urop_fall_2024/.env', override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

# list of tuple of (sae_id, sae object)
custom_saes = [(unique_id1, sae1)]
selected_saes = custom_saes

_ = run_all_evals_custom_saes.run_evals(
    model_name,
    selected_saes,
    llm_batch_size,
    dtype,
    device,
    eval_types,
    api_key=openai_api_key,
    force_rerun=False,
    save_activations=save_activations,
    peft_state_dict=peft_state_dict,
    output_folders=output_folders,
)

print(f"RESUTLS FOR {MODEL_TYPE}")
