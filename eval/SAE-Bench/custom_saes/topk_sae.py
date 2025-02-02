import torch
import torch.nn as nn
import numpy as np
from torch import Tensor, nn


from typing import NamedTuple
from .utils import decoder_impl

from safetensors import safe_open
import json
import os

class EncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents."""

    top_indices: Tensor
    """Indices of the top-k features."""



class TopKSAE(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
    ):
        print("d_in:", d_in)
        print("d_sae:", d_sae)
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype: torch.dtype = torch.float32

    def select_topk(self, latents: Tensor) -> EncoderOutput:
        """Select the top-k latents."""
        return EncoderOutput(*latents.topk(64, sorted=False))
    
    def decode(self, acts: Tensor) -> Tensor:
        return acts @ self.W_dec + self.b_dec

    def encode(self, input_acts):
        pre_acts = (input_acts - self.b_dec) @ self.W_enc + self.b_enc
        acts = torch.relu(pre_acts)

        top_k_values, top_k_indices = torch.topk(acts, 64, dim=-1)
        
        # Create a mask with zeros
        mask = torch.zeros_like(acts, dtype=torch.bool)
        
        # Use scatter to set the mask at the top-k indices
        mask.scatter_(-1, top_k_indices, True)
        
        # Zero out elements not in the top-k
        output_tensor = torch.where(mask, acts, torch.tensor(0.0, dtype=acts.dtype, device=acts.device))

        return output_tensor

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

    # required as we have device and dtype class attributes
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Update the device and dtype attributes based on the first parameter
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)

        # Update device and dtype if they were provided
        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self


def load_topk_sae(path_to_params: str) -> TopKSAE:

    # pt_params = torch.load(path_to_params, map_location=torch.device("cpu"))
    # Load config
    cfg_path = os.path.join(path_to_params, "cfg.json")
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    # Find the safetensor file
    safetensor_file = next(f for f in os.listdir(path_to_params) if f.endswith('.safetensors'))
    safetensor_path = os.path.join(path_to_params, safetensor_file)
    
    # Load parameters from safetensors
    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        pt_params = {key: f.get_tensor(key) for key in f.keys()}

    # Print original keys for debugging
    print("Original keys in state_dict:", pt_params.keys())

    # Map old keys to new keys
    key_mapping = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "encoder.bias": "b_enc",
        "bias": "b_dec",
    }

    # Create a new dictionary with renamed keys
    renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}

    # due to the way torch uses nn.Linear, we need to transpose the weight matrices
    renamed_params["W_enc"] = renamed_params["W_enc"].T
    renamed_params["W_dec"] = renamed_params["W_dec"]

    # Print renamed keys for debugging
    print("Renamed keys in state_dict:", renamed_params.keys())
    for key in renamed_params.keys():
        print(key, renamed_params[key].shape)

    # Create the VanillaSAE model
    sae = TopKSAE(d_in=renamed_params["b_dec"].shape[0], d_sae=renamed_params["b_enc"].shape[0])

    sae.load_state_dict(renamed_params)

    return sae


if __name__ == "__main__":
    pass