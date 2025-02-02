import argparse
import os
import json
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from eleuther_sae.sae import Sae
import lm_eval


class GemmaModelWithHook(PreTrainedModel):
    def __init__(
            self, 
            base_model, 
            tokenizer, 
            sae_path,
            hook_layer,
            peft_path=None, 
            device="cuda", 
            from_hf=False
        ):
        super().__init__(base_model.config)
        self.tokenizer = tokenizer
        self.model = base_model
        self.peft_path = peft_path
        self.sae_path = sae_path
        self.hook_layer = hook_layer
        self.from_hf = from_hf
        if sae_path is None:
            self.sae = None
        elif from_hf:
            from sae_lens import SAE
            print(f"Loading SAE from HF...")
            release = sae_path.split("/")[0]
            sae_id = "/".join(sae_path.split("/")[1:])
            self.sae = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=device
            )[0]
        else:
            self.sae = Sae.load_from_disk(sae_path, device=device) if sae_path else None

        # Load PEFT adapter if provided
        if peft_path:
            self.model = PeftModel.from_pretrained(self.model, peft_path).to(device)

        # Move the entire model to the specified device
        self.to(device)
        
        if sae_path:
            self._register_hook()

    def _register_hook(self):
        """Registers a forward hook for the sparse autoencoder."""
        def sae_hook(module, input, output):
            with torch.no_grad():
                # Get original shape and output tensor
                original_shape = output[0].shape
                output_tensor = output[0]

                if self.from_hf:
                    # Keep first token of each sequence unchanged
                    first_tokens = output_tensor[:, 0:1, :]  # Shape: (batch_size, 1, hidden_dim)
                    rest_tokens = output_tensor[:, 1:, :]    # Shape: (batch_size, seq_len-1, hidden_dim)
                    
                    # Process all tokens except first through SAE
                    flat_rest = rest_tokens.reshape(-1, original_shape[-1])
                    # Convert to float32 for SAE processing, then back to bfloat16
                    flat_rest = flat_rest.to(torch.float32)
                    reconstructed_rest = self.sae(flat_rest)
                    reconstructed_rest = reconstructed_rest.to(torch.bfloat16)
                    reconstructed_rest = reconstructed_rest.reshape(rest_tokens.shape)
                    
                    # Concatenate first tokens with reconstructed rest
                    reconstructed_output = torch.cat([first_tokens, reconstructed_rest], dim=1)
                else:
                    # Process all tokens through SAE as before
                    flat_output = output_tensor.reshape(-1, original_shape[-1])
                    flat_output = flat_output.to(torch.float32)
                    reconstructed_output = self.sae(flat_output).sae_out
                    reconstructed_output = reconstructed_output.to(torch.bfloat16)
                    reconstructed_output = reconstructed_output.reshape(original_shape)

                return (reconstructed_output,) + output[1:]

        # Replace with the appropriate layer name where the hook should be applied
        target_layer = self.model.model.model.layers[self.hook_layer] if self.peft_path else self.model.model.layers[self.hook_layer]
        target_layer.register_forward_hook(sae_hook)

    def generate(self, *args, **kwargs):
        """Delegate generation to the underlying model."""
        return self.model.generate(*args, **kwargs)

    def generate_text(self, prompt, max_length=50):
        """Modified to use the model's generate method directly"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def forward(self, *args, **kwargs):
        """Forward pass implementation required by PreTrainedModel"""
        return self.model(*args, **kwargs)

    def evaluate(self, inputs):
        """Evaluate on a batch of inputs."""
        return self.model(**inputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments related to running experiment")
    parser.add_argument("--device", type=int, required=True, help="CUDA device index")
    parser.add_argument("--model_type", type=str, required=True, choices=["peft", "base"], help="Model type: peft or base")
    parser.add_argument("--model_size", type=int, required=True, choices=[2, 9], help="Model size: 2b or 9b")
    parser.add_argument("--sae", action="store_true", help="Whether to use SAE")
    args = parser.parse_args()

    device = f"cuda:{args.device}"
    idx = 0 if args.model_size == 2 else 1

    sizes = [2, 9]
    widths = [16, 131]
    L0s = [82, 62]
    sae_layer = [12, 20]
    total_layers = [26, 42]

    size = sizes[idx]
    width = widths[idx]
    L0 = L0s[idx]
    layer = sae_layer[idx]
    total_layer = total_layers[idx]

    base_model_path = f"google/gemma-2-{size}b"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)

    print(f"Loaded base model: {base_model_path}")

    # Load PEFT adapter and sparse autoencoder
    peft_path = f"saved_models/gemma-2-{size}b/layer_{layer}_width_{width}k_average_l0_{L0}/peft_{layer+1}-{total_layer-1}/rank_64" if args.model_type == "peft" else None
    sae_path = f"gemma-scope-{size}b-pt-res/layer_{layer}/width_{width}k/average_l0_{L0}" if args.sae else None

    # Wrap the model
    custom_model = GemmaModelWithHook(
        base_model=base_model,
        tokenizer=tokenizer,
        sae_path=sae_path,
        peft_path=peft_path,
        hook_layer=layer,
        device=device,
        from_hf=True
    )

    # Create task dictionary for MMLU
    task_dict = lm_eval.tasks.get_task_dict(['mmlu', 'hellaswag', 'truthfulqa'])

    # Create HuggingFace model instance
    lm = lm_eval.models.huggingface.HFLM(
        pretrained=custom_model,
        tokenizer=tokenizer,
        device=device,
        batch_size=1  # Adjust based on your GPU memory
    )

    # Evaluate on MMLU
    results = lm_eval.evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        bootstrap_iters=100000,
        verbosity="DEBUG",  # Shows detailed progress
        write_out=True,     # Writes detailed results to disk
        log_samples=True    # Logs individual predictions
    )

    # Create results directory if it doesn't exist
    results_dir = "eval/benchmarks/results"
    os.makedirs(results_dir, exist_ok=True)
    sae_suffix = "sae" if args.sae else "no_sae"
    results_file = os.path.join(results_dir, base_model_path.split("/")[-1] + f"_{args.model_type}_{sae_suffix}.json")

    minimal_results = results.get("results", {})
    # Save results to JSON file
    with open(results_file, "w") as f:
        json.dump(minimal_results, f, indent=4)

    print(f"Results saved to {results_file}")
    print(results)

    del base_model
