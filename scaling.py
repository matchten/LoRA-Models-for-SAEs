import argparse
from train_lora import main

# TRAINING ARGS: DO NOT CHANGE
class args:
    batch_size = 1
    ctx_len = 1024
    num_val_tokens = 1_000_000 #1_000_000
    examples_per_eval = 1000 #1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments related to running experiment")
    parser.add_argument("--device", type=int, required=True, help="CUDA device index")
    parser.add_argument("--num_train_examples", type=int, help="Number of training examples", choices=[15_000, 30_000, 100_000], required=True)
    parser.add_argument("--specific", type=str, help="Format: [l0/width/layer]=[val]" )
    parser.add_argument("--save_model", action="store_true", help="Whether to save the model")
    parsed_args = parser.parse_args()

    for arg_name, arg_value in vars(parsed_args).items():
        setattr(args, arg_name, arg_value)
    
    def create_config(model_name, layer, width, L0, peft_rank, total_layers=26):
        """Helper function to create a standard config with given parameters."""
        model_size = model_name.split("-")[-1]  # 2b, 9b, or 27b
        hf_path = f"gemma-scope-{model_size}-pt-res/layer_{layer}/width_{width}k/average_l0_{L0}"
        
        return {
            "model_name": f"google/gemma-2-{model_size}",
            "sae_path": hf_path,
            "sae_from_hf": True,
            "dataset": "monology/pile-uncopyrighted",
            "experiment_name": f"Scaling_Laws",
            "run_name": f"{model_size}_layer_{layer}_width_{width}k_L0_{L0}" + (f"_rank_{peft_rank}" if peft_rank else ""),
            "sae_layer": layer,
            "peft_layers": list(range(layer+1, total_layers)),
            "peft_type": "both",
            "peft_rank": peft_rank,
            "track_evals": False,
            "use_16_bit": True,
            "num_train_examples": args.num_train_examples,
            "device": args.device,
            "save_model_file": args.save_model,
            "args": args
        }

    configs = []

    # diff sparsity
    base_layer, base_width = 12, 16
    for L0 in [22, 41, 82, 176, 445]:
        if args.specific and f"l0={L0}" != args.specific:
            continue
        for peft_rank in [1, 4, 16, 64, 256]:
            configs.append(create_config("2b", base_layer, base_width, L0, peft_rank))

    # diff widths
    widths_L0s = zip([32, 65, 131, 262, 524], [76, 72, 67, 67, 65])
    for width, L0 in widths_L0s:
        if args.specific and f"width={width}" != args.specific:
            continue
        for peft_rank in [1, 4, 16, 64, 256]:
            configs.append(create_config("2b", base_layer, width, L0, peft_rank))

    # different layers
    layers_L0s = zip([6, 9, 15, 18], [70, 73, 78, 74])
    for layer, L0 in layers_L0s:
        if args.specific and f"layer={layer}" != args.specific:
            continue
        for peft_rank in [1, 4, 16, 64, 256]:
            configs.append(create_config("2b", layer, base_width, L0, peft_rank))

    # one layer at a time
    for peft_layer in range(base_layer+1, 26):
        config = create_config("2b", base_layer, base_width, 82, 64)
        config["peft_layers"] = [peft_layer]
        config["run_name"] = f"2b_layer_{base_layer}_width_{base_width}k_L0_82_LoRA-layer_{peft_layer}_rank_64"
        config["device"] = args.device
        configs.append(config)

    # larger models
    model_configs = [
        ("9b", 20, 131, 62, 42),
        ("27b", 22, 131, 82, 46)
    ]
    for model_size, layer, width, L0, total_layers in model_configs:
        for peft_rank in [1, 4, 16, 64, 256]:
            configs.append(create_config(model_size, layer, width, L0, peft_rank, total_layers))

    for config in configs:
        main(**config)