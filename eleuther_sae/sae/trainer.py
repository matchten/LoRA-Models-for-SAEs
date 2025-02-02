from collections import defaultdict
from dataclasses import asdict
from typing import Sized
import time

import torch
import torch.distributed as dist
from datasets import Dataset as HfDataset
from fnmatch import fnmatchcase
from natsort import natsorted
from safetensors.torch import load_model
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, get_linear_schedule_with_warmup

from .config import TrainConfig
from .data import MemmapDataset
from .sae import Sae
from .utils import geometric_median, get_layer_list, resolve_widths


class SaeTrainer:
    def __init__(
        self, 
        cfg: TrainConfig, 
        train_dataset: HfDataset | MemmapDataset, 
        val_dataset: HfDataset | MemmapDataset,
        model: PreTrainedModel,
        eval_frequency_rate: float = 0.01
    ):
        assert 0 < eval_frequency_rate <= 1, "eval_frequency_rate must be between 0 and 1"
        self.eval_frequency_rate = eval_frequency_rate
        
        if cfg.hookpoints:
            assert not cfg.layers, "Cannot specify both `hookpoints` and `layers`."

            # Replace wildcard patterns
            raw_hookpoints = []
            for name, _ in model.named_modules():
                if any(fnmatchcase(name, pat) for pat in cfg.hookpoints):
                    raw_hookpoints.append(name)

            # Natural sort to impose a consistent order
            cfg.hookpoints = natsorted(raw_hookpoints)
        else:
            # If no layers are specified, train on all of them
            if not cfg.layers:
                N = model.config.num_hidden_layers
                cfg.layers = list(range(0, N, cfg.layer_stride))

            # Now convert layers to hookpoints
            layers_name, _ = get_layer_list(model)
            cfg.hookpoints = [f"{layers_name}.{i}" for i in cfg.layers]

        self.cfg = cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.distribute_modules()

        N = len(cfg.hookpoints)
        assert isinstance(train_dataset, Sized)
        assert isinstance(val_dataset, Sized)

        num_train_examples = len(train_dataset)
        num_val_examples = len(val_dataset)

        device = model.device
        input_widths = resolve_widths(model, cfg.hookpoints)
        unique_widths = set(input_widths.values())

        if cfg.distribute_modules and len(unique_widths) > 1:
            # dist.all_to_all requires tensors to have the same shape across ranks
            raise ValueError(
                f"All modules must output tensors of the same shape when using "
                f"`distribute_modules=True`, got {unique_widths}"
            )

        self.model = model
        model_dtype = next(model.parameters()).dtype
        if cfg.initialize_sae is not None:
            assert len(cfg.layers) == 1, "Can only initialize one SAE at a time for now, may change later"
            sae_path = cfg.initialize_sae
            print(f"Loading SAE from '{sae_path}'")
            self.saes = {
                hook: Sae.load_from_disk(sae_path, device=device)
                for hook in self.cfg.hookpoints
            }
        else:
            self.saes = {
            hook: Sae(input_widths[hook], cfg.sae, device, dtype=model_dtype)
            for hook in self.local_hookpoints()
        }

        pgs = [
            {
                "params": sae.parameters(),
                # Auto-select LR using 1 / sqrt(d) scaling law from Fig 3 of the paper
                "lr": cfg.lr or 2e-4 / (sae.num_latents / (2**14)) ** 0.5
            }
            for sae in self.saes.values()
        ]

        
        # Dedup the learning rates we're using, sort them, round to 2 decimal places
        lrs = [f"{lr:.2e}" for lr in sorted(set(pg["lr"] for pg in pgs))]
        print(f"Learning rates: {lrs}" if len(lrs) > 1 else f"Learning rate: {lrs[0]}")

        try:
            from bitsandbytes.optim import Adam8bit as Adam

            print("Using 8-bit Adam from bitsandbytes")
        except ImportError:
            from torch.optim import Adam

            print("bitsandbytes 8-bit Adam not available, using torch.optim.Adam")
            print("Run `pip install bitsandbytes` for less memory usage.")

        self.global_step = 0
        self.num_tokens_since_fired = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.long)
            for name, sae in self.saes.items()
        }
        self.optimizer = Adam(pgs)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, cfg.lr_warmup_steps, num_train_examples // cfg.batch_size
        )

        # Store these as instance variables
        self.name_to_module = {
            name: self.model.get_submodule(name) for name in self.cfg.hookpoints
        }
        self.module_to_name = {v: k for k, v in self.name_to_module.items()}
        self.hidden_dict: dict[str, Tensor] = {}
        
        # Register hooks once during initialization
        self.handles = []
        self.register_hooks()

    def load_state(self, path: str):
        """Load the trainer state from disk."""
        device = self.model.device

        # Load the train state first so we can print the step number
        train_state = torch.load(f"{path}/state.pt", map_location=device, weights_only=True)
        self.global_step = train_state["global_step"]
        self.num_tokens_since_fired = train_state["num_tokens_since_fired"]

        print(f"\033[92mResuming training at step {self.global_step} from '{path}'\033[0m")

        lr_state = torch.load(f"{path}/lr_scheduler.pt", map_location=device, weights_only=True)
        opt_state = torch.load(f"{path}/optimizer.pt", map_location=device, weights_only=True)
        self.optimizer.load_state_dict(opt_state)
        self.lr_scheduler.load_state_dict(lr_state)

        for name, sae in self.saes.items():
            load_model(sae, f"{path}/{name}/sae.safetensors", device=str(device))

    def evaluate(self) -> float:
        """Evaluate the model on the validation set."""
        if not self.val_dataset:
            return 0.0
            
        self.model.eval()
        device = self.model.device
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size // self.cfg.grad_acc_steps,
            shuffle=False,
        )
        
        total_loss = 0.0
        total_samples = 0
        
        # Add tqdm progress bar
        pbar = tqdm(val_loader, desc="Evaluating", leave=False)
        
        with torch.no_grad():
            for batch in pbar:
                inputs = batch["input_ids"].to(device)
                labels = inputs.clone()
                outputs = self.model(input_ids=inputs, labels=labels)
                loss = outputs.loss
                
                # Scale loss by batch size for proper averaging
                batch_loss = loss.item() * inputs.size(0)
                total_loss += batch_loss
                total_samples += inputs.size(0)
                
                # Update progress bar with current loss
                pbar.set_postfix({"loss": f"{batch_loss/inputs.size(0):.4f}"})

                del outputs, loss, inputs, labels
                torch.cuda.empty_cache()
        
        validation_loss = total_loss / total_samples
        print(f"Validation loss: {validation_loss:.4f}")
        return validation_loss


    def fit(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        # Calculate total number of examples and checkpoint points
        total_examples = len(self.train_dataset)
        checkpoint_points = [int(total_examples * p) for p in [p/100 for p in range(0, 101, 10)]]
        examples_processed = self.global_step * self.cfg.batch_size
        next_checkpoint_idx = next(i for i, p in enumerate(checkpoint_points) if p > examples_processed)

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        ddp = dist.is_initialized() and not self.cfg.distribute_modules

        if self.cfg.log_to_wandb and rank_zero:
            try:
                import wandb
                
                # Modified to resume existing run if we have a run_id
                if hasattr(self.cfg, 'wandb_run_id') and self.cfg.wandb_run_id:
                    wandb.init(
                        id=self.cfg.wandb_run_id,
                        resume="must",
                        project=self.cfg.wandb_project or "sae",
                    )
                else:
                    wandb.init(
                        name=self.cfg.run_name,
                        project=self.cfg.wandb_project or "sae",
                        config=asdict(self.cfg),
                        save_code=True,
                    )
            except ImportError:
                print("Weights & Biases not installed, skipping logging.")
                self.cfg.log_to_wandb = False

        num_sae_params = sum(
            p.numel() for s in self.saes.values() for p in s.parameters()
        )
        num_model_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of model parameters: {num_model_params:_}")
        print(f"Number of SAE parameters: {num_sae_params:_}")
        num_model_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of trainable model parameters: {num_model_trainable:_}")
        num_trainable = sum(p.numel() for pg in self.optimizer.param_groups for p in pg["params"])
        print(f"Total number of trainable parameters: {num_trainable:_}")

        num_batches = len(self.train_dataset) // self.cfg.batch_size
        if self.global_step > 0:
            assert hasattr(self.train_dataset, "select"), "Dataset must implement `select`"
            n = self.global_step * self.cfg.batch_size
            print(f"Resuming training from example {n}/{len(self.train_dataset)} (step {self.global_step})")
            ds = self.train_dataset.select(range(n, len(self.train_dataset)))  # type: ignore
        else:
            ds = self.train_dataset

        device = self.model.device
        dl = DataLoader(
            ds, # type: ignore
            batch_size=self.cfg.batch_size,
            # NOTE: We do not shuffle here for reproducibility; the dataset should
            # be shuffled before passing it to the trainer.
            shuffle=False,
        )
        pbar = tqdm(
            desc="Training", 
            disable=not rank_zero, 
            initial=self.global_step, 
            total=num_batches,
        )

        did_fire = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.bool)
            for name, sae in self.saes.items()
        }
        num_tokens_in_step = 0

        # For logging purposes
        avg_auxk_loss = defaultdict(float)
        avg_fvu = defaultdict(float)
        avg_multi_topk_fvu = defaultdict(float)
        avg_model_loss = 0.0

        hidden_dict: dict[str, Tensor] = {}
        name_to_module = {
            name: self.model.get_submodule(name) for name in self.cfg.hookpoints
        }
        maybe_wrapped: dict[str, DDP] | dict[str, Sae] = {}
        module_to_name = {v: k for k, v in name_to_module.items()}

        def hook(module: nn.Module, _, outputs):
            # Maybe unpack tuple outputs
            if isinstance(outputs, tuple):
                output = outputs[0]

            name = module_to_name[module]
            hidden_dict[name] = output.flatten(0, 1)

            if name in self.saes:
                sae_out = self.saes[name](hidden_dict[name])
                sae_out = sae_out.sae_out.view_as(output)
                return (sae_out,) + outputs[1:]
            else:
                return outputs

        # Keeping track of evaluation and data logging
        training_start_time = time.time()
        total_training_time = 0  # New variable to track cumulative training time
        training_time_between_evals = 0

        examples_per_eval = int(total_examples * self.eval_frequency_rate)
        next_eval_at = examples_per_eval

        # Evaluate before training starts
        if self.global_step == 0:
            print("Evaluating initial model + SAE...")
            initial_val_loss = self.evaluate()
            if self.cfg.log_to_wandb and rank_zero:
                wandb.log({
                    "val_loss": initial_val_loss,
                    "training_minutes_between_evals": 0,
                    "total_training_minutes": 0,
                }, step=0)

        for batch in dl:
            train_step_start = time.time()
            self.hidden_dict.clear()

            # Bookkeeping for dead feature detection
            num_tokens_in_step += batch["input_ids"].numel()

            # Forward pass without autocast
            if self.cfg.e2e:
                inputs = batch["input_ids"].to(device)
                
                # First get original model output without SAEs
                self.remove_hooks()
                with torch.no_grad():
                    original_outputs = self.model(input_ids=inputs)
                    original_logits = original_outputs.logits
                self.register_hooks()
                
                # Then get output with SAEs
                outputs = self.model(input_ids=inputs)
                sae_logits = outputs.logits
                
                # Calculate KL divergence
                model_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(sae_logits, dim=-1),
                    torch.nn.functional.softmax(original_logits, dim=-1),
                    reduction='batchmean'
                )
            else:
                with torch.no_grad():
                    self.model(batch["input_ids"].to(device))

            if self.cfg.distribute_modules:
                self.hidden_dict = self.scatter_hiddens(hidden_dict)

            for name, hiddens in self.hidden_dict.items():
                raw = self.saes[name]  # 'raw' never has a DDP wrapper

                # On the first iteration, initialize the decoder bias
                if self.global_step == 0:
                    # NOTE: The all-cat here could conceivably cause an OOM in some
                    # cases, but it's unlikely to be a problem with small world sizes.
                    # We could avoid this by "approximating" the geometric median
                    # across all ranks with the mean (median?) of the geometric medians
                    # on each rank. Not clear if that would hurt performance.
                    median = geometric_median(self.maybe_all_cat(hiddens))
                    raw.b_dec.data = median.to(raw.dtype)

                if not maybe_wrapped:
                    # Wrap the SAEs with Distributed Data Parallel. We have to do this
                    # after we set the decoder bias, otherwise DDP will not register
                    # gradients flowing to the bias after the first step.
                    maybe_wrapped = (
                        {
                            name: DDP(sae, device_ids=[dist.get_rank()])
                            for name, sae in self.saes.items()
                        }
                        if ddp
                        else self.saes
                    )

                # Make sure the W_dec is still unit-norm
                if raw.cfg.normalize_decoder:
                    raw.set_decoder_norm_to_unit_norm()

                acc_steps = self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
                denom = acc_steps * self.cfg.wandb_log_frequency
                wrapped = maybe_wrapped[name]

                # Save memory by chunking the activations
                for chunk in hiddens.chunk(self.cfg.micro_acc_steps):
                    if self.cfg.e2e:
                        # Use model_loss here instead of doing backward earlier
                        loss = model_loss
                        loss.div(acc_steps).backward()
                    else:
                        # Original SAE training logic
                        out = wrapped(
                            chunk,
                            dead_mask=(
                                self.num_tokens_since_fired[name]
                                > self.cfg.dead_feature_threshold
                                if self.cfg.auxk_alpha > 0
                                else None
                            ),
                        )

                        avg_fvu[name] += float(
                            self.maybe_all_reduce(out.fvu.detach()) / denom
                        )
                        if self.cfg.auxk_alpha > 0:
                            avg_auxk_loss[name] += float(
                                self.maybe_all_reduce(out.auxk_loss.detach()) / denom
                            )
                        if self.cfg.sae.multi_topk:
                            avg_multi_topk_fvu[name] += float(
                                self.maybe_all_reduce(out.multi_topk_fvu.detach()) / denom
                            )

                        loss = out.fvu + self.cfg.auxk_alpha * out.auxk_loss + out.multi_topk_fvu / 8
                        loss.div(acc_steps).backward()

                        # Update the did_fire mask
                        did_fire[name][out.latent_indices.flatten()] = True
                        self.maybe_all_reduce(did_fire[name], "max")  # max is boolean "any"

                # Clip gradient norm independently for each SAE
                torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)

            # Check if we need to actually do a training step
            step, substep = divmod(self.global_step + 1, self.cfg.grad_acc_steps)
            if substep == 0:
                if self.cfg.sae.normalize_decoder:
                    for sae in self.saes.values():
                        sae.remove_gradient_parallel_to_decoder_directions()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

                ###############
                with torch.no_grad():
                    # Update the dead feature mask
                    for name, counts in self.num_tokens_since_fired.items():
                        counts += num_tokens_in_step
                        counts[did_fire[name]] = 0

                    # Reset stats for this step
                    num_tokens_in_step = 0
                    for mask in did_fire.values():
                        mask.zero_()
                
                # Update examples processed and check for checkpoints
                examples_processed += self.cfg.batch_size
                if next_checkpoint_idx < len(checkpoint_points) and examples_processed >= checkpoint_points[next_checkpoint_idx]:
                    checkpoint_pct = int((next_checkpoint_idx) * 5)
                    self.save(checkpoint_suffix=f"{checkpoint_pct}pct")
                    next_checkpoint_idx += 1

                if examples_processed >= next_eval_at:
                    # Add the training time from this step before evaluation
                    training_time_between_evals += time.time() - train_step_start
                    total_training_time += training_time_between_evals
                    
                    # Evaluate (evaluation time not counted)
                    eval_start = time.time()
                    val_loss = self.evaluate()
                    
                    if self.cfg.log_to_wandb and rank_zero:
                        normalized_step = step * self.cfg.batch_size
                        wandb.log({
                            "val_loss": val_loss,
                            "training_minutes_between_evals": training_time_between_evals / 60,
                            "total_training_minutes": total_training_time / 60,
                        }, step=normalized_step)
                    
                    training_time_between_evals = 0
                    next_eval_at = examples_processed + examples_per_eval
                    
                    # Resume tracking training time from after evaluation
                    train_step_start = time.time()
                else:
                    training_time_between_evals += time.time() - train_step_start

                if (
                    self.cfg.log_to_wandb
                    and (step + 1) % self.cfg.wandb_log_frequency == 0
                ):
                    info = {}

                    if self.cfg.e2e:
                        info["train_loss"] = loss.item()

                    for name in self.saes:
                        mask = (
                            self.num_tokens_since_fired[name]
                            > self.cfg.dead_feature_threshold
                        )

                        info.update(
                            {
                                f"fvu/{name}": avg_fvu[name],
                                f"dead_pct/{name}": mask.mean(
                                    dtype=torch.float32
                                ).item(),
                            }
                        )
                        if self.cfg.auxk_alpha > 0:
                            info[f"auxk/{name}"] = avg_auxk_loss[name]
                        if self.cfg.sae.multi_topk:
                            info[f"multi_topk_fvu/{name}"] = avg_multi_topk_fvu[name]

                    avg_auxk_loss.clear()
                    avg_fvu.clear()
                    avg_multi_topk_fvu.clear()
                    avg_model_loss = 0.0

                    if self.cfg.distribute_modules:
                        outputs = [{} for _ in range(dist.get_world_size())]
                        dist.gather_object(info, outputs if rank_zero else None)
                        info.update({k: v for out in outputs for k, v in out.items()})

                    if rank_zero:
                        # Normalize step by batch size
                        normalized_step = step * self.cfg.batch_size
                        wandb.log(info, step=normalized_step)

                if (step + 1) % self.cfg.save_every == 0:
                    self.save()
                
            self.global_step += 1
            pbar.update()

        self.save()
        wandb.finish() if self.cfg.log_to_wandb else None
        pbar.close()

    def local_hookpoints(self) -> list[str]:
        return (
            self.module_plan[dist.get_rank()]
            if self.module_plan
            else self.cfg.hookpoints
        )

    def maybe_all_cat(self, x: Tensor) -> Tensor:
        """Concatenate a tensor across all processes."""
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        buffer = x.new_empty([dist.get_world_size() * x.shape[0], *x.shape[1:]])
        dist.all_gather_into_tensor(buffer, x)
        return buffer

    def maybe_all_reduce(self, x: Tensor, op: str = "mean") -> Tensor:
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        if op == "sum":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x /= dist.get_world_size()
        elif op == "max":
            dist.all_reduce(x, op=dist.ReduceOp.MAX)
        else:
            raise ValueError(f"Unknown reduction op '{op}'")

        return x

    def distribute_modules(self):
        """Prepare a plan for distributing modules across ranks."""
        if not self.cfg.distribute_modules:
            self.module_plan = []
            print(f"Training on modules: {self.cfg.hookpoints}")
            return

        layers_per_rank, rem = divmod(len(self.cfg.hookpoints), dist.get_world_size())
        assert rem == 0, "Number of modules must be divisible by world size"

        # Each rank gets a subset of the layers
        self.module_plan = [
            self.cfg.hookpoints[start : start + layers_per_rank]
            for start in range(0, len(self.cfg.hookpoints), layers_per_rank)
        ]
        for rank, modules in enumerate(self.module_plan):
            print(f"Rank {rank} modules: {modules}")

    def scatter_hiddens(self, hidden_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Scatter & gather the hidden states across ranks."""
        outputs = [
            # Add a new leading "layer" dimension to each tensor
            torch.stack([hidden_dict[hook] for hook in hookpoints], dim=1)
            for hookpoints in self.module_plan
        ]
        local_hooks = self.module_plan[dist.get_rank()]
        shape = next(iter(hidden_dict.values())).shape

        # Allocate one contiguous buffer to minimize memcpys
        buffer = outputs[0].new_empty(
            # The (micro)batch size times the world size
            shape[0] * dist.get_world_size(),
            # The number of layers we expect to receive
            len(local_hooks),
            # All other dimensions
            *shape[1:],
        )

        # Perform the all-to-all scatter
        inputs = buffer.split([len(output) for output in outputs])
        dist.all_to_all([x for x in inputs], outputs)

        # Return a list of results, one for each layer
        return {hook: buffer[:, i] for i, hook in enumerate(local_hooks)}

    def save(self, checkpoint_suffix: str | None = None):
        """Save the SAEs to disk with optional suffix for checkpoints."""
        path = f"saved_saes/{self.cfg.run_name}"
        if checkpoint_suffix:
            path = f"{path}-{checkpoint_suffix}"
            
        rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        if rank_zero or self.cfg.distribute_modules:
            print(f"Saving checkpoint to {path}")

            for hook, sae in self.saes.items():
                assert isinstance(sae, Sae)
                sae.save_to_disk(f"{path}/{hook}")
    
        if rank_zero:
            torch.save(self.lr_scheduler.state_dict(), f"{path}/lr_scheduler.pt")
            torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
            torch.save({
                "global_step": self.global_step,
                "num_tokens_since_fired": self.num_tokens_since_fired,
            }, f"{path}/state.pt")

            self.cfg.save_json(f"{path}/config.json")

        # Barrier to ensure all ranks have saved before continuing
        if dist.is_initialized():
            dist.barrier()

    def hook(self, module: nn.Module, _, outputs):
        # Maybe unpack tuple outputs
        if isinstance(outputs, tuple):
            output = outputs[0]

        name = self.module_to_name[module]
        self.hidden_dict[name] = output.flatten(0, 1)

        if name in self.saes:
            sae_out = self.saes[name](self.hidden_dict[name])
            sae_out = sae_out.sae_out.view_as(output)
            return (sae_out,) + outputs[1:]
        else:
            return outputs

    def register_hooks(self):
        # Clear any existing hooks
        self.remove_hooks()
        # Register new hooks
        self.handles = [
            mod.register_forward_hook(self.hook) 
            for mod in self.name_to_module.values()
        ]

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.hidden_dict.clear()

    def __del__(self):
        # Cleanup hooks when the trainer is destroyed
        self.remove_hooks()


