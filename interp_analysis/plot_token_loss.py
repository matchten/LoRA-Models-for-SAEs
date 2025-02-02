# %%
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
# %%

original_losses, sae_losses, peft_losses = torch.load(f"data/losses/all_original_losses.pt"), torch.load(f"data/losses/all_sae_losses.pt"), torch.load(f"data/losses/all_peft_losses.pt")
original_losses = original_losses.float().flatten().cpu()
sae_losses = sae_losses.float().flatten().cpu()
peft_losses = peft_losses.float().flatten().cpu()
stacked_tokens = torch.load(f"data/losses/stacked_tokens.pt")[:, :-1].flatten().cpu()
loss_diffs = sae_losses - peft_losses

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 5,
    'axes.labelsize': 6,
    'axes.titlesize': 6,
    'xtick.labelsize': 5.5,
    'ytick.labelsize': 5.5,
    'legend.fontsize': 5.5,
    'figure.dpi': 300
})

plt.figure(figsize=(3.25, 2))

# Split into positive and negative values
improvements = sae_losses - peft_losses
pos_improvements = improvements[improvements >= 0]
neg_improvements = -improvements[improvements < 0]  # Negate negative values

# Create log spaced bins from 1e-3 to 15
bins = np.logspace(-5, 1, 50)

# Plot positive values in blue
plt.hist(pos_improvements, bins=bins, alpha=0.7, color='#2196F3',
         edgecolor='black', linewidth=0.5, label='Loss Improvement')
# Plot negative values in red 
plt.hist(neg_improvements, bins=bins, alpha=0.7, color='#F44336',
         edgecolor='black', linewidth=0.5, label='Loss Degradation')

plt.xscale('log')
plt.xlabel('|Base Loss - LoRA Loss|')
plt.ylabel('# of Validation Tokens')
# plt.title('Distribution of Loss Improvements')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('plots/loss_improvement_distribution.pdf', bbox_inches='tight')

# %%