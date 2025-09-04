import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import matplotlib.cm as cm

# Assume the following are already in your environment:
# model: the pre-trained HookedTransformer model
# config: the dictionary with model and training parameters (p, d_model, etc.)
# test_data: your data tensors


def _rectangle(x):
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)

class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold); ctx.bandwidth = bandwidth
        return (x > threshold).to(x.dtype)
    @staticmethod
    def backward(ctx, output_grad):
        x, threshold = ctx.saved_tensors; bandwidth = ctx.bandwidth
        grad_threshold = -(1.0 / bandwidth) * _rectangle((x - threshold) / bandwidth) * output_grad
        return torch.zeros_like(x), grad_threshold, None

class JumpReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold); ctx.bandwidth = bandwidth
        return x * (x > threshold)
    @staticmethod
    def backward(ctx, output_grad):
        x, threshold = ctx.saved_tensors; bandwidth = ctx.bandwidth
        grad_x = (x > threshold).to(x.dtype) * output_grad
        grad_threshold = -(threshold / bandwidth) * _rectangle((x - threshold) / bandwidth) * output_grad
        return grad_x, grad_threshold, None

# --- 2. JumpReLU_SAE Class Definition ---
class JumpReLU_SAE(nn.Module):
    def __init__(self, d_model, d_sae, initial_log_threshold=0.001, bandwidth=0.001):
        super().__init__()
        self.d_model, self.d_sae, self.bandwidth = d_model, d_sae, bandwidth
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_model, d_sae)))
        self.b_enc = nn.Parameter(torch.zeros(d_sae)); self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.log_threshold = nn.Parameter(torch.log(torch.full((d_sae,), initial_log_threshold)))

    @property
    def W_dec_normed(self):
        return self.W_enc.t() / (torch.norm(self.W_enc.t(), p=2, dim=1, keepdim=True) + 1e-8)

    def forward(self, x):
        pre_activations = F.relu(x @ self.W_enc + self.b_enc)
        threshold = torch.exp(self.log_threshold)
        feature_magnitudes = JumpReLUFunction.apply(pre_activations, threshold, self.bandwidth)
        x_reconstructed = feature_magnitudes @ self.W_dec_normed + self.b_dec
        return x_reconstructed, feature_magnitudes, pre_activations

# --- 3. Activation Harvesting and Analysis Functions ---

@torch.no_grad()
def get_activations(model, data, layer, device, batch_size):
    model.eval(); activations = []
    print("Harvesting activations...")
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size].to(device)
        if batch.shape[0] == 0: continue
        _, cache = model.run_with_cache(batch)
        activations.append(cache[f"blocks.{layer}.hook_resid_post"][:, -1, :].cpu())
    return torch.cat(activations, dim=0)

@torch.no_grad()
def run_full_analysis(sae, model, p, layer, device, num_features_to_plot=5):
    print("\n--- Running Full Feature Analysis ---"); sae.eval().to(device); model.eval().to(device)
    analytical_inputs = torch.tensor([[i, 0, p] for i in range(p)], device=device)
    _, cache = model.run_with_cache(analytical_inputs)
    activations = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :]
    _, feature_activations, _ = sae(activations)
    feature_variances = torch.var(feature_activations, dim=0)
    top_feature_indices = torch.topk(feature_variances, k=num_features_to_plot).indices

    for feature_idx in top_feature_indices:
        fig, axes = plt.subplots(1, 3, figsize=(21, 5)); fig.suptitle(f"Comprehensive Analysis of Feature {feature_idx.item()}", fontsize=16)
        feature_data = feature_activations[:, feature_idx].cpu()
        axes[0].plot(range(p), feature_data); axes[0].set_title("Feature Activation vs. Input 'a'"); axes[0].grid(True)
        fft_result = torch.fft.fft(feature_data); fft_magnitude = torch.abs(fft_result); freqs = torch.fft.fftfreq(len(feature_data))
        axes[1].bar(freqs[:p//2] * p, fft_magnitude[:p//2]); axes[1].set_title("FFT of Activation Pattern"); axes[1].grid(True)
        feature_vector = sae.W_dec_normed[feature_idx]; logit_influence = feature_vector @ model.W_U
        axes[2].plot(range(p), logit_influence[:p].cpu()); axes[2].set_title("Influence on Output Logits"); axes[2].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

    feature_vectors = sae.W_dec_normed[top_feature_indices]; cosine_sim_matrix = feature_vectors @ feature_vectors.T
    fig, ax = plt.subplots(figsize=(8, 6)); im = ax.imshow(cosine_sim_matrix.cpu().numpy(), cmap=cm.viridis); plt.colorbar(im)
    ax.set_xticks(np.arange(len(top_feature_indices))); ax.set_yticks(np.arange(len(top_feature_indices)))
    ax.set_xticklabels(top_feature_indices.cpu().numpy()); ax.set_yticklabels(top_feature_indices.cpu().numpy())
    ax.set_title("Cosine Similarity Between Top Feature Vectors"); plt.show()
    return top_feature_indices

@torch.no_grad()
def analyze_feature_ablation(sae, model, layer, device, p, feature_to_ablate, a, b):
    """(Corrected) Analyzes the causal effect of ablating a single feature."""
    print(f"\n--- Causal Ablation Analysis for Feature {feature_to_ablate} ---")
    sae.eval().to(device); model.eval().to(device)
    test_input = torch.tensor([[a, b, p]], device=device)

    # --- THIS IS THE CORRECTED PART ---
    # We select the logits for the final token position BEFORE taking the argmax
    baseline_logits = model(test_input)[:, -1, :]
    baseline_pred = baseline_logits.argmax(-1).item()
    # ------------------------------------

    print(f"Input: ({a} + {b}) % {p} = {(a+b)%p} | Baseline model prediction: {baseline_pred}")

    _, cache = model.run_with_cache(test_input); activation_at_pos = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :]
    _, c, _ = sae(activation_at_pos)
    feature_contribution = c[:, feature_to_ablate].unsqueeze(1) * sae.W_dec_normed[feature_to_ablate]

    def ablation_hook(resid_post, hook):
        resid_post[:, -1, :] -= feature_contribution
        return resid_post
    ablated_logits = model.run_with_hooks(test_input, fwd_hooks=[(f"blocks.{layer}.hook_resid_post", ablation_hook)])

    # Also select the final token's logits here
    final_ablated_logits = ablated_logits[:, -1, :]
    ablated_pred = final_ablated_logits.argmax(-1).item()
    print(f"Prediction after ablating feature {feature_to_ablate}: {ablated_pred} | Effect: {'SUCCESS' if ablated_pred != baseline_pred else 'NO CHANGE'}")

# --- 4. Main Execution Block ---

def run_sae(model, test_data, config):
    jumprelu_sae_config = {
        'd_sae': config['d_model'] * 4, 'l0_lambda': 0.004, 'lr': 7e-5,
        'epochs': 25, 'batch_size': 4096, 'initial_threshold': 0.001,
        'bandwidth': 0.001, 'target_layer': 0
    }
    device = config['device']

    activations_data = get_activations(model, test_data, jumprelu_sae_config['target_layer'], device, jumprelu_sae_config['batch_size'])
    print(f"\n--- Training the JumpReLU Sparse Autoencoder ---")
    sae = JumpReLU_SAE(config['d_model'], jumprelu_sae_config['d_sae'], initial_log_threshold=jumprelu_sae_config['initial_threshold'], bandwidth=jumprelu_sae_config['bandwidth']).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=jumprelu_sae_config['lr'])
    num_batches = len(activations_data) // jumprelu_sae_config['batch_size']

    for epoch in range(jumprelu_sae_config['epochs']):
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{jumprelu_sae_config['epochs']}")
        for i in pbar:
            batch = activations_data[i*jumprelu_sae_config['batch_size']:(i+1)*jumprelu_sae_config['batch_size']].to(device)
            optimizer.zero_grad(); x_hat, c, pre_act = sae(batch)
            recon_loss = F.mse_loss(x_hat, batch, reduction='none').sum(dim=-1)
            threshold = torch.exp(sae.log_threshold)
            l0_norm = StepFunction.apply(pre_act, threshold, sae.bandwidth).sum(dim=-1)
            sparsity_loss = jumprelu_sae_config['l0_lambda'] * l0_norm
            loss = (recon_loss + sparsity_loss).mean(); loss.backward(); optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.2f}", "L0": f"{l0_norm.mean().item():.2f}"})

    top_indices = run_full_analysis(sae, model, config['p'], jumprelu_sae_config['target_layer'], device)

    if top_indices is not None and len(top_indices) > 0:
        analyze_feature_ablation(sae, model, jumprelu_sae_config['target_layer'], device, config['p'], top_indices[0].item(), a=10, b=20)
        analyze_feature_ablation(sae, model, jumprelu_sae_config['target_layer'], device, config['p'], top_indices[1].item(), a=10, b=20)
    else:
        print("Please ensure 'model' and 'test_data' are loaded in the environment before running.")