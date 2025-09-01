import torch
import torch.nn as nn
from transformer_lens import HookedTransformer


def gini(x):
    """
    Calculates the Gini coefficient, a measure of inequality or sparsity.

    Theory:
    - Sparsity: In neural networks, a sparse representation is one where most neurons are inactive (close to zero).
      A high Gini coefficient (close to 1) implies high sparsity.
    - Interpretability: Sparse representations are often considered more interpretable. If only a few
      neurons are active for a given input, it's easier to attribute meaning to those neurons.
      Tracking the Gini coefficient of embeddings and other weight matrices can indicate whether the
      model is learning a sparse, structured solution.
    """
    x = torch.abs(x.flatten())
    if torch.sum(x) == 0:
        return 0.0
    x = torch.sort(x)[0]  # Sort values in ascending order
    n = len(x)
    cumx = torch.cumsum(x, dim=0)
    # The Gini formula, normalized.
    return (n + 1 - 2 * torch.sum(cumx) / cumx[-1]) / n

@torch.no_grad()
def analyze_grokking_mechanics(model, test_data, config):
    """The actual mechanistic story of grokking"""
    # Cache all activations in one forward pass
    _, cache = model.run_with_cache(test_data)  # subsample for speed

    # 1. Fourier Analysis - THE key insight for modular arithmetic
    W_E = model.W_E[:config['p'], :]
    fft = torch.fft.fft(W_E, dim=0)
    fourier_sparsity = (torch.abs(fft) > 0.1).float().mean()  # tracks discrete Fourier basis emergence

    # 2. Attention patterns - look for modular structure
    # attn_patterns = cache['pattern', 0]  # layer 0 attention
    # # Check if attention learns modular periodicity
    # attn_periodicity = torch.std(attn_patterns.reshape(-1, config['p'], 3, 3).mean(0))

    # 3. Direct Logit Attribution - which components contribute to correct answers
    logits, cache = model.run_with_cache(test_data)
    logit_lens = model.unembed(cache['resid_post', 0])  # direct path contribution
    mlp_out = model.unembed(cache['mlp_out', 0])  # MLP contribution
    attn_out = model.unembed(cache['attn_out', 0])  # attention contribution

    # 4. Neuron specialization in MLP
    mlp_acts = cache['post', 0]  # post-activation in MLP
    neuron_specialization = (mlp_acts > 0).float().mean(0).std()  # how specialized are neurons

    return {
        'fourier_sparsity': fourier_sparsity.item(),
        # 'attn_periodicity': attn_periodicity.item(),
        'logit_attribution': {
            'direct': logit_lens.std().item(),
            'mlp': mlp_out.std().item(),
            'attn': attn_out.std().item()
        },
        'neuron_specialization': neuron_specialization.item()
    }

def analyze_fourier_structure(model, p):
    """Track emergence of specific modular arithmetic frequencies"""
    W_E = model.W_E[:p, :]
    W_U = model.W_U[:, :p]

    # The product W_U @ W_E.T should become circulant for modular arithmetic
    W_logit = W_U.T @ W_E.T

    # Check if it's learning the cyclic group structure
    # In frequency domain, circulant = diagonal
    fft_logit = torch.fft.fft2(W_logit)
    diagonality = (torch.abs(fft_logit.diag()).sum() /
                   torch.abs(fft_logit).sum())

    # Track specific frequency components that matter for mod p
    fft_embed = torch.fft.fft(W_E, dim=0)
    top_k_freqs = torch.topk(torch.abs(fft_embed).mean(1), k=5)
    freq_concentration = top_k_freqs.values.sum() / torch.abs(fft_embed).sum()

    return {
        'circulant_score': diagonality.item(),
        'freq_concentration': freq_concentration.item(),
        'dominant_freqs': top_k_freqs.indices.tolist()
    }



@torch.no_grad()
def evaluate(model: HookedTransformer, test_data, test_labels, config: dict):
    """
    Theory:
    - Test Accuracy vs. Loss: The key metric for grokking. We expect test accuracy to remain near random
      for a long time, even as training loss drops. The "grokking" moment is when test accuracy
      suddenly jumps to ~100%.
    - L2 Norm: This is the sum of the squares of all model parameters. It's a measure of the model's complexity.
      In grokking literature, it's observed that models often find a simpler (lower L2 norm) solution
      during the phase transition to generalization.
    - Gini Coefficients (Embed/Unembed): Tracking the sparsity of the embedding (W_E) and
      unembedding (W_U) matrices. An increase in Gini suggests the model is learning to represent
      numbers in a more structured, sparse way (e.g., via a Fourier basis) rather than as
      opaque, dense vectors.
    """
    model.eval()
    logits = model(test_data)[:, -1, :]
    test_loss = nn.CrossEntropyLoss()(logits, test_labels).item()
    test_acc = (
        (torch.argmax(logits, dim=-1) == test_labels).float().mean().item()
    )
    l2_norm = sum(p.pow(2).sum() for p in model.parameters()).item()

    W_E = model.W_E[: config["p"], :]
    W_U = model.W_U[:, : config["p"]]
    gini_embed = gini(W_E).item()
    gini_unembed = gini(W_U.T).item()

    metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "l2_norm": l2_norm,
        "gini_embed": gini_embed,
        "gini_unembed": gini_unembed,
    }
    return {**metrics, **analyze_grokking_mechanics(model, test_data, config), "fourier_structure": analyze_fourier_structure(model, config['p'])}




