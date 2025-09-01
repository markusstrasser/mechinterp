import torch


PROBE_REGISTRY = {}

def probe(name):
    """A decorator to register a function as a probe."""
    def decorator(func):
        PROBE_REGISTRY[name] = func
        return func
    return decorator

# @probe("performance")
# @torch.no_grad()
# def performance_probe(model, data, labels, **kwargs):
#     # ... implementation
#     return {"test_loss": loss, "test_acc": acc}
#
# @probe("l2_norm")
# @torch.no_grad()
# def l2_norm_probe(model, **kwargs):
#     # ... implementation
#     return {"l2_norm": l2_norm}

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




