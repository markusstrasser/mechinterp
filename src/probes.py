from torch import nn
import torch
import inspect
import functools
from collections import defaultdict
from typing import Callable, Any, TypedDict, Optional

# 1. The Context "Schema" using TypedDict
# This is just a type hint; no class is created at runtime.
class ProbeContext(TypedDict, total=False):
    model: Any
    config: Any
    step: int
    epoch: int
    test_data: Optional[Any]
    test_labels: Optional[Any]

_PROBES = defaultdict(dict)

def probe_on(event: str) -> Callable:
    def decorator(user_fn: Callable) -> Callable:
        sig_params = inspect.signature(user_fn).parameters

        @functools.wraps(user_fn)
        def wrapper(ctx: ProbeContext):
            call_args = {p: ctx.get(p) for p in sig_params if p in ctx}
            return user_fn(**call_args)

        _PROBES[event][user_fn.__name__] = wrapper
        return wrapper
    return decorator

def probe(user_fn: Callable) -> Callable:
    return probe_on("manual")(user_fn)


def trigger_probes(event: str, ctx: ProbeContext) -> dict:
    results = {}
    for func in _PROBES.get(event, {}).values():
        results.update(func(ctx))
    return results

@probe
def performance(model, test_data, test_labels, **kwargs):
    model.eval()
    logits = model(test_data)[:, -1, :]
    test_loss = nn.CrossEntropyLoss()(logits, test_labels).item()
    test_acc = (torch.argmax(logits, dim=-1) == test_labels).float().mean().item()
    return {"test_loss": test_loss, "test_acc": test_acc}

@probe
def l2_norm(model, **kwargs):
    return {"l2_norm": sum(p.pow(2).sum() for p in model.parameters()).item()}



def gini(x):
    x = torch.abs(x.flatten())
    if torch.sum(x) == 0: return 0.0
    x = torch.sort(x)[0]
    n = len(x)
    cumx = torch.cumsum(x, dim=0)
    return (n + 1 - 2 * torch.sum(cumx) / cumx[-1]) / n

@probe
def sparsity(model, config, **kwargs):
    # ... (implementation is the same)
    p = config.p
    W_E = model.W_E[:p, :]
    W_U = model.W_U[:, :p]
    gini_embed = gini(W_E).item()
    gini_unembed = gini(W_U.T).item()
    return {"gini_embed": gini_embed, "gini_unembed": gini_unembed}

@probe
def mechanics(model, test_data, config, **kwargs):
    _, cache = model.run_with_cache(test_data)
    W_E = model.W_E[:config.p, :]
    fft = torch.fft.fft(W_E, dim=0)
    fourier_sparsity = (torch.abs(fft) > 0.1).float().mean()

    logits, cache = model.run_with_cache(test_data)
    logit_lens = model.unembed(cache['resid_post', 0])
    mlp_out = model.unembed(cache['mlp_out', 0])
    attn_out = model.unembed(cache['attn_out', 0])

    mlp_acts = cache['post', 0]
    neuron_specialization = (mlp_acts > 0).float().mean(0).std()

    # --- MODIFIED: Return a flat dictionary ---
    return {
        'fourier_sparsity': fourier_sparsity.item(),
        'logit_attribution.direct': logit_lens.std().item(),
        'logit_attribution.mlp': mlp_out.std().item(),
        'logit_attribution.attn': attn_out.std().item(),
        'neuron_specialization': neuron_specialization.item()
    }

@probe
def fourier_structure(model, config, **kwargs):
    # ... (implementation is the same)
    p = config.p
    W_E = model.W_E[:p, :]
    W_U = model.W_U[:, :p]
    W_logit = W_U.T @ W_E.T
    fft_logit = torch.fft.fft2(W_logit)
    diagonality = (torch.abs(fft_logit.diag()).sum() / torch.abs(fft_logit).sum())
    fft_embed = torch.fft.fft(W_E, dim=0)
    top_k_freqs = torch.topk(torch.abs(fft_embed).mean(1), k=5)
    freq_concentration = top_k_freqs.values.sum() / torch.abs(fft_embed).sum()
    return {
        'circulant_score': diagonality.item(),
        'freq_concentration': freq_concentration.item(),
    }

