import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
from .types import TrainConfig


def create_model(config: TrainConfig) -> HookedTransformer:
    """
    Create and initialize a HookedTransformer model for modular arithmetic.
    
    Theory: Model Configuration
    - Small Model: The model is intentionally small (1 layer, small d_model). This makes it
      harder to simply memorize the entire p*p table via brute force, encouraging it to find
      the underlying algorithmic structure.
    - d_vocab = p + 1: We need tokens for numbers 0 to p-1, plus one special token for '='.
    - n_ctx = 3: The context length is fixed at 3 because our input is always `(a, b, =)`.
    """
    model_config = HookedTransformerConfig(
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_model=config.d_model,
        d_head=config.d_model // config.n_heads,
        d_mlp=config.d_ffn,
        d_vocab=config.p + 1,
        n_ctx=3,
        act_fn="relu",
        normalization_type=None,
        device=config.device,
    )
    
    model = HookedTransformer(model_config)
    
    # Theory: Zeroing out W_U
    # - This is a very specific and unusual choice. Typically W_U is tied to W_E or learned.
    # - By zeroing it, we force the model to learn the mapping from its internal representations
    #   to output logits from scratch. This may create a cleaner separation between the model's
    #   internal "computation" and its "output formatting," making analysis easier.
    with torch.no_grad():
        model.W_U.data.zero_()
    
    return model