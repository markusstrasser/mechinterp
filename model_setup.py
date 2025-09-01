import time
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer, HookedTransformerConfig


def generate_dataset(config: dict):
    """
    Generates the dataset for the modular arithmetic task (a + b) % p.

    This version is modified to be memory-efficient by sampling directly,
    rather than creating the full p*p table. The core theory remains.

    Theory:
    - Task Design: Modular arithmetic is a classic algorithmic task used to study "grokking."
      It's simple to specify but requires the model to learn a non-trivial, periodic structure.
      The entire dataset is finite and can be perfectly learned.
    - Tokenization: The inputs `a` and `b` and the special "equals" token are represented as integers.
      `p` is used as the "equals" token, signifying the position where the model should output the answer.
      This transforms a math problem into a standard seq-to-seq prediction task.
    - Data Split: A small training set (frac_train) and a large test set are used. This setup is
      CRUCIAL for observing grokking. The model first memorizes the small training set (achieving low
      train loss but high test loss) and only much later, after extensive training, does it "grok"
      the underlying pattern, causing test accuracy to suddenly spike.
    """
    p, n_examples, frac_train, device = (
        config["p"],
        config["n_examples"],
        config["frac_train"],
        config["device"],
    )

    # Sample 'n_examples' pairs of (a, b) directly to avoid memory issues with large p.
    a = torch.randint(0, p, (n_examples,), device=device)
    b = torch.randint(0, p, (n_examples,), device=device)

    # Create prompts and labels from the sampled pairs.
    prompts = torch.stack(
        [
            a,
            b,
            torch.full(
                (n_examples,), p, device=device
            ),  # The 'p' token acts as our '=' sign.
        ],
        dim=1,
    )
    labels = (a + b) % p

    # Randomly shuffle to ensure the train/test sets are not structured.
    indices = torch.randperm(n_examples, device=device)
    cutoff = int(n_examples * frac_train)
    train_indices, test_indices = indices[:cutoff], indices[cutoff:]

    train_data, test_data = prompts[train_indices], prompts[test_indices]
    train_labels, test_labels = labels[train_indices], labels[test_indices]

    return train_data, train_labels, test_data, test_labels



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
    return {**metrics, **analyze_grokking_mechanics(model, test_data, config)}


def train(config):
    train_data, train_labels, test_data, test_labels = generate_dataset(config)

    starttime = time.time()
    torch.manual_seed(config["seed"])
    # Theory: Model Configuration
    # - Small Model: The model is intentionally small (1 layer, small d_model). This makes it
    #   harder to simply memorize the entire p*p table via brute force, encouraging it to find
    #   the underlying algorithmic structure.
    # - d_vocab = p + 1: We need tokens for numbers 0 to p-1, plus one special token for '='.
    # - n_ctx = 3: The context length is fixed at 3 because our input is always `(a, b, =)`.
    model_config = HookedTransformerConfig(
        n_layers=config["n_layers"],
        n_heads=config["n_head"],
        d_model=config["d_model"],
        d_head=config["d_model"] // config["n_head"],
        d_mlp=config["d_ffn"],
        d_vocab=config["p"] + 1,
        n_ctx=3,
        act_fn="relu",
        normalization_type=None,
        device=config["device"],
    )
    model = HookedTransformer(model_config)
    # model = torch.compile(model)
    # Theory: Zeroing out W_U
    # - This is a very specific and unusual choice. Typically W_U is tied to W_E or learned.
    # - By zeroing it, we force the model to learn the mapping from its internal representations
    #   to output logits from scratch. This may create a cleaner separation between the model's
    #   internal "computation" and its "output formatting," making analysis easier. It ensures
    #   that any initial structure comes purely from the embedding and processing, not the unembedding.
    with torch.no_grad():
        model.W_U.data.zero_()

    # Theory: Optimizer
    # - AdamW: A standard, robust optimizer.
    # - Weight Decay: This is a critical hyperparameter. High weight decay penalizes large weights,
    #   acting as a form of regularization. In grokking, weight decay is thought to be a key driver,
    #   pushing the model away from complex, memorizing solutions towards simpler, generalizing ones.
    #   The value of `weight_decay` can dramatically affect whether and when grokking occurs.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # Initialize a history dictionary to store metrics over time
    history = {
        "steps": [],
        "train_loss": [],
        "test_loss": [],
        "test_acc": [],
        "l2_norm": [],
        "gini_embed": [],
        "gini_unembed": [],

        'fourier_sparsity': [],
        'attn_periodicity': [],
        'logit_attribution': [],
        'neuron_specialization': [],
    }

    # Pretty-print table header
    header = (
        f"{'Step':>6} | {'Time(s)':>8} | {'TrainLoss':>9} | {'TestLoss':>8} | "
        f"{'TestAcc':>7} | {'L2Norm':>10} | {'Gini(E)':>7} | {'Gini(U)':>7} | "
        f"{'FourierS':>8} | {'LogitDir':>8} | {'LogitMLP':>8} | {'LogitAttn':>9} | {'NeuronSp':>8}"
    )
    separator = "-" * len(header)
    print(header)
    print(separator)
    loss_fn = nn.CrossEntropyLoss()
        # Theory: Loss Calculation
        # - The choice `nn.CrossEntropyLoss()(logits, train_labels)` computes loss over all p+1
        #   possible output tokens. This means the model is penalized if it assigns high probability
        #   to the special '=' token (p) in the output position, which is desirable.
        # - The alternative, `logits[:, :config['p']]`, would ignore the logit for the '=' token.
        #   The comment in your original code suggests this has an effect on the Gini coefficient,
        #   likely because forcing the logit for token 'p' to be low might distribute probability
        #   mass differently across the other tokens, affecting sparsity.

    for step in range(config["steps"]):
        model.train()
        logits = model(train_data)[:, -1, :]
        loss = loss_fn(logits, train_labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % config['eval_interval'] == 0:
            metrics = evaluate(model, test_data, test_labels, config)  # <-- now gated
            metrics["train_loss"] = loss.item()

            history["steps"].append(step)
            history["train_loss"].append(metrics["train_loss"])
            history["test_loss"].append(metrics["test_loss"])
            history["test_acc"].append(metrics["test_acc"])
            history["l2_norm"].append(metrics["l2_norm"])
            history["gini_embed"].append(metrics["gini_embed"])
            history["gini_unembed"].append(metrics["gini_unembed"])

            history['fourier_sparsity'].append(metrics['fourier_sparsity'])
            # history['attn_periodicity'].append(metrics['attn_periodicity'])
            history['logit_attribution'].append(metrics['logit_attribution'])
            history['neuron_specialization'].append(metrics['neuron_specialization'])

            # Extract scalar components for printing
            la = metrics['logit_attribution']
            la_direct = la.get('direct', float('nan'))
            la_mlp = la.get('mlp', float('nan'))
            la_attn = la.get('attn', float('nan'))

            elapsed = time.time() - starttime
            print(
                f"{step:6d} | {elapsed:8.2f} | {metrics['train_loss']:9.4f} | "
                f"{metrics['test_loss']:8.4f} | {metrics['test_acc']:7.4f} | "
                f"{metrics['l2_norm']:10.4f} | {metrics['gini_embed']:7.4f} | {metrics['gini_unembed']:7.4f} | "
                f"{metrics['fourier_sparsity']:8.4f} | {la_direct:8.4f} | {la_mlp:8.4f} | {la_attn:9.4f} | "
                f"{metrics['neuron_specialization']:8.4f}"
            )

    # Return the trained model AND the full history of metrics
    return model, history