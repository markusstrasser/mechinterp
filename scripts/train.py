import time

import torch, os, json
from torch import nn as nn
from transformer_lens import HookedTransformerConfig, HookedTransformer

from model_setup import evaluate

DEFAULT_CONFIG={
    "p": 113,
    "n_examples": 113 * 113,
    "frac_train": 0.3,
    "n_layers": 1,
    "n_head": 3,
    "d_model": 32,
    "d_ffn": 128,
    "lr": 1e-3,
    "weight_decay": 1,
    "steps": 60001,
    "eval_interval": 1000,
    "seed": 43,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

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
        'fourier_structure': []
    }

    # Pretty-print table header
    header = (
        f"{'Step':>6} | {'Time(s)':>8} | {'TrainLoss':>9} | {'TestLoss':>8} | "
        f"{'TestAcc':>7} | {'L2Norm':>10} | {'Gini(E)':>7} | {'Gini(U)':>7} | "
        f"{'FourierS':>8} | {'LogitDir':>8} | {'LogitMLP':>8} | {'LogitAttn':>9} | {'NeuronSp':>8}"
        f"{'FourierC':>8} | {'FreqConc':>8} | {'Freqs':>10}"
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
            history['fourier_structure'].append(metrics['fourier_structure'])
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
f"{metrics['fourier_structure']['circulant_score']:8.4f} | {metrics['fourier_structure']['freq_concentration']:8.4f} | {str(metrics['fourier_structure']['dominant_freqs']):10}"
            )

    # Return the trained model AND the full history of metrics
    return model, history


if __name__ == "__main__":
    # Local run preserves old behavior for convenience.
    cfg={**DEFAULT_CONFIG, "steps": 10001}
    m, his = train(cfg)
    state = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    # History from model_setup.py is numeric lists already; safe to return.
    SAVE_ROOT = "runs"
    run_name = f"p{cfg['p']}_d{cfg['d_model']}_seed{cfg['seed']}"
    run_dir = os.path.join(SAVE_ROOT, run_name)
    os.makedirs(run_dir, exist_ok=True)
    base = os.path.join(run_dir, run_name)
    torch.save(state, base + ".pt")
    try:
        json.dump(his, open(base + ".history.json", "w"))
    except Exception as e:
        print("JSON save failed:", e)

