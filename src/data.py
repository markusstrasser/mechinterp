import torch
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
        config.p,
        config.n_examples,
        config.frac_train,
        config.device,
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