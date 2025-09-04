# TODO

* DO DYNAMIC CHECKPOINTING?
* What should be filmed? At what point? 
* 3 Models: "no", "almost" "yes" , ... maybe "yes, easily"

## Design

The experimental setup is explicitly designed to induce and study the grokking phenomenon, where a model first memorizes its training data before suddenly learning to generalize.

* **Modular Arithmetic Task**: We use the task $(a + b) \pmod p$. This is a classic choice in grokking literature because it's a simple-to-define algorithmic problem with a non-trivial, periodic structure. The entire dataset is finite, allowing for perfect generalization to be unambiguously achieved.

* **Tokenization Strategy**: The problem is framed as a sequence-to-sequence task. The input `(a, b)` is represented by two integer tokens. A special token, `p`, is used as an "equals sign" or prompt for the answer. The model thus receives the sequence `[a, b, p]` and is trained to predict the token for `(a + b) \pmod p` at the final position.

* **Grokking-Inducing Data Split**: A small training set (e.g., 30% of all possible pairs) and a large test set are used. This is **the crucial component** for observing grokking. The model has enough capacity to completely memorize the small training set, leading to low training loss but high test loss. Only much later in training does the model undergo a phase transition, "grokking" the underlying algorithm and causing test accuracy to suddenly spike to ~100%.

The model's architecture and initialization are deliberately constrained to encourage the discovery of algorithmic solutions over brute-force memorization.

* **Minimalist Model**: The transformer is intentionally small (e.g., 1 layer, small `d_model`). This makes it harder for the model to simply memorize the entire multiplication table via its parameters, creating implicit pressure to find a more compressed, algorithmic solution.

* **Unembedding Initialization (`W_U`)**: The unembedding matrix `W_U` is initialized to all zeros. This is a highly unusual but deliberate choice for interpretability. Typically, `W_U` is tied to the embedding matrix `W_E`. By zeroing it, we force the model to learn the mapping from its internal representations to output logits from scratch. This creates a cleaner separation between the model's internal "computation" and its "output formatting," ensuring that any initial structure comes purely from the embedding and processing paths.
