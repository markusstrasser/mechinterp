
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchinfo import summary
from collections import defaultdict
import io
import contextlib
from sklearn.decomposition import PCA

def _fingerprint_tensor(tensor: torch.Tensor) -> dict:
    """Computes a set of statistics for a single parameter tensor."""
    stats = {}
    t = tensor.detach().float().cpu()

    stats['shape'] = tuple(t.shape)
    stats['norm_l2'] = torch.linalg.norm(t).item()
    stats['mean'] = t.mean().item()
    stats['std'] = t.std().item()
    stats['min'] = t.min().item()
    stats['max'] = t.max().item()
    stats['sparsity_1e-5'] = (torch.sum(t.abs() < 1e-5) / t.numel()).item()

    if t.dim() == 2:
        svd_vals = torch.linalg.svdvals(t)
        stats['svd_max'] = svd_vals[0].item()
        stats['stable_rank'] = (svd_vals.sum()**2 / (svd_vals**2).sum()).item()
        stats['rank'] = torch.linalg.matrix_rank(t).item()

    return stats

def _create_ascii_pca_plot(data: torch.Tensor, width: int = 50, height: int = 20) -> str:
    """Creates a 2D ASCII scatter plot of the data using PCA."""
    if data.shape[0] < 2: return "Not enough data points for PCA plot."

    pca = PCA(n_components=2)
    try:
        proj = pca.fit_transform(data.cpu().numpy())  # simplified - no detach needed
    except Exception:
        return "PCA failed to converge."

    # Normalize projected data to fit within the grid dimensions
    min_x, max_x = proj[:, 0].min(), proj[:, 0].max()
    min_y, max_y = proj[:, 1].min(), proj[:, 1].max()

    # Avoid division by zero if all points are the same
    if max_x == min_x or max_y == min_y: return "Data has zero variance in projection."

    proj[:, 0] = (proj[:, 0] - min_x) / (max_x - min_x) * (width - 1)
    proj[:, 1] = (proj[:, 1] - min_y) / (max_y - min_y) * (height - 1)

    # Create the grid and plot points
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    for x, y in proj:
        # Flip y-axis for standard plot orientation
        grid[height - 1 - int(y)][int(x)] = '*'

    return "\n".join("".join(row) for row in grid)


def generate_model_cartography(model: nn.Module, example_input_shape: tuple) -> str:
    """
    Generates the definitive comprehensive text report for a PyTorch model.
    """
    with torch.no_grad():  # THE FIX - no more grad tracking nonsense
        report_parts = ["# 🗺️ Model Cartography Report\n"]

        # --- 1. Architectural Overview ---
        report_parts.append("## 🏛️ Architectural Overview")
        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            summary(model, input_size=example_input_shape, dtypes=[torch.long],
                    col_names=("input_size", "output_size", "num_params", "mult_adds"), verbose=0)
        report_parts.append(f"```\n{s.getvalue()}\n```")

        # --- 2. Statistical Fingerprints ---
        report_parts.append("## 🔬 Statistical Fingerprints")
        all_stats = {name: _fingerprint_tensor(param) for name, param in model.named_parameters()}
        stats_df = pd.DataFrame.from_dict(all_stats, orient='index').round(4)
        report_parts.append(stats_df.to_markdown())

        # --- 3. Attention Head Analysis ---
        report_parts.append("## 👁️ Attention Head Deep Dive & Pattern Analysis")
        attn_weights = defaultdict(dict)
        for name, param in model.named_parameters():
            if '.attn.W_' in name and param.dim() > 2:
                parts = name.split('.')
                weight_type = parts[3]
                block_idx = int(parts[1])
                for head_idx in range(param.shape[0]):
                    key = f"B{block_idx}-H{head_idx}"
                    attn_weights[key][weight_type] = param[head_idx]

        if attn_weights:
            # Part A: Statistical & Functional Properties
            head_analysis = []
            for key, weights in attn_weights.items():
                W_Q, W_K, W_V, W_O = (weights.get(t) for t in ['W_Q', 'W_K', 'W_V', 'W_O'])
                if all(t is not None for t in [W_Q, W_K, W_V, W_O]):
                    analysis = {'head': key}
                    # QK Orthogonality
                    q_norm, k_norm = W_Q.norm(), W_K.norm()
                    if q_norm > 0 and k_norm > 0:
                        analysis['QK_diff'] = torch.linalg.norm(W_Q / q_norm - W_K / k_norm).item()
                    # OV Composition (functional analysis)
                    composed_vo = W_V @ W_O
                    analysis['VO_stable_rank'] = _fingerprint_tensor(composed_vo).get('stable_rank')
                    head_analysis.append(analysis)

            if head_analysis:
                report_parts.append("### Functional & Statistical Properties")
                report_parts.append("`QK_diff`≈1.41 indicates orthogonal Q/K projections. `VO_stable_rank` shows the head's functional complexity (low=copying, high=complex).")
                head_df = pd.DataFrame(head_analysis).round(4)
                report_parts.append(head_df.to_markdown(index=False))

            # Part B: Positional Attention Patterns
            report_parts.append("\n### Positional Attention Patterns")
            W_pos = model.pos_embed.W_pos
            pattern_report = [ "Data-independent attention patterns. Rows are query_pos, columns are key_pos." ]
            for key, weights in attn_weights.items():
                W_Q, W_K = weights.get('W_Q'), weights.get('W_K')
                if W_Q is not None and W_K is not None:
                    QK_circuit = W_Q @ W_K.T
                    pos_attention_logits = W_pos @ QK_circuit @ W_pos.T
                    pos_attention_pattern = torch.softmax(pos_attention_logits, dim=-1)
                    header = f"**{key}**:"
                    pattern_str = "```\n" + np.array2string(pos_attention_pattern.cpu().numpy(), formatter={'float_kind': lambda x: "%.2f" % x}) + "\n```"
                    pattern_report.append(header + "\n" + pattern_str)
            if len(pattern_report) > 1: report_parts.append("\n".join(pattern_report))

        # --- 4. Embedding Geometry Analysis ---
        report_parts.append("## Embedding Geometry Analysis")
        report_parts.append("Visualizing the geometric structure of the learned token embeddings.")

        # Get d_vocab robustly from the embedding matrix shape.
        d_vocab = model.embed.W_E.shape[0]
        num_tokens = d_vocab - 1
        W_E = model.embed.W_E[:num_tokens, :]

        # Part A: ASCII PCA Plot
        report_parts.append("\n### PCA Projection of W_E")
        pca_plot = _create_ascii_pca_plot(W_E)
        report_parts.append(f"```\n{pca_plot}\n```")

        # Part B: Cosine Similarity Matrix
        report_parts.append("\n### W_E Cosine Similarity")
        W_E_normalized = W_E / W_E.norm(dim=-1, keepdim=True)
        cosine_sim_matrix = W_E_normalized @ W_E_normalized.T
        patch_size = 16
        patch = cosine_sim_matrix[:patch_size, :patch_size]
        header = f"Top-Left {patch_size}x{patch_size} Patch:"
        patch_str = "```\n" + np.array2string(patch.cpu().numpy(), formatter={'float_kind': lambda x: "%.2f" % x}) + "\n```"
        report_parts.append(header + "\n" + patch_str)

        # --- 5. Key MLP Neuron Analysis ---
        report_parts.append("## 💡 Key MLP Neuron Analysis")
        report_parts.append("Identifying the 'most important' neurons in the MLP layer by L2 norm of their connection weights.")
        W_in = model.blocks[0].mlp.W_in
        W_out = model.blocks[0].mlp.W_out
        top_k = 5
        in_norms, out_norms = W_in.norm(dim=0), W_out.norm(dim=1)
        top_in_indices = torch.topk(in_norms, top_k).indices
        top_out_indices = torch.topk(out_norms, top_k).indices
        report_parts.append(f"Top {top_k} neurons by **input** weight norm: {top_in_indices.tolist()}")
        report_parts.append(f"Top {top_k} neurons by **output** weight norm: {top_out_indices.tolist()}")

        return "\n\n---\n\n".join(report_parts)
