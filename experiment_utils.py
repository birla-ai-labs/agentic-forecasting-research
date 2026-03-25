"""
Shared utilities for Migas 1.5 reverse-engineering experiments.

Provides:
- extract_internals(): Hook-based extraction of all intermediate representations
- Helper functions for plotting, metrics, and synthetic data generation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Internal extraction via PyTorch forward hooks
# ---------------------------------------------------------------------------

class MigasInternalsExtractor:
    """Extracts intermediate representations from a Migas 1.5 forward pass.

    Uses PyTorch forward hooks to non-invasively capture:
    - Convex combination weights (w)
    - Fusion gate values
    - Cross-attention weights (factual vs predictive)
    - All intermediate embeddings
    - Component forecasts (Chronos-only, fusion-head-only, final blended)

    Usage:
        extractor = MigasInternalsExtractor(pipeline)
        results = extractor.run(context_values, summary_text, pred_len=8)
    """

    def __init__(self, pipeline):
        """
        Args:
            pipeline: A loaded MigasPipeline instance.
        """
        self.pipeline = pipeline
        self.model = pipeline.model
        self.device = pipeline.device
        self._hooks = []
        self._captured = {}

    def _register_hooks(self):
        """Register forward hooks on key modules to capture intermediates."""
        self._hooks = []
        self._captured = {}

        # 1. Cross-attention inside GatedAttentionFusion
        #    nn.MultiheadAttention returns (attn_output, attn_weights) by default
        def hook_cross_attn(module, input, output):
            attn_output, attn_weights = output
            self._captured["cross_attn_output"] = attn_output.detach().cpu()
            self._captured["cross_attn_weights"] = attn_weights.detach().cpu() if attn_weights is not None else None

        self._hooks.append(
            self.model.fusion.cross_attn.register_forward_hook(hook_cross_attn)
        )

        # 2. Gate network inside GatedAttentionFusion
        def hook_gate_net(module, input, output):
            self._captured["gate_values"] = output.detach().cpu()
            self._captured["gate_input"] = input[0].detach().cpu()

        self._hooks.append(
            self.model.fusion.gate_net.register_forward_hook(hook_gate_net)
        )

        # 3. Convex weight network
        if hasattr(self.model, "convex_weight_net"):
            def hook_convex(module, input, output):
                self._captured["convex_weights"] = output.detach().cpu()
                self._captured["convex_input"] = input[0].detach().cpu()

            self._hooks.append(
                self.model.convex_weight_net.register_forward_hook(hook_convex)
            )

        # 4. Forecast head (fusion-only output before convex blending)
        def hook_forecast_head(module, input, output):
            self._captured["forecast_head_output"] = output.detach().cpu()

        self._hooks.append(
            self.model.forecast_head.register_forward_hook(hook_forecast_head)
        )

        # 5. Time series embedder (Chronos forecast → embedding)
        def hook_ts_embedder(module, input, output):
            self._captured["ts_embedding_raw"] = output.detach().cpu()
            self._captured["ts_embedder_input"] = input[0].detach().cpu()

        self._hooks.append(
            self.model.timeseries_embedder.register_forward_hook(hook_ts_embedder)
        )

        # 6. Factual text embedder (FinBERT embedding → projected)
        def hook_fact_embedder(module, input, output):
            self._captured["fact_embedding_projected"] = output.detach().cpu()
            self._captured["fact_embedding_raw"] = input[0].detach().cpu()

        self._hooks.append(
            self.model.fact_embedder.register_forward_hook(hook_fact_embedder)
        )

        # 7. Predictive text embedder (FinBERT embedding → projected)
        def hook_pred_embedder(module, input, output):
            self._captured["pred_embedding_projected"] = output.detach().cpu()
            self._captured["pred_embedding_raw"] = input[0].detach().cpu()

        self._hooks.append(
            self.model.prediction_embedder.register_forward_hook(hook_pred_embedder)
        )

        # 8. LayerNorms (to get the normalized versions)
        def hook_ts_norm(module, input, output):
            self._captured["ts_embedding_normed"] = output.detach().cpu()

        def hook_fact_norm(module, input, output):
            self._captured["fact_embedding_normed"] = output.detach().cpu()

        def hook_pred_norm(module, input, output):
            self._captured["pred_embedding_normed"] = output.detach().cpu()

        self._hooks.append(self.model.ts_norm.register_forward_hook(hook_ts_norm))
        self._hooks.append(self.model.fact_norm.register_forward_hook(hook_fact_norm))
        self._hooks.append(self.model.pred_norm.register_forward_hook(hook_pred_norm))

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def run(
        self,
        context_values: np.ndarray,
        summary: str,
        pred_len: int = 8,
    ) -> Dict[str, Any]:
        """Run a single forward pass and return all captured internals.

        Args:
            context_values: 1-D numpy array of historical values.
            summary: Pre-computed summary string (FACTUAL SUMMARY + PREDICTIVE SIGNALS).
            pred_len: Forecast horizon.

        Returns:
            Dict with keys:
                - 'forecast': final Migas forecast, shape (pred_len,), denormalized
                - 'chronos_forecast': Chronos-only forecast, shape (pred_len,), denormalized
                - 'forecast_head_output': fusion head output before convex blend, normalized space, shape (pred_len,)
                - 'convex_weights': per-step blend weights w, shape (pred_len,); w=1 means pure Chronos
                - 'gate_values': fusion gate (512-d sigmoid), shape (512,)
                - 'cross_attn_weights': attention over [factual, predictive] tokens, shape (1, 2)
                - 'ts_embedding_normed': normalized TS embedding, shape (512,)
                - 'fact_embedding_normed': normalized factual embedding, shape (512,)
                - 'pred_embedding_normed': normalized predictive embedding, shape (512,)
                - 'fact_embedding_raw': raw FinBERT embedding, shape (768,)
                - 'pred_embedding_raw': raw FinBERT embedding, shape (768,)
                - 'norm_params': dict with 'mu' and 'sigma' used for normalization
        """
        self._register_hooks()

        try:
            context = torch.tensor(context_values, dtype=torch.float32).reshape(1, -1)

            # Run the pipeline's predict method (handles normalization)
            fc, ts_fc = self.pipeline.predict(
                context,
                text=None,
                pred_len=pred_len,
                summaries=[summary],
                return_univariate=True,
            )

            # Compute normalization params for reference
            mu = context_values.mean()
            sigma = context_values.std(ddof=0)

            results = {
                "forecast": fc[0, :pred_len, 0].detach().cpu().numpy(),
                "chronos_forecast": ts_fc[0, :pred_len, 0].detach().cpu().numpy(),
                "norm_params": {"mu": float(mu), "sigma": float(sigma)},
            }

            # Unpack captured internals (squeeze batch dim)
            if "convex_weights" in self._captured:
                results["convex_weights"] = self._captured["convex_weights"][0, :pred_len].numpy()
            if "forecast_head_output" in self._captured:
                results["forecast_head_output"] = self._captured["forecast_head_output"][0, :pred_len].numpy()
            if "gate_values" in self._captured:
                results["gate_values"] = self._captured["gate_values"][0].numpy()
            if "cross_attn_weights" in self._captured and self._captured["cross_attn_weights"] is not None:
                results["cross_attn_weights"] = self._captured["cross_attn_weights"][0].numpy()
            if "ts_embedding_normed" in self._captured:
                results["ts_embedding_normed"] = self._captured["ts_embedding_normed"][0].numpy()
            if "fact_embedding_normed" in self._captured:
                results["fact_embedding_normed"] = self._captured["fact_embedding_normed"][0].numpy()
            if "pred_embedding_normed" in self._captured:
                results["pred_embedding_normed"] = self._captured["pred_embedding_normed"][0].numpy()
            if "fact_embedding_raw" in self._captured:
                results["fact_embedding_raw"] = self._captured["fact_embedding_raw"][0].numpy()
            if "pred_embedding_raw" in self._captured:
                results["pred_embedding_raw"] = self._captured["pred_embedding_raw"][0].numpy()
            if "ts_embedder_input" in self._captured:
                results["chronos_forecast_normalized"] = self._captured["ts_embedder_input"][0].numpy()
            if "cross_attn_output" in self._captured:
                results["cross_attn_output"] = self._captured["cross_attn_output"][0].numpy()

            return results

        finally:
            self._remove_hooks()

    def run_batch(
        self,
        context_values: np.ndarray,
        summaries: Dict[str, str],
        pred_len: int = 8,
    ) -> Dict[str, Dict[str, Any]]:
        """Run extraction for multiple summaries on the same context.

        Args:
            context_values: 1-D numpy array of historical values.
            summaries: Dict mapping scenario names to summary strings.
            pred_len: Forecast horizon.

        Returns:
            Dict mapping scenario names to their extraction results.
        """
        results = {}
        for name, summary in summaries.items():
            results[name] = self.run(context_values, summary, pred_len)
        return results


# ---------------------------------------------------------------------------
# Embedding-only extraction (no forward pass needed)
# ---------------------------------------------------------------------------

def extract_text_embeddings(pipeline, texts: List[str]) -> np.ndarray:
    """Get raw FinBERT/text embeddings for a list of text strings.

    Does NOT run the full model forward pass. Just calls the text encoder.

    Args:
        pipeline: Loaded MigasPipeline.
        texts: List of text strings to embed.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim).
    """
    from migaseval.model.util import encode_texts
    with torch.no_grad():
        embeddings = encode_texts(texts, batch_size=len(texts))
    return np.array(embeddings)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def text_shift(migas_fc: np.ndarray, chronos_fc: np.ndarray) -> float:
    """Mean absolute difference between Migas and Chronos forecasts.

    Measures how much the text conditioning shifted the forecast away from
    the pure univariate baseline.

    Args:
        migas_fc: Migas forecast array.
        chronos_fc: Chronos-only forecast array.

    Returns:
        Scalar MAE between the two forecasts.
    """
    return float(np.mean(np.abs(migas_fc - chronos_fc)))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.

    Args:
        a, b: 1-D numpy arrays of the same length.

    Returns:
        Cosine similarity in [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_sim_matrix(embeddings: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise cosine similarity matrix for named embeddings.

    Args:
        embeddings: Dict mapping names to 1-D numpy arrays.

    Returns:
        Square DataFrame of cosine similarities.
    """
    names = list(embeddings.keys())
    n = len(names)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = cosine_sim(embeddings[names[i]], embeddings[names[j]])
    return pd.DataFrame(mat, index=names, columns=names)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_convex_weights(
    weights_dict: Dict[str, np.ndarray],
    title: str = "Convex Combination Weights (w)",
    figsize: tuple = (12, 5),
):
    """Bar chart of convex weights across scenarios.

    w=1.0 means pure Chronos (text ignored), w=0.0 means pure fusion head.

    Args:
        weights_dict: Dict mapping scenario names to weight arrays of shape (pred_len,).
        title: Plot title.
        figsize: Figure size.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=figsize)
    n_scenarios = len(weights_dict)
    n_steps = len(next(iter(weights_dict.values())))
    x = np.arange(n_steps)
    width = 0.8 / n_scenarios

    for i, (name, w) in enumerate(weights_dict.items()):
        offset = (i - n_scenarios / 2 + 0.5) * width
        ax.bar(x + offset, w, width, label=name, alpha=0.8)

    ax.set_xlabel("Forecast Step")
    ax.set_ylabel("w (1=Chronos, 0=Fusion Head)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"t+{i+1}" for i in range(n_steps)])
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.4)
    plt.tight_layout()
    return fig, ax


def plot_forecast_comparison(
    context: np.ndarray,
    forecasts: Dict[str, np.ndarray],
    ground_truth: Optional[np.ndarray] = None,
    title: str = "Forecast Comparison",
    figsize: tuple = (13, 5),
):
    """Plot context + multiple forecast scenarios.

    Args:
        context: 1-D array of historical values.
        forecasts: Dict mapping names to forecast arrays.
        ground_truth: Optional ground truth array.
        title: Plot title.
        figsize: Figure size.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=figsize)
    t_ctx = range(len(context))
    pred_len = len(next(iter(forecasts.values())))
    t_fc = range(len(context), len(context) + pred_len)

    ax.plot(t_ctx[-12:], context[-12:], "o-", color="steelblue", ms=3, label="Context (last 12)")

    if ground_truth is not None:
        ax.plot(t_fc, ground_truth, "o-", color="black", ms=5, lw=2, label="Ground Truth")

    colors = plt.cm.Set2(np.linspace(0, 1, len(forecasts)))
    for (name, fc), color in zip(forecasts.items(), colors):
        ax.plot(t_fc, fc, "s--", color=color, ms=4, label=name)

    ax.axvline(len(context) - 0.5, color="gray", ls=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.set_xlabel("Step")
    plt.tight_layout()
    return fig, ax


def plot_similarity_matrix(
    sim_df: pd.DataFrame,
    title: str = "Cosine Similarity Matrix",
    figsize: tuple = (8, 6),
):
    """Heatmap of pairwise cosine similarities.

    Args:
        sim_df: Square DataFrame from cosine_sim_matrix().
        title: Plot title.
        figsize: Figure size.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sim_df.values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(sim_df.columns)))
    ax.set_yticks(range(len(sim_df.index)))
    ax.set_xticklabels(sim_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(sim_df.index, fontsize=8)
    for i in range(len(sim_df)):
        for j in range(len(sim_df)):
            ax.text(j, i, f"{sim_df.values[i, j]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def plot_gate_values(
    gate_dict: Dict[str, np.ndarray],
    title: str = "Fusion Gate Values (512-d)",
    figsize: tuple = (14, 4),
):
    """Histogram overlay of gate values across scenarios.

    The gate is a 512-dimensional sigmoid vector. Values near 0 mean the
    text attention output is suppressed; near 1 means it's fully added.

    Args:
        gate_dict: Dict mapping scenario names to gate arrays of shape (512,).
        title: Plot title.
        figsize: Figure size.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, gate in gate_dict.items():
        ax.hist(gate, bins=50, alpha=0.5, label=f"{name} (mean={gate.mean():.3f})", density=True)
    ax.set_xlabel("Gate Value (0=suppress text, 1=pass text)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    return fig, ax


def plot_attention_weights(
    attn_dict: Dict[str, np.ndarray],
    title: str = "Cross-Attention Weights: Factual vs Predictive",
    figsize: tuple = (10, 5),
):
    """Bar chart showing how much attention the TS query pays to factual vs predictive tokens.

    Args:
        attn_dict: Dict mapping scenario names to attention weight arrays of shape (1, 2).
                   Column 0 = factual, Column 1 = predictive.
        title: Plot title.
        figsize: Figure size.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=figsize)
    names = list(attn_dict.keys())
    n = len(names)
    x = np.arange(n)
    width = 0.35

    fact_weights = [attn_dict[name][0, 0] if attn_dict[name] is not None else 0 for name in names]
    pred_weights = [attn_dict[name][0, 1] if attn_dict[name] is not None else 0 for name in names]

    ax.bar(x - width / 2, fact_weights, width, label="Factual", color="steelblue")
    ax.bar(x + width / 2, pred_weights, width, label="Predictive", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Attention Weight")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Decomposition: reconstruct forecast from components
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Synthetic series generation
# ---------------------------------------------------------------------------

SYNTHETIC_PATTERNS = [
    "constant", "trend_up", "trend_down", "sine",
    "random_walk", "volatile", "intermittent", "regime_change",
]


def generate_synthetic_series(
    pattern: str,
    length: int = 50,
    scale: float = 1000.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic time series with a canonical pattern.

    All patterns produce non-negative values in approximately [0, scale].

    Args:
        pattern: One of SYNTHETIC_PATTERNS.
        length: Number of time steps.
        scale: Value range scaling factor.
        seed: Random seed for reproducible stochastic patterns.

    Returns:
        1-D float32 numpy array of shape (length,).
    """
    rng = np.random.RandomState(seed)
    t = np.arange(length, dtype=np.float64)

    if pattern == "constant":
        values = np.full(length, 0.5 * scale)
    elif pattern == "trend_up":
        values = np.linspace(0.2, 0.8, length) * scale
    elif pattern == "trend_down":
        values = np.linspace(0.8, 0.2, length) * scale
    elif pattern == "sine":
        values = (0.5 + 0.3 * np.sin(2 * np.pi * t / 12)) * scale
    elif pattern == "random_walk":
        steps = rng.randn(length) * 0.02 * scale
        values = 0.5 * scale + np.cumsum(steps)
        values = np.clip(values, 0.01 * scale, 0.99 * scale)
    elif pattern == "volatile":
        values = 0.5 * scale + rng.randn(length) * 0.25 * scale
        values = np.clip(values, 0.01 * scale, 0.99 * scale)
    elif pattern == "intermittent":
        values = np.full(length, 0.01 * scale)
        n_spikes = max(1, length // 5)
        spikes = rng.choice(length, size=n_spikes, replace=False)
        values[spikes] = rng.uniform(0.5, 1.0, n_spikes) * scale
    elif pattern == "regime_change":
        split = length * 3 // 5
        values = np.concatenate([
            np.full(split, 0.3 * scale),
            np.full(length - split, 0.7 * scale),
        ])
    else:
        raise ValueError(
            f"Unknown pattern '{pattern}'. Choose from {SYNTHETIC_PATTERNS}"
        )

    return values.astype(np.float32)


def describe_series(values: np.ndarray) -> str:
    """Generate an accurate factual description of a time series.

    Suitable for the FACTUAL SUMMARY section of a Migas prompt.

    Args:
        values: 1-D array of time series values.

    Returns:
        Plain-text factual description.
    """
    n = len(values)
    mean_v = values.mean()
    std_v = values.std()
    min_v, max_v = values.min(), values.max()
    recent = values[-4:].mean() if n >= 4 else mean_v

    if n >= 2:
        slope = (values[-1] - values[0]) / (n - 1)
        if abs(slope) < 0.005 * mean_v:
            trend = "flat"
        elif slope > 0:
            trend = "upward"
        else:
            trend = "downward"
    else:
        trend = "flat"

    return (
        f"The series spans {n} observations ranging from {min_v:.1f} to "
        f"{max_v:.1f}. Mean value is {mean_v:.1f} with standard deviation "
        f"{std_v:.1f}. Recent 4-period average: {recent:.1f}. "
        f"Overall {trend} trend."
    )


def build_summary(factual: str, predictive: str) -> str:
    """Build a properly formatted Migas summary string.

    Args:
        factual: Content for the FACTUAL SUMMARY section.
        predictive: Content for the PREDICTIVE SIGNALS section.

    Returns:
        Formatted string with both sections.
    """
    return f"FACTUAL SUMMARY:\n{factual}\n\nPREDICTIVE SIGNALS:\n{predictive}"


# ---------------------------------------------------------------------------
# Heatmap visualization for experiment matrices
# ---------------------------------------------------------------------------

def plot_heatmap(
    data: pd.DataFrame,
    title: str = "Heatmap",
    cmap: str = "YlOrRd",
    fmt: str = ".0f",
    figsize: tuple = (12, 6),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Annotated heatmap for a DataFrame (e.g. text_shift matrix).

    Args:
        data: DataFrame with row/column labels and numeric values.
        title: Plot title.
        cmap: Matplotlib colormap name.
        fmt: Number format string for annotations.
        figsize: Figure size.
        vmin: Colormap minimum.
        vmax: Colormap maximum.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(data.columns)))
    ax.set_yticks(range(len(data.index)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(data.index, fontsize=9)
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.values[i, j]
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=8,
                    color="white" if val > (data.values.max() * 0.65) else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def decompose_forecast(results: Dict[str, Any], pred_len: int = 8) -> pd.DataFrame:
    """Decompose the final forecast into its Chronos and fusion-head components.

    Shows how w blends the two sources at each timestep.

    Args:
        results: Output from MigasInternalsExtractor.run().
        pred_len: Number of forecast steps.

    Returns:
        DataFrame with columns: step, chronos_component, fusion_component,
        convex_weight, final_forecast.
    """
    w = results["convex_weights"][:pred_len]
    chronos_norm = results["chronos_forecast_normalized"][:pred_len]
    fusion_head = results["forecast_head_output"][:pred_len]

    mu = results["norm_params"]["mu"]
    sigma = results["norm_params"]["sigma"]

    # In normalized space: final = w * chronos_norm + (1-w) * fusion_head
    blended_norm = w * chronos_norm + (1 - w) * fusion_head

    # Denormalize
    blended_denorm = blended_norm * (sigma + 1e-8) + mu

    return pd.DataFrame({
        "step": [f"t+{i+1}" for i in range(pred_len)],
        "w (Chronos weight)": w,
        "1-w (Fusion weight)": 1 - w,
        "Chronos (norm)": chronos_norm,
        "Fusion head (norm)": fusion_head,
        "Blended (norm)": blended_norm,
        "Final forecast": results["forecast"][:pred_len],
        "Reconstructed": blended_denorm,
    })
