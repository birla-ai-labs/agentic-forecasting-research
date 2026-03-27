# Migas 1.5 Reverse-Engineering Research

Systematic experiments to understand how Migas 1.5's text conditioning, gated attention fusion, and convex combination weights actually behave.

## Architecture

```
Raw Context (e.g. 41 weekly values)
        │
        ▼
   [Normalization]  ──── mu, sigma saved for later
        │
        ▼
   Normalized Context
        │
    ┌───┴───────────────────────┐
    ▼                           ▼
[Chronos-2]              [Summary Text]
(frozen, univariate)      "FACTUAL SUMMARY: ..."
    │                     "PREDICTIVE SIGNALS: ..."
    ▼                           │
Chronos Forecast          ┌─────┴─────┐
(pred_len × 1,           ▼           ▼
 normalized)         [FinBERT]    [FinBERT]
    │              (fact text)  (pred text)
    ▼                 │           │
[ts_embedder]         ▼           ▼
ResidualBlock     768-dim      768-dim
    │             raw emb      raw emb
    ▼                │           │
 512-dim         [fact_embedder] [pred_embedder]
 h_ts            2-layer MLP    2-layer MLP
    │                │           │
    ▼                ▼           ▼
[LayerNorm]     [LayerNorm]  [LayerNorm]
    │                │           │
    ▼                ▼           ▼
 h_ts_norm      h_fact_norm  h_pred_norm
    │                │           │
    └────────┬───────┘───────────┘
             ▼
   [GatedAttentionFusion]
             │
             ▼
        fused (512-d)
             │
     ┌───────┴───────┐
     ▼               ▼
[forecast_head]  [convex_weight_net]
     │               │
     ▼               ▼
  fusion_fc (pred_len)   w (pred_len, sigmoid)
     │               │
     └───────┬───────┘
             ▼
   final = w × Chronos + (1−w) × fusion_fc
             │
             ▼
      [Denormalize: × sigma + mu]
             │
             ▼
      Final Forecast (original scale)
```

## Notebooks

| Notebook | Focus | Key Experiments |
|----------|-------|-----------------|
| `01_architecture_probing.ipynb` | Model internals | Factual vs predictive isolation, convex weight extraction, gate inspection, embedding similarity, ablation |
| `02_synthetic_patterns.ipynb` | Controlled synthetic data | Text sensitivity matrix (8×6), convex weight patterns, magnitude/scale/context-length sensitivity |
| `03_real_data_validation.ipynb` | Real financial + cement data | Factual bottleneck test, predictive steering, cement confound fix, multi-SKU volatility, quality ladder |

## Key Findings

1. **The gate is structurally sparse:** Only ~11% of the 512 gate dimensions exceed 0.5. The gate pattern is nearly identical across all text inputs (cosine sim > 0.96), meaning it learned a fixed mask during training rather than dynamically selecting based on content.

2. **Factual text dominates predictive text 16:1** in cross-attention weights. Changing the predictive section (bullish vs bearish) barely moves the forecast; changing the factual section produces 16× larger shifts.

3. **Rich factual text paradoxically suppresses text influence:** Data-specific factual descriptions cause w ≈ 0.91–0.97 (high Chronos trust), while generic/vague factual text lowers w to ≈ 0.79–0.91, opening the text channel. This is the "factual bottleneck."

4. **The model is fundamentally a Chronos model with a narrow text correction channel.** Across all experiments, the convex weight w stays above 0.80, meaning Chronos always contributes at least 80% of the final forecast.

5. **Empty text is the strongest disruptor:** Providing no text at all produces the largest text shifts (up to 88.9), because the zero-norm FinBERT embedding is maximally different from any real embedding, causing the fusion head to diverge.

6. **Scale is not perfectly invariant:** The same pattern described at different numerical scales produces different FinBERT embeddings, causing the relative text shift to roughly double from tiny (0–1) to large (0–1M) scales.

## Shared Utilities

`experiment_utils.py` provides:
- `MigasInternalsExtractor` — hook-based extraction of all intermediate representations
- `GENERIC_FACTUAL` — a deliberately vague factual description that lowers w
- `load_financial_data()` — yfinance wrapper for weekly close prices
- `load_cement_skus()` — multi-SKU selector sorted by CV
- `generate_synthetic_series()` — 8 canonical synthetic patterns
- `describe_series()` / `build_summary()` — prompt construction
- `text_shift()` / `directional_accuracy()` — metrics
- Plotting: `plot_forecast_comparison`, `plot_convex_weights`, `plot_gate_values`, `plot_heatmap`, etc.

## Setup

```bash
# Install dependencies (in your conda env)
pip install yfinance

# Model weights download automatically from HuggingFace
# Repo: Synthefy/migas-1.5
```

Run notebooks in order: NB1 → NB2 → NB3. Each builds on findings from the previous.