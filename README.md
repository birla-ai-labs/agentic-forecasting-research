# Migas 1.5 Reverse-Engineering Research

Experiments to understand how Migas 1.5's text conditioning actually works: what the gated attention fusion does, how the convex combination weights behave, and whether the text channel meaningfully steers forecasts on real data.

## Architecture

```
Raw Context (e.g. 41 weekly values)
        │
        ▼
   [Normalization]  (mu, sigma saved for denormalization)
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
| `03_real_data_validation.ipynb` | Real financial + cement data | Baselines, predictive steering ladder, factual bottleneck test |

## Key Findings

### NB1: Architecture probing

**The gate is basically static.** Only ~11% of the 512 gate dimensions exceed 0.5. Cosine similarity between gate activations across completely different text inputs is above 0.96, so this isn't a dynamic content-aware filter. It looks like a fixed mask learned during training.

**Factual text dominates predictive text roughly 16:1** in cross-attention weights. Swapping bullish for bearish in the predictive section barely shifts the forecast. Changing the factual section moves things by ~16x more.

**The model is mostly Chronos.** The convex weight w stays above 0.80 across all experiments. Text accounts for less than 20% of the final output at every forecast step.

**Empty text is surprisingly disruptive.** Providing nothing produces the largest forecast shifts (up to 88.9), because a zero-norm FinBERT embedding is maximally far from any real embedding and confuses the fusion head.

**Scale is not invariant.** The same pattern described at different numerical scales produces different FinBERT embeddings, and the relative text shift roughly doubles going from tiny (0-1) to large (0-1M) scale.

### NB2: Synthetic patterns

**Rich, data-specific factual text paradoxically suppresses text influence.** When the factual section contains actual statistics computed from the series, w climbs to 0.91-0.97 (very high Chronos trust). Generic or vague factual text lowers w to 0.79-0.91, which opens up the text channel more. This "factual bottleneck" effect is consistent across all 8 synthetic patterns.

### NB3: Real data validation

8-step weekly forecasts on S&P 500, Gold, Bitcoin, and a cement SKU (high-volume, stable demand). Results are % MAE improvement vs Chronos-2 baseline (positive = better than Chronos-2).

**Exp 3A: Baselines**

Chronos-2 and Migas with neutral text (no directional signal) produce similar accuracy. Neither has a consistent edge.

| Asset | Chronos-2 MAE | Migas neutral MAE |
|-------|--------------|-------------------|
| S&P 500 | 187.8 | 196.2 |
| Gold | 23.1 | 23.4 |
| Bitcoin | 3328.9 | 3437.3 |
| Cement | 166045.7 | 103746.9 |

Cement is the outlier here: even neutral text gives a big improvement, likely because the series has a strong structural pattern that FinBERT embeddings happen to encode well.

**Exp 3B: Predictive steering ladder**

Can progressively better text inputs steer the model toward better forecasts?

| Asset | generic+neutral | rich+neutral | rich+accurate | LLM-generated | rich+exact |
|-------|----------------|-------------|--------------|--------------|-----------|
| S&P 500 | -4.5% | -8.2% | -2.7% | -3.6% | -3.1% |
| Gold | -1.3% | +3.7% | +7.4% | -0.9% | +7.4% |
| Bitcoin | -3.3% | -15.4% | -22.2% | -0.4% | -6.4% |
| Cement | +37.5% | +49.8% | +43.8% | +27.0% | +41.4% |

Cement benefits consistently from richer text. Financial assets are much harder to steer, and Bitcoin in particular gets worse as text quality improves, which is consistent with the convex weight suppression story from NB1/NB2.

**Exp 3C: Factual bottleneck test**

All three scenarios here use an exact oracle predictive value (the actual future price). The only variable is how much factual context we provide.

| Asset | empty+exact | generic+exact | rich+exact |
|-------|------------|--------------|-----------|
| S&P 500 | -1.5% | -1.8% | -3.1% |
| Gold | +9.5% | +0.5% | +7.4% |
| Bitcoin | -7.8% | -1.2% | -6.4% |
| Cement | +47.1% | +23.2% | +41.4% |

On Gold and Cement, `empty+exact` beats `generic+exact`, which beats nothing. Providing richer factual context raises w and reduces how much of the oracle predictive signal the model actually uses. This confirms the NB1/NB2 bottleneck finding on real data.

## Shared Utilities

`experiment_utils.py` provides:
- `MigasInternalsExtractor`: hook-based extraction of all intermediate representations (convex weights, gate values, attention weights, embeddings)
- `GENERIC_FACTUAL`: a deliberately vague factual description that keeps w low and opens the text channel
- `load_financial_data()`: yfinance wrapper for weekly close prices
- `load_cement_skus()`: multi-SKU selector sorted by coefficient of variation
- `generate_synthetic_series()`: 8 canonical synthetic patterns
- `describe_series()` / `build_summary()`: prompt construction helpers
- `text_shift()` / `directional_accuracy()`: forecast quality metrics
- Plotting: `plot_forecast_comparison`, `plot_convex_weights`, `plot_gate_values`, `plot_heatmap`, etc.

## Setup

```bash
# Install dependencies
pip install yfinance

# Model weights download automatically from HuggingFace
# Repo: Synthefy/migas-1.5
```

Run notebooks in order: NB1, NB2, NB3. Each builds on findings from the previous.