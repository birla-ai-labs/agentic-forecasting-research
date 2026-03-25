# Migas 1.5 Reverse-Engineering Experiment Plan

A systematic series of experiments across 3 notebooks to understand how Migas 1.5's text conditioning, gated attention fusion, and convex combination weights actually behave — using synthetic controlled inputs and diverse real datasets.

---

## Key Architectural Insight (Why Your Bullish ≈ Bearish)

Before the experiments, note a critical confound in the original notebook: the **neutral** summary used a *generic* factual section (`"The series shows typical weekly fluctuations..."`), while bullish and bearish **shared the same data-rich factual section** (with actual numbers, ranges, means). Since FinBERT embeds these very differently, the forecast divergence was likely driven by the **factual embedding**, not the predictive signals. Experiments 1A–1C directly test this.

---

## Notebook 1: `01_architecture_probing.ipynb` — Model Internals

**Goal:** Extract and visualize every intermediate representation to understand where text influence enters and how much it matters.

### Exp 1A — Factual vs. Predictive Signal Isolation
- Use the **same cement SKU context** from the original experiment
- Construct 4 summaries that cross factual × predictive:
  - **Rich Factual + Bullish Predictive** (original bullish)
  - **Rich Factual + Bearish Predictive** (original bearish)
  - **Generic Factual + Bullish Predictive** (new)
  - **Generic Factual + Bearish Predictive** (new)
- Also test: **Rich Factual + Neutral Predictive** vs **Generic Factual + Neutral Predictive**
- **Metric:** Forecast delta between pairs that share the same factual but differ in predictive (and vice versa)
- **Hypothesis:** Factual section drives most of the forecast shift; predictive signals have minimal effect

### Exp 1B — Convex Weight Extraction
- Write a helper function that hooks into `model.forward()` to capture:
  - `w` from `convex_weight_net(fused)` — the per-timestep blend weights
  - `timeseries_forecast` — raw Chronos output (normalized)
  - `forecast_head(fused)` — the pure fusion-head output (before convex blending)
- Run this for all 6 summaries from Exp 1A
- **Visualize:** Bar chart of `w` across the 8 (or 16) forecast steps; heatmap across scenarios
- **Key question:** Is `w` close to 1.0 (text ignored) or does it vary meaningfully?

### Exp 1C — Gate & Attention Inspection
- Hook into `GatedAttentionFusion.forward()` to extract:
  - **Attention weights** from `cross_attn` (how much TS queries attend to factual vs. predictive tokens)
  - **Gate values** from `gate_net` (the sigmoid gate that controls residual blending)
- Compare gate magnitudes across bullish / bearish / neutral / empty-text scenarios
- **Visualize:** Heatmap of gate values (512-dim) averaged or PCA-reduced

### Exp 1D — Text Embedding Similarity Analysis
- Extract raw FinBERT embeddings for ~10 different summary texts (bullish, bearish, neutral, contradictory, empty, numeric-heavy, vague, etc.)
- Compute pairwise **cosine similarity** matrix
- Project into 2D with t-SNE or PCA
- **Goal:** Understand which text variations FinBERT actually differentiates vs. collapses

### Exp 1E — Ablation: Zero/Random Text Embeddings
- Run forward pass with:
  - Normal text embeddings
  - All-zero text embeddings (both factual and predictive)
  - Random Gaussian text embeddings
  - Swapped: factual ↔ predictive embeddings
- Compare forecasts to quantify the **maximum possible text effect**

---

## Notebook 2: `02_synthetic_patterns.ipynb` — Controlled Data Experiments

**Goal:** Isolate how Migas responds to different time series shapes × different text signals, using fully controlled synthetic data.

### Synthetic Series Generator
Create a utility function producing 8 canonical patterns (all ~50 steps, values in [0, 1000] range):

| Pattern | Description |
|---------|-------------|
| **Constant** | Flat line at 500 |
| **Linear Trend Up** | Steady increase from 200 → 800 |
| **Linear Trend Down** | Steady decrease from 800 → 200 |
| **Sine Wave** | Periodic oscillation (period ~12 steps) |
| **Random Walk** | Gaussian increments from starting point |
| **Volatile / High-CV** | Large random jumps (std ≈ 0.5 × mean) |
| **Intermittent** | Mostly zeros with occasional spikes |
| **Regime Change** | Flat at 300 for 30 steps, then jumps to 700 |

### Prompt Battery
For each pattern, test a standardized set of summaries:

| Prompt ID | Factual Section | Predictive Section |
|-----------|----------------|-------------------|
| P1 | Accurate description of the pattern | "Expect continuation" |
| P2 | Accurate description | "Sharp reversal expected, 50% move in opposite direction" |
| P3 | Accurate description | "Extreme surge, 200% increase expected" |
| P4 | **Misleading** (claims trend when flat) | "Expect continuation" |
| P5 | Empty / minimal | Empty / minimal |
| P6 | Accurate description | Contradictory ("both increase and decrease likely") |

### Exp 2A — Text Sensitivity Matrix
- Run all 8 patterns × 6 prompts = **48 forecasts**
- For each, also extract convex weights and Chronos-only baseline
- **Metric:** `text_shift = MAE(migas_forecast, chronos_forecast)` — how far text pushed the forecast from the univariate baseline
- **Visualize:** 8×6 heatmap of text_shift values
- **Key question:** Which patterns are most/least susceptible to text conditioning?

### Exp 2B — Convex Weight Patterns
- Extract `w` for all 48 runs
- **Visualize:** Weight profiles (w vs. step) grouped by pattern type
- **Key question:** Does the model learn to trust Chronos more for certain pattern types?

### Exp 2C — Magnitude Sensitivity
- For the **linear trend up** pattern, test predictive signals with increasing intensity:
  - "slight increase expected" → "moderate increase" → "strong increase" → "extreme surge, 500% jump"
- Plot forecast vs. intensity level
- **Key question:** Is the text influence monotonic with signal strength, or does it saturate/plateau?

### Exp 2D — Scale Sensitivity
- Take the same pattern shape but at different value scales: [0-1], [0-1000], [0-1M]
- Test with the same text summaries
- **Key question:** Does normalization make the model scale-invariant, or do large numbers behave differently?

### Exp 2E — Context Length Sensitivity
- Use the same series but vary context length: 10, 20, 30, 40, 50 steps
- **Key question:** How does context length affect the balance between Chronos and text?

---

## Notebook 3: `03_real_data_validation.ipynb` — Real Dataset Experiments

**Goal:** Validate synthetic findings on real data and test domain transfer.

### Recommended Real Datasets

All should be simple to download (CSV or API), weekly/daily frequency, ~50-200 data points:

| Dataset | Source | Why Useful |
|---------|--------|------------|
| **S&P 500** (^GSPC) | `yfinance` | Financial — model's training domain; strong trend + volatility |
| **Gold** (GLD) | `yfinance` | Financial — mentioned in Migas docs; safe-haven dynamics |
| **Bitcoin** (BTC-USD) | `yfinance` | Crypto — extreme volatility, regime changes |
| **US 10Y Treasury Yield** | `yfinance` (^TNX) | Macro — mean-reverting, rate-driven |
| **Electricity demand** (ENTSOE or UCI) | UCI ML Repo / Kaggle | Energy — strong seasonality, non-financial domain |
| **Australian beer production** | `statsmodels.datasets` | Classic stationary + seasonal |
| **Air passengers** | `statsmodels.datasets` | Classic trend + seasonality |
| **Walmart weekly sales** | Kaggle M5 subset | Retail — intermittent, promotional spikes |

### Exp 3A — Financial Domain (Home Turf)
- Download S&P 500, Gold, BTC weekly close prices (last ~60 weeks)
- For each, test 3 prompts: LLM-generated (via `generate_summary`), manually bullish, manually bearish
- Extract convex weights, compare text sensitivity to Notebook 2 findings
- **Goal:** Establish baseline text influence on the model's native domain

### Exp 3B — Non-Financial Domain Transfer
- Use electricity demand and beer production data
- Test with:
  - Financial-style language ("bullish outlook, strong momentum")
  - Domain-appropriate language ("seasonal peak expected, summer demand surge")
  - Mismatched domain language ("earnings beat expectations" for electricity data)
- **Key question:** Does the model respond differently to financial vs. non-financial text?

### Exp 3C — Cement Data Revisited
- Rerun the original experiment with the confound fixed:
  - All scenarios share the **same rich factual section**
  - Only predictive signals vary
- Compare results to the original notebook
- **Goal:** Definitively answer "how much do predictive signals alone move the forecast?"

### Exp 3D — Multi-SKU Comparison
- Pick 3 cement SKUs with different CV values (high, medium, low volatility)
- Run the same prompt battery on all 3
- **Key question:** Does data volatility affect text sensitivity (confirming/denying Notebook 2 synthetic findings)?

### Exp 3E — LLM Summary Quality Ladder
- For one financial asset, generate summaries with:
  - `generate_summary` (full LLM pipeline)
  - Hand-crafted expert summary (detailed, accurate)
  - Hand-crafted vague summary
  - Completely random text
  - Empty string
- Plot forecast quality (MAE vs. ground truth) as a function of summary quality
- **Goal:** Quantify the practical value of good summaries

---

## Shared Infrastructure

Each notebook will import a common `experiment_utils.py` module with:
- `extract_internals(pipeline, context, summary)` → dict with convex weights, gate values, attention weights, embeddings
- `generate_synthetic_series(pattern, length, scale)` → DataFrame in Migas format
- `text_shift_metric(migas_fc, chronos_fc)` → scalar measuring text influence
- `plot_forecast_comparison(...)` → standardized visualization
- `plot_weight_heatmap(...)` → convex weight visualization


## Execution Order

1. **Notebook 1** first — understanding the architecture unlocks interpretation of everything else
2. **Notebook 2** second — synthetic experiments give clean, unambiguous results
3. **Notebook 3** last — validates that synthetic findings hold in practice

---

## Expected Deliverables Per Notebook
- Clear markdown headers per experiment
- Quantitative metrics tables
- Publication-quality matplotlib figures---

- Summary cell at the end with key findings


