# agentic-forecasting-research

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