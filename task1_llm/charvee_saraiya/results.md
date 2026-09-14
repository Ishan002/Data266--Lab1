# Task 1 Results — Charvee Saraiya

## What I built

A character-level, decoder-only GPT implemented entirely from scratch (no
`torch.nn.MultiheadAttention`, no `torch.nn.Transformer*`), deliberately architected to
differ from my teammate's implementation, trained on an independent, non-overlapping
100,000-character slice of the shared TinyStories corpus (offset 150,000), validated on the
following 10,000 characters.

## Architecture

- **Block style:** Post-LayerNorm — `x = LN(x + Attn(x))`, `x = LN(x + FFN(x))`. This is the
  original ("Attention Is All You Need") normalization placement, as opposed to the
  Pre-LN style used in my teammate's model.
- **Self-attention:** hand-written scaled dot-product multi-head attention with separate
  `key`/`query`/`value` projection layers (rather than one fused QKV projection) and a
  registered causal mask.
- **Feed-forward:** `Linear(96→384) → ReLU → Linear(384→96)`, dropout 0.15 — ReLU instead of
  GELU.
- **Embeddings:** learned token embedding (96-dim) + learned positional embedding
  (96 positions).
- **LM head:** a separate, untied linear layer (no weight sharing with the embedding table)
  — a genuine architectural difference from the weight-tied head used in the teammate's
  model.
- **Config:** `n_layer=6, n_head=4, n_embd=96, block_size=96, dropout=0.15` → 698,205
  parameters — deeper but narrower than the teammate's 4-layer/128-dim model.

## Why these choices

- I chose to go deeper-and-narrower (6 layers × 96-dim vs. 4 layers × 128-dim) to test
  whether depth or width matters more at this tiny scale.
- Post-LN was chosen specifically to differ from the teammate's Pre-LN design and to see the
  known stability trade-off in practice: Post-LN networks are harder to train at depth
  because gradients aren't renormalized before entering each sublayer, which shows up in my
  results as a higher max gradient norm (14.1 vs. 4.4).
- An untied LM head was chosen to give the model independent input/output character
  representations, at the cost of ~74K extra parameters relative to a tied head.

## Training

- Loss: cross-entropy over next-character prediction.
- Optimizer: AdamW, weight decay 0.01, peak LR 5e-4 (higher than teammate's 3e-4, to
  compensate for the narrower hidden size).
- Schedule: linear warmup over the first 10% of steps, then **linear** decay to 0 (not
  cosine — another deliberate difference from the teammate's schedule).
- Gradient clipping at global norm 1.0.
- 60 epochs, batch size 48.

## Results (see `metrics_report.csv` for the full row)

| Metric | Value |
|---|---|
| Final train loss | 1.627 |
| Final val loss | 1.733 |
| Perplexity (val) | 5.66 |
| Bits-per-character | 2.50 |
| Generalization gap | 0.106 |
| Top-1 next-char accuracy | 47.8% |
| Distinct-1 / 2 / 3 | 0.021 / 0.133 / 0.331 |
| Repeated 4-gram rate | 51.4% |
| Grad norm (mean / max) | 0.98 / 14.09 |
| NaN count | 0 |
| Parameters | 698,205 |
| Train throughput | ~40.6K tokens/sec |
| Generation throughput | ~58.5 tokens/sec |
| Peak GPU memory | 273.1 MB |
| Total training time | 142.9 s |

## Failure analysis

See [`failure_analysis.md`](failure_analysis.md) for three annotated generation failures
(greedy repetition, broken-word grammar under sampling, and short-range-only coherence).

## How this compares to Ishan's model

My model reaches a lower final validation loss (1.73 vs. 1.98) and higher next-character
accuracy (47.8% vs. 40.4%) despite having fewer parameters (698K vs. 822K), which suggests
depth was more valuable than width for this task at this scale. The trade-off is a larger
generalization gap (0.106 vs. 0.049) and a much higher peak gradient norm (14.1 vs. 4.4) —
the expected cost of Post-LN training being less self-stabilizing than Pre-LN, especially at
6 layers. Training also took roughly twice as long per epoch (142.9s vs. 74.0s total) due to
the extra depth, despite the narrower hidden dimension and smaller batch size. Full
side-by-side numbers are in the team report's Task 1 comparison table.
