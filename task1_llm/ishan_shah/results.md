# Task 1 Results — Ishan Shah

## What I built

A character-level, decoder-only GPT implemented entirely from scratch (no
`torch.nn.MultiheadAttention`, no `torch.nn.Transformer*`) and trained on an independent
100,000-character slice of the shared TinyStories corpus (`task1_llm/data/tinystories_raw.txt`,
offset 0), validated on the following 10,000 characters.

## Architecture

- **Block style:** Pre-LayerNorm — `x = x + Attn(LN(x))`, `x = x + FFN(LN(x))`. Pre-LN keeps
  gradients better-scaled through depth and is the choice used by GPT-2 onward.
- **Self-attention:** hand-written scaled dot-product multi-head attention with a
  registered lower-triangular causal mask (`torch.tril`), so the model can never attend to
  future positions.
- **Feed-forward:** `Linear(128→512) → GELU → Linear(512→128)`, dropout 0.1.
- **Embeddings:** learned token embedding (128-dim) + learned positional embedding
  (128 positions), summed and dropped out before the first block.
- **LM head:** weight-tied to the token embedding matrix (reduces parameter count and is a
  well-known regularizer for small LMs).
- **Config:** `n_layer=4, n_head=4, n_embd=128, block_size=128, dropout=0.1` → 821,632
  parameters.

## Why these choices

- 4 layers / 4 heads / 128-dim is a standard "tiny GPT" configuration (similar order of
  magnitude to nanoGPT's smallest configs) — big enough to show clear learning curves within
  a modest compute budget, small enough to train in under two minutes on a laptop GPU.
- Weight tying was chosen to cut parameter count meaningfully at this scale (the embedding
  table would otherwise be ~30% of all parameters) without hurting quality.
- Pre-LN was chosen over Post-LN specifically so the two teammates' implementations would
  differ architecturally, not just numerically (Charvee Saraiya's uses Post-LN — see her
  `results.md` for the comparison).

## Training

- Loss: cross-entropy over next-character prediction.
- Optimizer: AdamW, weight decay 0.01, peak LR 3e-4.
- Schedule: linear warmup over the first 10% of steps, then cosine decay to 10% of peak LR.
- Gradient clipping at global norm 1.0.
- 60 epochs (defined as one pass over `100,000 // block_size` non-overlapping windows,
  sampled with replacement each step) — well above the assignment's 10-epoch minimum, chosen
  because training is fast enough (~74s total) to afford it and produces a visibly converged
  loss curve (see `outputs/loss_curve.png`).

## Results (see `metrics_report.csv` for the full row)

| Metric | Value |
|---|---|
| Final train loss | 1.935 |
| Final val loss | 1.984 |
| Perplexity (val) | 7.27 |
| Bits-per-character | 2.86 |
| Generalization gap | 0.049 |
| Top-1 next-char accuracy | 40.4% |
| Distinct-1 / 2 / 3 | 0.025 / 0.134 / 0.325 |
| Repeated 4-gram rate | 52.3% |
| Grad norm (mean / max) | 0.87 / 4.43 |
| NaN count | 0 |
| Parameters | 821,632 |
| Train throughput | ~79.7K tokens/sec |
| Generation throughput | ~97 tokens/sec |
| Peak GPU memory | 493.6 MB |
| Total training time | 74.0 s |

The small generalization gap (0.049) and zero NaNs indicate stable training with no
overfitting at this scale — expected, since dropout (0.1) and weight decay are both active
and the model is comfortably under-parameterized relative to what would be needed to
memorize 100K characters.

## Failure analysis

See [`failure_analysis.md`](failure_analysis.md) for three annotated generation failures
(repetition under greedy decoding, broken-word grammar under sampling, and a training-format
artifact leaking into generated text).

## How this compares to Charvee Saraiya's model

Charvee Saraiya's model is deeper-but-narrower (6 layers, 96-dim vs. my 4 layers, 96-dim... actually
128-dim), Post-LN instead of Pre-LN, ReLU instead of GELU, untied instead of tied head, and
trained on an independent, non-overlapping 100K-character slice starting at character
150,000 of the shared corpus. Her model reached a lower final validation loss (1.73 vs. 1.98)
and higher next-char accuracy (47.8% vs. 40.4%), at the cost of a larger generalization gap
(0.106 vs. 0.049) and a much higher max gradient norm (14.1 vs. 4.4) — consistent with a
deeper, untied-head network being more expressive but also less stable during training. Full
side-by-side numbers are in the team report's Task 1 comparison table.
