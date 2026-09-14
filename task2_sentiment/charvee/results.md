# Task 2 Results — Charvee

## What I built

Three sentiment classifiers for Yelp Polarity, all with embeddings learned from scratch,
deliberately differentiated from my teammate's lineup in both preprocessing and
architecture. Trained/evaluated on my own independent balanced subsample: 12,000 train /
3,000 test reviews (seed 7).

| | Baseline | Experimental 1 | Experimental 2 |
|---|---|---|---|
| Name | Max-pool + MLP | GRU | Dilated CNN |
| Mechanism | Max-pool over embeddings → 1-hidden-layer MLP | Unidirectional GRU (hidden=72), final hidden state | Stacked dilated Conv1d (dilations 1/2/4) → global max-pool |
| Embedding dim | 80 | 80 | 80 |
| Params | 962,625 | 993,337 | 1,000,193 |

## Preprocessing (deliberately different from teammate's)

- Lowercasing, regex-based token extraction, the same ~90-word stopword list as my
  teammate — but **no stemming**, to test whether the extra normalization step actually
  helps or just adds noise.
- Own 12,000-word vocabulary (smaller than teammate's 15,000), sequences padded/truncated to
  150 tokens (shorter window than teammate's 200).
- Class balance 50/50 by construction; review lengths right-skewed (median 97 words, mean
  134, max 1009) — very similar distribution to the teammate's independent sample, as
  expected since both are drawn from the same shared corpus.
- No missing or empty-after-cleaning reviews.

## Why these choices

- The max-pool + MLP baseline was chosen specifically to differ from a plain-linear /
  mean-pool baseline: max-pooling picks out the single most extreme embedding dimension per
  feature (rather than averaging), and the extra hidden layer gives it a small amount of
  non-linear capacity a pure linear baseline lacks.
- GRU was chosen over (bi-)LSTM to test a lighter-weight recurrent unit with roughly half the
  gates of an LSTM, unidirectional (cheaper, tests whether backward context is actually
  necessary).
- Dilated CNN was chosen over parallel multi-kernel CNN to grow the receptive field
  *without* adding parallel branches — each layer's dilation (1→2→4) doubles context reach
  while keeping a single narrow kernel (3) per layer.

## Results (full row-per-model detail in `metrics_report.csv`)

| Model | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | MCC | Brier | ECE | Train time (s) | Examples/sec | Peak mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_maxpool_mlp | 0.8090 | 0.8089 | 0.8963 | 0.8967 | 0.6189 | 0.1307 | 0.0219 | 3.52 | 20,478 | 42.5 |
| experimental_gru | 0.8700 | 0.8700 | 0.9444 | 0.9433 | 0.7400 | 0.0998 | 0.0677 | 20.46 | 3,518 | 67.4 |
| experimental_dilated_cnn | 0.8463 | 0.8463 | 0.9247 | 0.9274 | 0.6932 | 0.1236 | 0.0876 | 4.36 | 16,499 | 65.6 |

95% bootstrap CIs and per-slice metrics are in `metrics_report.csv`. Unlike the teammate's
models, my three models' accuracy CIs are **non-overlapping** between the baseline
([0.795,0.823]) and both experimental models (GRU [0.857,0.882], dilated CNN
[0.834,0.860]) — a real, not just noise-level, improvement from adding model capacity.

**Paired McNemar test** (baseline vs. each experimental, `outputs/mcnemar_results.json`):
both experimental models are **significantly** better than the baseline (GRU: p=1.5e-14;
dilated CNN: p=5.6e-6) — a much stronger signal than my teammate's TextCNN/BiLSTM-vs-NBOW
comparison, which was not significant.

Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA).

## Comparative analysis (2.3)

- **Strengths:** the GRU is my best model on every metric (87.0% accuracy, 0.944 ROC-AUC)
  while being the cheapest of the two experimental models is not true — it's actually the
  slowest to train (20.5s vs. 4.4s for the dilated CNN) since recurrence can't be
  parallelized across timesteps the way convolutions can, but it delivers the best accuracy.
- **Weaknesses:** the simple max-pool + MLP baseline is meaningfully worse than either
  experimental model (McNemar confirms this is real, not noise), suggesting that — unlike
  my teammate's NBOW baseline — max-pooling alone loses too much information for a
  single-hidden-layer MLP to recover.
- **Limitations:** calibration (ECE) gets *worse* as accuracy improves — baseline ECE 0.022,
  GRU ECE 0.068, dilated CNN ECE 0.088 — meaning my more accurate models are also more
  overconfident. The "long" review slice again has the highest error rate for the baseline
  and dilated CNN (19.9%, 17.1%) though the GRU narrows this gap considerably (13.9% long vs.
  12.5% short/medium).
- **Future work:** apply temperature scaling to the GRU and dilated CNN to fix the
  calibration regression, and try bidirectional GRU to see if it closes the remaining gap
  with the teammate's BiLSTM without paying its full training-time cost.

## Failure analysis

See [`failure_analysis.md`](failure_analysis.md) for the 20-case manual error review on the
GRU model.

## How this compares to Ishan's models

See the team report's Task 2 comparison table for the full six-model side-by-side. My best
model (GRU, 87.0%) slightly outperforms his best model (BiLSTM, 88.3%)... actually his BiLSTM
edges mine out on raw accuracy, but my baseline-to-experimental *improvement* is much larger
and statistically significant (his was not). This suggests stemming (his pipeline) versus no
stemming (mine) had less effect than architecture choice, and that his higher token budget
(200 vs. 150) and larger vocabulary (15K vs. 12K) may explain his baseline already starting
from a stronger position.
