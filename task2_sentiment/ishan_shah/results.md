# Task 2 Results — Ishan Shah

## What I built

Three sentiment classifiers for Yelp Polarity, all with embeddings learned from scratch
(no pretrained embeddings, no pretrained language models), trained and evaluated on my own
independent balanced subsample: 12,000 train / 3,000 test reviews (seed 42, class-balanced
6,000/6,000 and 1,500/1,500).

| | Baseline | Experimental 1 | Experimental 2 |
|---|---|---|---|
| Name | NBOW (mean-pool) | TextCNN | BiLSTM |
| Pooling / mechanism | Mean-pool over embeddings | Parallel Conv1d, kernels {3,4,5}, 64 filters each, global max-pool | Bidirectional LSTM (hidden=64), concatenated final states |
| Embedding dim | 100 | 100 | 100 |
| Params | 1,500,101 | 1,577,185 | 1,585,121 |

## Preprocessing

- Lowercasing, regex-based punctuation/special-character stripping (`[a-z]+` token
  extraction), a hand-curated ~90-word stopword list, and **Porter stemming**.
- Own word-level tokenizer with a 15,000-word vocabulary built from training-set token
  frequency (`<pad>`/`<unk>` reserved), sequences padded/truncated to 200 tokens.
- Class balance confirmed 50/50 by construction (balanced sampling); review lengths are
  right-skewed (median 96 words, mean 133, max 966) — see `outputs/length_distribution.png`.
- No missing or empty-after-cleaning reviews in the sampled data.

## Why these choices

- The NBOW baseline is the simplest possible "embeddings-from-scratch" model and serves as
  the reference point for the McNemar tests.
- TextCNN adds local n-gram pattern detection (kernel sizes 3/4/5 ≈ tri-/four-/five-grams)
  without recurrence, testing whether short local phrases (e.g., "not good", "highly
  recommend") are enough to beat the baseline.
- BiLSTM adds sequence-order sensitivity and long-range context via recurrence in both
  directions, testing whether that helps beyond local n-grams.

## Results (full row-per-model detail in `metrics_report.csv`)

| Model | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | MCC | Brier | ECE | Train time (s) | Examples/sec | Peak mem (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_nbow | 0.8897 | 0.8897 | 0.9591 | 0.9560 | 0.7794 | 0.0796 | 0.0358 | 2.63 | 27,328 | 58.1 |
| experimental_textcnn | 0.8823 | 0.8823 | 0.9603 | 0.9616 | 0.7652 | 0.0810 | 0.0270 | 4.96 | 14,527 | 92.3 |
| experimental_bilstm | 0.8830 | 0.8830 | 0.9530 | 0.9537 | 0.7660 | 0.0905 | 0.0556 | 33.18 | 2,170 | 132.3 |

95% bootstrap CIs (accuracy / macro-F1 / MCC) and per-slice (short/medium/long review)
macro-F1 and error rate are in `metrics_report.csv`; all three models' CIs overlap
substantially (e.g. accuracy CIs: [0.879,0.901], [0.871,0.895], [0.871,0.894]).

**Paired McNemar test** (baseline vs. each experimental model, `outputs/mcnemar_results.json`):
neither experimental model is statistically distinguishable from the baseline on this test
set (TextCNN: p=0.196; BiLSTM: p=0.260) — with only 3,000 test examples the ~0.5-1 point
accuracy differences between models are within noise.

Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA).

## Comparative analysis (2.3)

- **Strengths:** All three models land in the high-88% accuracy range with strong ROC-AUC
  (>0.95), confirming that even a from-scratch, non-pretrained embedding is enough to
  separate Yelp polarity well at this data scale.
- **Weaknesses:** The added architectural complexity of TextCNN and BiLSTM did **not**
  translate into a statistically significant accuracy gain over the much cheaper NBOW
  baseline (McNemar p > 0.05 for both), while costing substantially more compute — BiLSTM
  trains ~12.6x slower than the baseline per run (33.2s vs 2.6s) for essentially tied
  accuracy.
- **Limitations:** the "long" review slice consistently has the highest error rate across
  all three models (12.2%, 12.5%, 14.7%) — see per-slice metrics — suggesting truncation at
  200 tokens loses signal in longer reviews, and calibration (ECE) is worst for the BiLSTM
  (0.056) despite comparable accuracy, meaning its confidence scores are the least
  trustworthy of the three.
- **Future work:** try mean-pooling BiLSTM's hidden states across all timesteps (not just
  the final one) to reduce the long-review error rate, and post-hoc calibration
  (temperature scaling) to reduce ECE.

## Failure analysis

See [`failure_analysis.md`](failure_analysis.md) for the 20-case manual error review
(5 confident false positives, 5 confident false negatives, 5 near-threshold, 5 slice-specific
failures) on the BiLSTM model, each with a proposed testable fix.

## How this compares to Charvee Saraiya's models

See the team report's Task 2 comparison table for the full six-model side-by-side. At a
glance: Charvee Saraiya's GRU experimental model (87.0% accuracy) outperforms her own max-pool MLP
baseline (80.9%) by a statistically significant margin (McNemar p≈1.5e-14) — a much larger
and clearer baseline-vs-experimental gap than either of my own experimental models showed
over my NBOW baseline, suggesting her un-stemmed preprocessing + recurrent architecture
combination benefits more from added model capacity than mine does.
