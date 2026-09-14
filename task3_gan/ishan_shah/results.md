# Task 3 Results — Ishan Shah

## What I built

A CycleGAN for unpaired Monet ↔ Photo translation, implemented from the paper's core
components: two ResNet-style generators (monet→photo and photo→monet) and two PatchGAN
discriminators, trained with adversarial + cycle-consistency + identity loss on my own
independent 300-Monet / 300-photo subsample (seed 42) from the shared data pool, at 128×128
resolution — a deliberately reduced scale (vs. the paper's 256×256, thousands of images,
hundreds of epochs) to fit training within a single GPU session.

## Architecture

- **Generator (`G_AB`, `G_BA`):** ResNet-style — a 7×7 reflection-padded stem, two stride-2
  downsampling convolutions, 6 residual blocks at the bottleneck resolution, two stride-2
  transposed-convolution upsampling layers, and a final 7×7 conv + Tanh. InstanceNorm2d
  throughout (the standard choice for style transfer — normalizes per-image style
  statistics, unlike BatchNorm which mixes statistics across the batch).
- **Discriminator (`D_A`, `D_B`):** a 70×70-receptive-field PatchGAN — 4 stride-2 (then one
  stride-1) convolutions with InstanceNorm2d and LeakyReLU, judging local image patches
  rather than the whole image as real/fake.
- **Adversarial loss:** LSGAN (mean-squared error against 1/0 targets) rather than vanilla
  BCE — known to produce more stable gradients and reduce vanishing-gradient issues in the
  discriminator.
- **Cycle-consistency loss:** L1 between `x` and `G_BA(G_AB(x))` (and the reverse), weighted
  λ_cycle=10.
- **Identity loss:** L1 between `G_BA(real_A)` and `real_A` (and the reverse), weighted
  λ_identity=5 — encourages the generator to act as identity when already given an image from
  the target domain, which helps preserve color composition.
- **Replay buffer:** a 50-image history buffer feeds the discriminator a mix of the latest
  and previously-generated fakes (as in the original paper), reducing oscillation.

## Why these choices

This is the closest-to-canonical CycleGAN configuration, chosen as my baseline
implementation to validate correctness against the well-documented original paper before
any deviation. Charvee's implementation deliberately diverges (U-Net generator, BatchNorm,
vanilla GAN loss, no identity loss) to give the team a genuine architecture-choice
comparison rather than two near-identical models.

## Training

- 128×128 resolution, batch size 4, 40 epochs, Adam (β=0.5, 0.999), LR 2e-4 with linear decay
  to 0 over the second half of training (epochs 20–40).
- 300 Monet paintings / 300 photos (independent random sample, seed 42) for training; a
  held-out 40/40 sample from the dataset's own test split for all evaluation below.

## Results (see `metrics_report.csv` / `full_metrics_report.csv` for the full row)

| Metric | Monet→Photo | Photo→Monet |
|---|---|---|
| FID | 6.29 | 4.18 |
| KID (mean) | 11.93 | 7.54 |
| Cycle-reconstruction L1 | 0.176 (monet) | 0.163 (photo) |
| LPIPS | 0.290 | 0.437 |
| Content cosine similarity | 0.788 | 0.640 |

| Training metric | Value |
|---|---|
| Final loss G / D | 2.315 / 0.116 |
| Final cycle loss / identity loss | 0.118 / 0.124 |
| Grad norm G (mean / max) | 22.06 / 30.06 |
| NaN count | 0 |
| Total parameters (2G + 2D) | 21,204,872 |
| Training time | 1,012 s (~16.9 min) |
| Train images/sec | 23.7 |
| Gen images/sec | 733.7 |
| Peak GPU memory | 2,186 MB |

FID/KID are computed on only 40 held-out images per domain (a deliberately small sample to
keep evaluation fast), so treat the absolute FID/KID values as noisy, directional estimates
rather than precise scores — with n=40 the sampling variance on FID is substantial.

## Kaggle leaderboard

`evaluate_local.py` generates `images.zip` (200 photo→Monet translations, direct model
inference, no manual editing) ready for `kaggle competitions submit`. Actual submission is
**pending** — requires my own Kaggle account/API token (see `submission.csv`).

## Failure analysis

See [`failure_analysis.md`](failure_analysis.md).

## How this compares to Charvee's model

Charvee's U-Net/BatchNorm/vanilla-GAN model achieved notably better FID scores in both
directions (3.28/1.88 vs. my 6.29/4.18) and trained roughly 1.7× faster per epoch on
average, but her generator's gradient norm grew steadily across training (53→315) with no
gradient clipping, an instability signature my LSGAN+InstanceNorm+identity-loss combination
does not show (my grad norm mean/max is a comparatively controlled 22.1/30.1, and actually
*decreases* over the second half of training as the LR decays). This is a genuine,
interesting team finding: her setup reaches better *distributional* similarity (FID) faster,
at the cost of training stability that could plausibly diverge on a longer run — worth
flagging explicitly in the demo/viva.
