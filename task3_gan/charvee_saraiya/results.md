# Task 3 Results — Charvee Saraiya

## What I built

A CycleGAN for unpaired Monet ↔ Photo translation, deliberately architected to differ from
my teammate's implementation: a U-Net-style generator (encoder-decoder with skip
connections) rather than a ResNet-with-residual-blocks generator, BatchNorm2d rather than
InstanceNorm2d, vanilla GAN (BCEWithLogits) adversarial loss rather than LSGAN, and **no
identity loss term** — trained on my own independent 400-Monet / 400-photo subsample (seed
7) from the shared data pool, at 128×128 resolution.

## Architecture

- **Generator (`G_AB`, `G_BA`):** a 4-level U-Net encoder-decoder. Each encoder level
  halves spatial resolution via a stride-2 convolution (128→64→32→16→8→4 at the bottleneck);
  each decoder level uses a stride-2 transposed convolution and concatenates the
  corresponding encoder feature map via a skip connection, so fine spatial detail from early
  encoder layers reaches the decoder directly rather than only through the bottleneck.
  BatchNorm2d + LeakyReLU in the encoder, BatchNorm2d + ReLU (+ dropout on the first two
  decoder levels) in the decoder, final Tanh output.
- **Discriminator (`D_A`, `D_B`):** the same PatchGAN receptive-field design as the ResNet
  baseline, but with BatchNorm2d instead of InstanceNorm2d.
- **Adversarial loss:** vanilla GAN (`BCEWithLogitsLoss`) rather than LSGAN — the original
  GAN formulation, known to be more prone to vanishing gradients when the discriminator
  becomes confident, as a deliberate point of comparison against my teammate's LSGAN choice.
- **Cycle-consistency loss:** L1, weighted λ_cycle=8 (vs. teammate's 10).
- **No identity loss** (λ_identity=0) — a deliberate simplification to test whether the
  identity term is actually necessary for reasonable results at this scale, or whether skip
  connections in a U-Net generator provide enough of a "stay close to input" bias on their
  own.

## Why these choices

U-Net's skip connections are a natural point of comparison against ResNet's residual blocks
for preserving spatial detail — both are ways of giving the network a shortcut past the
bottleneck, but structured very differently. Dropping the identity loss and switching to
vanilla GAN loss were chosen specifically to create a genuine ablation-style contrast with
my teammate's more "by-the-book" CycleGAN configuration, rather than submitting a
near-identical model.

## Training

- 128×128 resolution, batch size 4, 40 epochs, Adam (β=0.5, 0.999), LR 2.5e-4 with linear
  decay to 0 over the second half of training.
- 400 Monet paintings / 400 photos (independent random sample, seed 7); the same held-out
  40/40 test sample structure as my teammate for evaluation.

## Results (see `metrics_report.csv` / `full_metrics_report.csv` for the full row)

| Metric | Monet→Photo | Photo→Monet |
|---|---|---|
| FID | 2.13 | 1.10 |
| KID (mean) | 3.42 | 0.26 |
| Cycle-reconstruction L1 | 0.113 (monet) | 0.104 (photo) |
| LPIPS | 0.474 | 0.489 |
| Content cosine similarity | 0.739 | 0.601 |

| Training metric | Value |
|---|---|
| Final loss G / D | 2.809 / 0.159 |
| Final cycle loss | 0.123 |
| Grad norm G (mean / max) | 136.7 / 253.5 |
| NaN count | 0 |
| Total parameters (2G + 2D) | 38,858,120 |
| Training time | 703 s (~11.7 min) |
| Train images/sec | 45.5 |
| Gen images/sec | 638.8 |
| Peak GPU memory | 845 MB |

As with my teammate's model, FID/KID here are computed on only 40 held-out images per
domain and should be read as noisy, directional estimates.

## Kaggle leaderboard

`evaluate_local.py` generates `images.zip` (200 photo→Monet translations, direct model
inference). Actual submission is **pending** — requires my own Kaggle account/API token.

## Failure analysis

See [`failure_analysis.md`](failure_analysis.md).

## How this compares to Ishan's model

My model reaches substantially better FID/KID in both directions (2.13/1.10 vs. his
6.29/4.18) and trains faster (703s vs. 1,012s, despite training on more images — 400 vs. 300
per domain), suggesting U-Net's skip connections do help at this small scale, and that
identity loss is not strictly necessary to get reasonable style transfer. However, my
generator's gradient norm grows essentially monotonically across training (11.7 at epoch 1
→ 253.5 at epoch 40, peaking even higher mid-training), unlike his controlled/decreasing
gradient norm — a real instability signature of combining BatchNorm (which behaves
differently under adversarial training than InstanceNorm) with vanilla GAN loss and no
gradient clipping. On a longer training run, my configuration would be the one at greater
risk of diverging; his is the more conservatively stable choice. This is a good example of a
real speed/quality-vs-stability trade-off between our two designs, worth walking through in
the demo/viva.
