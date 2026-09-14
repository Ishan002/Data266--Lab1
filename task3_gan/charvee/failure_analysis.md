# Task 3 — Evaluation & Failure Analysis (Charvee)

## Visual quality assessment

Looking at `outputs/pred_A2B/sample_grid.png` (Monet→Photo) and `outputs/pred_B2A/sample_grid.png`
(Photo→Monet):

- **Bright circular "blob" artifacts.** Several outputs — most strikingly the 5th
  Photo→Monet sample, which shows four distinct bright white circular blobs arranged in a
  pattern, and the 1st Photo→Monet sample, which has one large bright blob near center —
  show a defect not present in my teammate's ResNet-based outputs. This looks characteristic
  of the U-Net bottleneck/skip-connection design combined with BatchNorm: at batch size 4,
  BatchNorm's running statistics are estimated from very few samples, and a channel that
  saturates for one image in a batch can produce a spatially localized, disproportionately
  bright region. This is a plausible, architecture-specific failure mode worth testing by
  switching to InstanceNorm or GroupNorm in the U-Net as a follow-up.
- **A faint grid/checkerboard texture** is also visible in my outputs (e.g. background
  regions of the 1st and 8th Monet→Photo samples), the same transposed-convolution
  upsampling artifact seen in my teammate's ResNet generator — expected, since both
  generators use `ConvTranspose2d` for upsampling.
- **Color-cast failures.** The last Photo→Monet sample renders almost entirely in a
  saturated green cast, losing most of the original photo's color variation — a case where
  the generator appears to have collapsed toward a specific stylization rather than adapting
  to the input's actual content.
- **Where it works well:** several Monet→Photo samples (the 1st, showing a canal/bridge, and
  the 7th, showing a building's reflection in water) preserve recognizable structure and
  achieve noticeably more photographic texture/detail than my teammate's equivalent samples,
  consistent with my better FID scores.

## Cycle-consistency verification

Cycle-reconstruction L1 is 0.113 (Monet direction) and 0.104 (photo direction) — both lower
(better) than my teammate's 0.176/0.163 — and cycle loss decreased steadily across training
(0.236 at epoch 1 → 0.123 at epoch 40, `outputs/loss_curves.png`), confirming the
cycle-consistency term is implemented correctly and is being effectively optimized, even
without an identity loss term to additionally constrain the mapping.

## Training stability analysis

- Cycle loss decreases smoothly with no spikes and zero NaNs across all 40 epochs.
- **The generator's gradient norm is not well-behaved:** it starts at 11.7 (epoch 1) and
  grows to 253.5 (epoch 40), peaking briefly even higher mid-training (epoch 35: 294.6) —
  an upward trend, not the stable-or-decreasing pattern my teammate's LSGAN+InstanceNorm
  model shows. This did not cause NaNs or visible divergence within 40 epochs, but the trend
  is concerning: it suggests this configuration (vanilla GAN loss + BatchNorm + no gradient
  clipping + no identity loss) is accumulating instability that a longer training run could
  plausibly turn into a collapse.
- The discriminator loss fluctuates more than my teammate's (e.g. jumping from 0.15 to 0.26
  between epochs 10-11) rather than monotonically decreasing, consistent with vanilla GAN's
  known tendency toward less stable discriminator dynamics than LSGAN.

## Shortcomings of this analysis

- Same small-sample caveat as my teammate's: FID/KID computed on only 40 held-out images per
  domain, so absolute values are noisy.
- I did not apply gradient clipping in this run; given the growing gradient norm trend, a
  natural next experiment would be adding `clip_grad_norm_` and re-running to see whether it
  controls the instability without hurting the (currently strong) FID scores.
- Only one training run/seed was evaluated; given the visible late-training instability,
  results could vary more between seeds for this configuration than for my teammate's more
  stable one.
