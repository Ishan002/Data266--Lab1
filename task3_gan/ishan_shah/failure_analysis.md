# Task 3 — Evaluation & Failure Analysis (Ishan Shah)

## Visual quality assessment

Looking at `outputs/pred_A2B/sample_grid.png` (Monet→Photo) and `outputs/pred_B2A/sample_grid.png`
(Photo→Monet):

- **Checkerboard/grid artifacts.** Nearly every generated image shows a faint but
  consistent regular grid pattern superimposed on the image, most visible in flat sky/water
  regions (e.g. the leftmost Monet→Photo sample, and several Photo→Monet skies). This is the
  textbook artifact of `ConvTranspose2d`-based upsampling — kernel-size/stride mismatches in
  the two upsampling layers cause uneven overlap between adjacent output pixels. It is a
  known, well-documented CycleGAN/DCGAN failure mode, not a bug in the loss computation.
- **Content preservation is inconsistent across samples.** Some translations preserve scene
  structure well (the single-tree image and the mountain/lake image in Photo→Monet keep
  their silhouette and composition clearly recognizable); others (the first and second
  Photo→Monet samples) show the underlying content becoming muddled — colors and shapes
  blend into an abstract wash rather than a recognizable translated scene. This matches the
  content-preservation cosine similarity numbers (0.788 monet→photo vs. only 0.640
  photo→monet — the photo→monet direction preserves content less reliably).
- **Monet→Photo outputs still look painterly, not photographic.** Several outputs (e.g. the
  6th and 7th samples) retain visible brushstroke-like texture rather than photographic
  detail, meaning the generator has not fully learned to strip Monet's stylistic signature
  when producing "photos."

## Cycle-consistency verification

Cycle-reconstruction L1 is 0.176 (Monet direction) and 0.163 (photo direction) on the
held-out test set — i.e., averaged per-pixel absolute difference (in [-1,1] normalized
space) between an original image and its reconstruction after a full A→B→A round-trip. Both
values are low relative to the [-2, 2] possible range and steadily decreased across training
(cycle loss 0.323 at epoch 1 → 0.117 at epoch 40, see `outputs/loss_curves.png`), confirming
the cycle-consistency constraint is correctly implemented and is doing real work: the model
is not just learning an arbitrary domain-to-domain mapping but one that is (approximately)
invertible, as CycleGAN requires in the absence of paired supervision.

## Training stability analysis

- Generator and discriminator losses both decrease smoothly and monotonically-ish across all
  40 epochs (loss_G: 5.16→2.31; loss_D: 0.33→0.12) with no spikes, divergence, or NaNs
  (`nan_count: 0`).
- Gradient norm for the combined generators peaks at 30.1 and *decreases* over the second
  half of training (correlating with the linear LR decay from epoch 20), suggesting the LSGAN
  loss + InstanceNorm + identity-loss combination trains in a well-behaved regime at this
  scale — unlike Charvee Saraiya's model, whose gradient norm grows through the entire run (see her
  `failure_analysis.md`).
- Identity loss (0.307→0.123) and cycle loss decrease together, consistent with both
  regularizers pulling the model in compatible directions rather than fighting each other.

## Shortcomings of this analysis

- FID/KID are computed on only 40 held-out images per domain — far below the hundreds-to-
  thousands typically used for stable FID estimates, so the reported 6.29 / 4.18 FID values
  should be read as rough, high-variance signals, not precise scores.
  the LPIPS metric here is computed between an image and its own cross-domain translation
  (a proxy for "how much did the pixels change"), not between generated and real
  distributions as LPIPS is more commonly used — a limitation of the available fixed test set
  size, noted here for transparency rather than silently presented as a standard usage.
- Only one random seed / one training run was evaluated; CycleGAN training is known to be
  seed-sensitive, so a single run's FID/KID should not be over-interpreted as *the*
  performance of this architecture.
