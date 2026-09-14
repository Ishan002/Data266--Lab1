# DATA266 Lab 1 — Fall 2026 — Team 4

**Members:** Ishan Shah, Charvee Saraiya

**Repository:** https://github.com/Ishan002/Data266--Lab1

LLM Pretraining · Sentiment Classification · CycleGAN Style Transfer

This repository contains both members' independent implementations for all three lab tasks:

1. **Task 1** ([`task1_llm/`](task1_llm/)) — a GPT-style, character-level language model built
   from scratch (no prebuilt Transformer/attention modules), trained on TinyStories.
2. **Task 2** ([`task2_sentiment/`](task2_sentiment/)) — Yelp Polarity sentiment
   classification with 3 from-scratch models per member (no pretrained embeddings/LMs).
3. **Task 3** ([`task3_gan/`](task3_gan/)) — CycleGAN unpaired image-to-image translation
   between Monet paintings and photos, with a Kaggle leaderboard submission.

The combined final report is at
[`report/DATA266_Lab1_Report_Team_4.pdf`](report/).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Requires a CUDA-capable GPU for training in a reasonable time (all runs in this repo were
executed on an NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM); CPU also works but will be much slower.

## Shared datasets

Each task's shared raw data lives in `task*/data/` and is downloaded once via a script in
that folder (not committed to the repo — see `.gitignore`):

```bash
python task1_llm/data/download_tinystories.py      # TinyStories (roneneldan/TinyStories via HF)
python task2_sentiment/data/download_yelp_polarity.py   # Yelp Polarity (fancyzhx/yelp_polarity via HF)
```

Task 3's Monet/photo images are downloaded from the original CycleGAN paper's public dataset
mirror (Berkeley) — see [`task3_gan/data/README.md`](task3_gan/data/README.md) for the exact
command, since it is a direct `curl`/`unzip` rather than a Python script.

Each member then draws their **own** independent subsample/split from this shared raw data
inside their own notebook (see each member's `src/` and `results.md`) — preprocessing output
itself is never shared, per the lab's submission rules.

## Reproducing a smoke test (one command, any member's run)

Every training script in Ishan Shah's folders supports a `SMOKE_TEST=1` environment variable
that shrinks epochs/dataset size to run a full correctness check (real training loop, real
metrics, real checkpoint) in well under a minute, without needing the full multi-minute run:

```bash
SMOKE_TEST=1 python task1_llm/ishan_shah/src/train_gpt.py
SMOKE_TEST=1 python task2_sentiment/ishan_shah/src/train_sentiment.py
SMOKE_TEST=1 python task3_gan/ishan_shah/src/train_cyclegan.py
```

(On Windows PowerShell: `$env:SMOKE_TEST=1; python task1_llm/ishan_shah/src/train_gpt.py`.)

To reproduce a member's **full** run exactly as reported, run the same script without
`SMOKE_TEST` set — all hyperparameters are hard-coded at the top of each script (no
config file indirection), and all scripts are seeded for repeatability of the *procedure*
(exact bit-for-bit reproduction of GPU-nondeterministic ops like cuDNN convolutions is not
guaranteed, per PyTorch's own reproducibility caveats).

Each script's corresponding executed notebook (`task*/*/src/*.ipynb`) shows the real output
of the exact run that produced the numbers in that member's `metrics_report.csv`.

## Repository layout

```
team-repo/
├── README.md                     <- this file
├── requirements.txt
├── task1_llm/
│   ├── data/                     <- shared TinyStories corpus (download script + raw text)
│   ├── ishan_shah/                <- Pre-LN GPT, 4L/4H/128d
│   └── charvee_saraiya/                   <- Post-LN GPT, 6L/4H/96d
├── task2_sentiment/
│   ├── data/                     <- shared Yelp Polarity parquet (download script)
│   ├── ishan_shah/                <- NBOW / TextCNN / BiLSTM
│   └── charvee_saraiya/                   <- MaxPool-MLP / GRU / Dilated-CNN
├── task3_gan/
│   ├── data/                     <- shared Monet/Photo images (download script)
│   ├── ishan_shah/                <- ResNet generator, LSGAN, InstanceNorm
│   └── charvee_saraiya/                   <- U-Net generator, vanilla GAN, BatchNorm
├── common/                       <- shared *measurement-only* evaluation utilities
│   ├── eval_classifier.py        <- Task 2 metrics (both members call this identically)
│   ├── eval_gan.py               <- Task 3 metrics (FID/KID/LPIPS/content-similarity)
│   └── py_to_notebook.py         <- converts each member's .py training script to an
│                                     executed .ipynb with real captured outputs
├── reproducibility/
│   ├── manifests/                <- environment.txt, pip freeze, per-run checkpoint map
│   └── raw_logs/                 <- copies of every member's raw_train_log.txt
└── report/
    └── DATA266_Lab1_Report_Team_4.pdf
```

Each member's folder under every task follows:

```
member_name/
├── src/            <- train_*.py (source of truth) + the executed .ipynb generated from it
├── checkpoints/    <- trained model weights (.pt)
├── outputs/        <- plots, generated samples/images, history.json
├── metrics_report.csv
├── failure_analysis.md
└── results.md
```

## Why shared `common/` code doesn't violate "own model" rules

`common/eval_classifier.py` and `common/eval_gan.py` contain **only measurement code**
(accuracy, FID, bootstrap CIs, etc.) — no model architecture, no training loop, no
preprocessing. Both members call the same measurement functions so that reported numbers are
directly, fairly comparable in the team report; every model architecture, every
hyperparameter choice, every preprocessing decision, and every training loop is independently
authored per member, per the lab's Section 2 requirements. `common/py_to_notebook.py` is
tooling (a script-to-notebook converter), not model code.

## Known gaps / what still needs a human

- **Task 3 Kaggle submission:** each member's `task3_gan/<member>/src/evaluate_local.py`
  generates the exact `images.zip` to submit, but actually running
  `kaggle competitions submit` requires each member's own Kaggle account/API token, which
  this repo does not have. `submission.csv` is a placeholder until that's done.
- **Task 3 blinded human audit:** `task3_gan/human_audit/` contains the fixed 30-sample audit
  sheet: it must be independently filled in by both team members before inter-rater agreement
  (Cohen's kappa) can be computed and reported.
- **Demo/Viva:** each member's `results.md` and `failure_analysis.md` are written to be
  studied and defended by that member individually — see the lab's Section 4 requirements.
