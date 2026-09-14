# Checkpoint → Result Mapping

This file traces every reported number back to the exact checkpoint and log that produced it,
per the lab's reproducibility requirement.

## Task 1 — GPT from scratch

| Member | Checkpoint | Metrics | Raw log |
|---|---|---|---|
| Ishan Shah | `task1_llm/ishan_shah/checkpoints/gpt_ishan.pt` | `task1_llm/ishan_shah/metrics_report.csv` | `task1_llm/ishan_shah/raw_train_log.txt` |
| Charvee Saraiya | `task1_llm/charvee_saraiya/checkpoints/gpt_charvee_saraiya.pt` | `task1_llm/charvee_saraiya/metrics_report.csv` | `task1_llm/charvee_saraiya/raw_train_log.txt` |

## Task 2 — Yelp Polarity sentiment classification

| Member | Model | Checkpoint | Metrics row | Raw log |
|---|---|---|---|---|
| Ishan Shah | baseline_nbow | `task2_sentiment/ishan_shah/checkpoints/baseline_nbow.pt` | `task2_sentiment/ishan_shah/metrics_report.csv` (row 1) | `task2_sentiment/ishan_shah/raw_train_log.txt`* |
| Ishan Shah | experimental_textcnn | `task2_sentiment/ishan_shah/checkpoints/experimental_textcnn.pt` | (row 2) | — |
| Ishan Shah | experimental_bilstm | `task2_sentiment/ishan_shah/checkpoints/experimental_bilstm.pt` | (row 3) | — |
| Charvee Saraiya | baseline_maxpool_mlp | `task2_sentiment/charvee_saraiya/checkpoints/baseline_maxpool_mlp.pt` | `task2_sentiment/charvee_saraiya/metrics_report.csv` (row 1) | `task2_sentiment/charvee_saraiya/raw_train_log.txt`* |
| Charvee Saraiya | experimental_gru | `task2_sentiment/charvee_saraiya/checkpoints/experimental_gru.pt` | (row 2) | — |
| Charvee Saraiya | experimental_dilated_cnn | `task2_sentiment/charvee_saraiya/checkpoints/experimental_dilated_cnn.pt` | (row 3) | — |

\* Task 2 does not have a separate per-epoch raw log file (unlike Tasks 1 & 3) — the console
output of all three models' training loops is captured directly in each member's executed
notebook (`src/task2_sentiment_*.ipynb`), which is the evidence trail for this task.

## Task 3 — CycleGAN

| Member | Checkpoints | Metrics | Raw log |
|---|---|---|---|
| Ishan Shah | `task3_gan/ishan_shah/checkpoints/{G_AB,G_BA,D_A,D_B}.pt` | `task3_gan/ishan_shah/metrics_report.csv` | `task3_gan/ishan_shah/raw_train_log.txt` |
| Charvee Saraiya | `task3_gan/charvee_saraiya/checkpoints/{G_AB,G_BA,D_A,D_B}.pt` | `task3_gan/charvee_saraiya/metrics_report.csv` | `task3_gan/charvee_saraiya/raw_train_log.txt` |

Environment used for every run: see `environment.txt` and `pip_freeze.txt` in this folder
(Python 3.12.10, PyTorch 2.2.1+cu121, NVIDIA GeForce RTX 4050 Laptop GPU, CUDA 12.1).
