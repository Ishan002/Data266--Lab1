"""
Builds the combined team PDF report: DATA266_Lab1_Report_Team_[Team Number].pdf

Reads every member's metrics_report.csv (Tasks 1-3) plus the loss-curve / sample-generation
PNGs already produced by each training script, and assembles one PDF with a team ownership
statement, per-task comparison tables (architecture + hyperparameters + every required
metric, side by side), jointly-written analysis, and references.

Run from the repo root:
    python report/build_report.py
"""
import ast
import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, ListFlowable, ListItem
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAM_NUMBER = "[Team Number]"  # fill in once assigned on Canvas
GITHUB_URL = "https://github.com/Ishan002/Data266--Lab1"

OUT_PATH = os.path.join(ROOT, "report", f"DATA266_Lab1_Report_Team_{TEAM_NUMBER}.pdf")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleBig", fontSize=22, leading=26, spaceAfter=18, alignment=1))
styles.add(ParagraphStyle(name="H1", fontSize=16, leading=20, spaceBefore=18, spaceAfter=8, textColor=colors.HexColor("#1a3d5c")))
styles.add(ParagraphStyle(name="H2", fontSize=13, leading=16, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#2a5a82")))
styles.add(ParagraphStyle(name="Body", fontSize=9.5, leading=13, spaceAfter=6))
styles.add(ParagraphStyle(name="Small", fontSize=7.5, leading=9.5))
styles.add(ParagraphStyle(name="Caption", fontSize=8, leading=10, textColor=colors.grey, spaceAfter=10, alignment=1))

story = []


def h1(text):
    story.append(Paragraph(text, styles["H1"]))


def h2(text):
    story.append(Paragraph(text, styles["H2"]))


def body(text):
    story.append(Paragraph(text, styles["Body"]))


cell_style = ParagraphStyle(name="Cell", fontSize=6.8, leading=8.2)
header_style = ParagraphStyle(name="CellHeader", fontSize=6.8, leading=8.2, textColor=colors.white)


def small_table(data, col_widths=None):
    wrapped = []
    for r_idx, row in enumerate(data):
        style = header_style if r_idx == 0 else cell_style
        wrapped.append([Paragraph(str(cell), style) for cell in row])
    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d5c")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f7")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def fmt_num(v):
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        return f"{v:.4f}"
    return str(v)


def add_image(path, width=6.5 * inch, caption=None):
    if not os.path.exists(path):
        body(f"<i>[missing image: {path}]</i>")
        return
    img = Image(path)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    story.append(img)
    if caption:
        story.append(Paragraph(caption, styles["Caption"]))
    story.append(Spacer(1, 6))


# ============================================================================================
# Title page
# ============================================================================================
story.append(Spacer(1, 2 * inch))
story.append(Paragraph("DATA266 Lab 1 — Fall 2026", styles["TitleBig"]))
story.append(Paragraph("LLM Pretraining · Sentiment Classification · CycleGAN Style Transfer", styles["Caption"]))
story.append(Spacer(1, 0.4 * inch))
story.append(Paragraph(f"Team {TEAM_NUMBER}", styles["H1"]))
story.append(Paragraph("Members: Ishan Shah, Charvee Saraiya", styles["Body"]))
story.append(Paragraph(f"GitHub repository: {GITHUB_URL}", styles["Body"]))
story.append(PageBreak())

# ============================================================================================
# Team ownership statement
# ============================================================================================
h1("Team Ownership Statement")
body(
    "Both team members independently designed, implemented, and trained their own models for "
    "all three tasks. <b>Ishan Shah</b> built a Pre-LayerNorm character-level GPT (Task 1), "
    "three from-scratch Yelp Polarity classifiers — a mean-pooled neural bag-of-words baseline, "
    "a multi-kernel TextCNN, and a bidirectional LSTM (Task 2) — and a ResNet-generator, "
    "LSGAN-loss CycleGAN with identity loss (Task 3). <b>Charvee Saraiya</b> built an independently "
    "architected Post-LayerNorm character-level GPT (Task 1), three differently-designed Yelp "
    "Polarity classifiers — a max-pooled MLP baseline, a unidirectional GRU, and a dilated 1D "
    "CNN (Task 2) — and a U-Net-generator, vanilla-GAN-loss CycleGAN without identity loss "
    "(Task 3). Every dataset split, preprocessing pipeline, and hyperparameter choice was made "
    "independently by each member on their own subsample of the shared raw data; only "
    "measurement code (evaluation metrics, not model architecture) is shared, so that both "
    "members' numbers are computed identically and can be compared fairly in the tables below. "
    "This report was jointly assembled by both members from each other's individual "
    "<code>results.md</code> and <code>metrics_report.csv</code> files."
)

# ============================================================================================
# Task 1
# ============================================================================================
story.append(PageBreak())
h1("Task 1 — GPT-Style LLM From Scratch (TinyStories, character-level)")

t1_ishan = pd.read_csv(os.path.join(ROOT, "task1_llm", "ishan_shah", "metrics_report.csv")).iloc[0]
t1_charvee_saraiya = pd.read_csv(os.path.join(ROOT, "task1_llm", "charvee_saraiya", "metrics_report.csv")).iloc[0]

h2("Architecture & hyperparameters")
arch_rows = [["", "Ishan Shah", "Charvee Saraiya"]]
for label, key in [
    ("Block style", "architecture"), ("Layers", "n_layer"), ("Heads", "n_head"),
    ("Embedding dim", "n_embd"), ("Block size", "block_size"), ("Dropout", "dropout"),
    ("Epochs", "epochs"), ("Batch size", "batch_size"), ("Peak LR", "peak_lr"),
]:
    arch_rows.append([label, str(t1_ishan[key]), str(t1_charvee_saraiya[key])])
story.append(small_table(arch_rows, col_widths=[1.6 * inch, 2.7 * inch, 2.7 * inch]))
story.append(Spacer(1, 10))

h2("All Task 1 metrics, side by side")
metric_keys_1 = [
    ("Train loss", "train_loss"), ("Val loss", "val_loss"), ("Perplexity", "perplexity"),
    ("Bits/char", "bits_per_char"), ("Generalization gap", "generalization_gap"),
    ("Top-1 next-char acc", "top1_next_char_acc"), ("Distinct-1", "distinct_1"),
    ("Distinct-2", "distinct_2"), ("Distinct-3", "distinct_3"),
    ("Repeated 4-gram rate", "repeated_4gram_rate"), ("Grad norm (mean)", "grad_norm_mean"),
    ("Grad norm (max)", "grad_norm_max"), ("NaN count", "nan_count"),
    ("Parameters", "param_count"), ("Train tokens/sec", "train_tokens_per_sec"),
    ("Gen tokens/sec", "gen_tokens_per_sec"), ("Peak memory (MB)", "peak_memory_MB"),
    ("Total train time (s)", "total_training_time_sec"),
]
rows = [["Metric", "Ishan Shah", "Charvee Saraiya"]]
for label, key in metric_keys_1:
    v1, v2 = t1_ishan[key], t1_charvee_saraiya[key]
    rows.append([label, fmt_num(v1), fmt_num(v2)])
story.append(small_table(rows, col_widths=[2.0 * inch, 2.5 * inch, 2.5 * inch]))

story.append(PageBreak())
h2("Loss curves")
add_image(os.path.join(ROOT, "task1_llm", "ishan_shah", "outputs", "loss_curve.png"), width=5.5*inch,
          caption="Ishan Shah — Pre-LN GPT train/val loss.")
add_image(os.path.join(ROOT, "task1_llm", "charvee_saraiya", "outputs", "loss_curve.png"), width=5.5*inch,
          caption="Charvee Saraiya — Post-LN GPT train/val loss.")

h2("Joint analysis")
body(
    "<b>Strengths:</b> both from-scratch character-level GPTs train stably (zero NaNs, "
    "bounded gradient norms) and converge to sensible perplexities (7.3 and 5.7) given the "
    "very small 100K-character training budget. Charvee Saraiya's deeper (6-layer) Post-LN model "
    "reaches a lower validation loss and higher next-character accuracy than Ishan's "
    "shallower (4-layer) Pre-LN model, suggesting depth mattered more than width at this scale."
)
body(
    "<b>Weaknesses/limitations:</b> Charvee Saraiya's model shows a larger generalization gap "
    "(0.106 vs 0.049) and a much higher peak gradient norm (14.1 vs 4.4), the expected "
    "instability cost of Post-LN training at depth without Pre-LN's gradient renormalization. "
    "Both models' generations are qualitatively similar failure-wise — greedy-decoding "
    "repetition loops, broken word formation under sampling, and short-range-only coherence — "
    "indicating these failures are governed more by the 100K-character training budget than by "
    "the specific architectural choices compared here (see both members' failure_analysis.md)."
)
body(
    "<b>What we'd try next:</b> train on a larger character budget (the full TinyStories "
    "corpus rather than a 100K-character slice) to test whether coherence failures are purely "
    "data-limited; try a hybrid Pre-LN-with-weight-tying-and-depth configuration to combine "
    "both members' apparent advantages."
)

# ============================================================================================
# Task 2
# ============================================================================================
story.append(PageBreak())
h1("Task 2 — Yelp Polarity Sentiment Classification")

t2_ishan = pd.read_csv(os.path.join(ROOT, "task2_sentiment", "ishan_shah", "metrics_report.csv"))
t2_charvee_saraiya = pd.read_csv(os.path.join(ROOT, "task2_sentiment", "charvee_saraiya", "metrics_report.csv"))
t2_all = pd.concat([t2_ishan.assign(member="Ishan Shah"), t2_charvee_saraiya.assign(member="Charvee Saraiya")], ignore_index=True)

h2("All 6 models: architecture & core metrics")
core_cols = [
    ("Member", "member"), ("Model", "model"), ("Accuracy", "accuracy"), ("Macro-F1", "f1_macro"),
    ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("MCC", "mcc"), ("Brier", "brier_score"),
    ("ECE", "expected_calibration_error"), ("Params", "param_count"),
    ("Train time (s)", "training_time_sec"), ("Examples/sec", "examples_per_sec"),
    ("Peak mem (MB)", "peak_memory_MB"),
]
rows = [[c[0] for c in core_cols]]
for _, r in t2_all.iterrows():
    row = []
    for label, key in core_cols:
        v = r[key]
        row.append(fmt_num(v))
    rows.append(row)
story.append(small_table(rows, col_widths=[0.55*inch, 0.85*inch] + [0.5*inch]*11))

story.append(Spacer(1, 10))
h2("Precision / recall (macro, micro, weighted) & confusion matrices")
pr_cols = [
    ("Member", "member"), ("Model", "model"),
    ("P-macro", "precision_macro"), ("R-macro", "recall_macro"),
    ("P-micro", "precision_micro"), ("R-micro", "recall_micro"), ("F1-micro", "f1_micro"),
    ("P-wtd", "precision_weighted"), ("R-wtd", "recall_weighted"), ("F1-wtd", "f1_weighted"),
]
rows = [[c[0] for c in pr_cols]]
for _, r in t2_all.iterrows():
    row = []
    for label, key in pr_cols:
        v = r[key]
        row.append(fmt_num(v))
    rows.append(row)
story.append(small_table(rows, col_widths=[0.65*inch, 1.0*inch] + [0.62*inch]*8))

story.append(PageBreak())
h2("95% bootstrap CIs, per-slice robustness, and calibration")
for _, r in t2_all.iterrows():
    acc_ci = ast.literal_eval(r["accuracy_ci95"])
    f1_ci = ast.literal_eval(r["macro_f1_ci95"])
    mcc_ci = ast.literal_eval(r["mcc_ci95"])
    per_slice = ast.literal_eval(r["per_slice"])
    body(f"<b>{r['member']} — {r['model']}</b>: accuracy 95% CI [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}] | "
         f"macro-F1 CI [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}] | MCC CI [{mcc_ci[0]:.3f}, {mcc_ci[1]:.3f}]")
    slice_txt = " | ".join(f"{s}: F1={v['macro_f1']:.3f}, err={v['error_rate']:.1%} (n={v['n']})" for s, v in per_slice.items())
    body(f"Per-slice: {slice_txt}")

story.append(Spacer(1, 8))
h2("Paired McNemar test (baseline vs. each experimental model, own test set)")
body(
    "<b>Ishan Shah:</b> baseline_nbow vs. experimental_textcnn p=0.196 (not significant); "
    "baseline_nbow vs. experimental_bilstm p=0.260 (not significant) — added architecture did "
    "not beat the simple mean-pooled baseline on his 3,000-review test set."
)
body(
    "<b>Charvee Saraiya:</b> baseline_maxpool_mlp vs. experimental_gru p=1.5e-14 (highly significant); "
    "baseline_maxpool_mlp vs. experimental_dilated_cnn p=5.6e-6 (highly significant) — both "
    "experimental models are genuinely, not just numerically, better than her baseline."
)

story.append(PageBreak())
h2("Confusion matrices and loss curves")
add_image(os.path.join(ROOT, "task2_sentiment", "ishan_shah", "outputs", "confusion_matrices.png"), width=6.3*inch,
          caption="Ishan Shah — confusion matrices, all 3 models.")
add_image(os.path.join(ROOT, "task2_sentiment", "charvee_saraiya", "outputs", "confusion_matrices.png"), width=6.3*inch,
          caption="Charvee Saraiya — confusion matrices, all 3 models.")

h2("Joint analysis")
body(
    "<b>Strengths:</b> all six from-scratch models land between 80.9% and 88.9% test accuracy "
    "with strong discrimination (ROC-AUC 0.90-0.96), showing that a from-scratch learned "
    "embedding is sufficient for strong Yelp Polarity performance at this data scale, without "
    "any pretrained embeddings or language models."
)
body(
    "<b>Weaknesses/limitations:</b> the two members' baseline-vs-experimental patterns "
    "diverge sharply — Ishan's added architecture (TextCNN/BiLSTM) does not significantly "
    "beat his NBOW baseline (McNemar p>0.05 both), while Charvee Saraiya's does (p<1e-5 both). Both "
    "members independently found the 'long review' length slice to be the hardest for almost "
    "every model, and calibration (ECE) tends to get worse as accuracy improves for Charvee Saraiya's "
    "models — a real trade-off worth flagging."
)
body(
    "<b>What we'd try next:</b> apply temperature-scaling calibration post-hoc on the "
    "best-performing model from each member; test whether Ishan's stemmed vs. Charvee Saraiya's "
    "unstemmed preprocessing is the reason her baseline starts weaker relative to her "
    "experimental models, by swapping preprocessing pipelines across each other's "
    "architectures as a follow-up ablation."
)

# ============================================================================================
# Task 3
# ============================================================================================
story.append(PageBreak())
h1("Task 3 — CycleGAN Image Style Transfer (Monet ↔ Photo)")

t3_ishan = pd.read_csv(os.path.join(ROOT, "task3_gan", "ishan_shah", "metrics_report.csv")).iloc[0]
t3_charvee_saraiya_path = os.path.join(ROOT, "task3_gan", "charvee_saraiya", "metrics_report.csv")
t3_charvee_saraiya = pd.read_csv(t3_charvee_saraiya_path).iloc[0] if os.path.exists(t3_charvee_saraiya_path) else None

h2("Architecture & hyperparameters")
arch_rows = [["", "Ishan Shah", "Charvee Saraiya"]]
for label, key in [
    ("Architecture", "architecture"), ("Image size", "img_size"), ("Epochs", "epochs"),
    ("Batch size", "batch_size"), ("Lambda cycle", "lambda_cycle"), ("Lambda identity", "lambda_identity"),
    ("Train Monet imgs", "n_train_monet"), ("Train photo imgs", "n_train_photo"),
]:
    v2 = str(t3_charvee_saraiya[key]) if t3_charvee_saraiya is not None else "pending"
    arch_rows.append([label, str(t3_ishan[key]), v2])
story.append(small_table(arch_rows, col_widths=[1.5 * inch, 3.0 * inch, 3.0 * inch]))
story.append(Spacer(1, 10))

h2("All Task 3 metrics, side by side")
metric_keys_3 = [
    ("FID monet->photo", "fid_monet2photo"), ("KID monet->photo (mean)", "kid_monet2photo_mean"),
    ("FID photo->monet", "fid_photo2monet"), ("KID photo->monet (mean)", "kid_photo2monet_mean"),
    ("Cycle L1 (monet)", "cycle_l1_monet"), ("Cycle L1 (photo)", "cycle_l1_photo"),
    ("LPIPS monet->photo", "lpips_monet2photo"), ("LPIPS photo->monet", "lpips_photo2monet"),
    ("Content cos-sim monet->photo", "content_cosine_sim_monet2photo"),
    ("Content cos-sim photo->monet", "content_cosine_sim_photo2monet"),
    ("Final loss G", "final_loss_G"), ("Final loss D", "final_loss_D"),
    ("Final cycle loss", "final_loss_cycle"), ("Grad norm G (mean)", "grad_norm_G_mean"),
    ("Grad norm G (max)", "grad_norm_G_max"), ("NaN count", "nan_count"),
    ("Total params", "param_count_total"), ("Train time (s)", "training_time_sec"),
    ("Train images/sec", "train_images_per_sec"), ("Gen images/sec", "gen_images_per_sec"),
    ("Peak memory (MB)", "peak_memory_MB"),
]
rows = [["Metric", "Ishan Shah", "Charvee Saraiya"]]
for label, key in metric_keys_3:
    v1 = t3_ishan[key] if key in t3_ishan else "n/a"
    v2 = (t3_charvee_saraiya[key] if (t3_charvee_saraiya is not None and key in t3_charvee_saraiya) else "pending")
    rows.append([label, fmt_num(v1), fmt_num(v2)])
story.append(small_table(rows, col_widths=[2.1 * inch, 2.3 * inch, 2.3 * inch]))

story.append(Spacer(1, 8))
h2("Kaggle leaderboard")
body(
    "Both members' <code>evaluate_local.py</code> scripts generate the exact "
    "<code>images.zip</code> submission from their own trained generator's inference on real "
    "photos (no manual editing, no external images, no pretrained generation models — see the "
    "leaderboard-integrity note in the lab spec). Actually submitting via "
    "<code>kaggle competitions submit</code> requires each member's own Kaggle account and is "
    "<b>pending</b> — see <code>task3_gan/&lt;member&gt;/submission.csv</code> for status."
)

h2("Blinded human audit (30 fixed samples, 2 raters)")
body(
    "The blinded audit sheet (15 samples from each member's photo→Monet outputs, "
    "filenames anonymized) is built at <code>task3_gan/human_audit/</code>. Both members' "
    "independent ratings and the resulting Cohen's kappa / percent agreement are "
    "<b>pending</b> completion by both raters — see "
    "<code>task3_gan/human_audit/README.md</code> for the exact steps."
)

story.append(PageBreak())
h2("Loss curves and sample translations")
add_image(os.path.join(ROOT, "task3_gan", "ishan_shah", "outputs", "loss_curves.png"), width=6.3*inch,
          caption="Ishan Shah — ResNet/LSGAN CycleGAN loss curves.")
add_image(os.path.join(ROOT, "task3_gan", "ishan_shah", "outputs", "pred_B2A", "sample_grid.png"), width=6.3*inch,
          caption="Ishan Shah — sample photo→Monet translations (test set).")
if t3_charvee_saraiya is not None:
    add_image(os.path.join(ROOT, "task3_gan", "charvee_saraiya", "outputs", "loss_curves.png"), width=6.3*inch,
              caption="Charvee Saraiya — U-Net/vanilla-GAN CycleGAN loss curves.")
    add_image(os.path.join(ROOT, "task3_gan", "charvee_saraiya", "outputs", "pred_B2A", "sample_grid.png"), width=6.3*inch,
              caption="Charvee Saraiya — sample photo→Monet translations (test set).")

h2("Joint analysis")
body(
    "<b>Strengths:</b> both CycleGANs train without NaNs and show generator/discriminator "
    "losses converging to a stable adversarial equilibrium rather than diverging, and cycle-"
    "consistency loss decreases steadily across training for both members, confirming both "
    "cycle-consistency constraints are correctly implemented and functioning."
)
body(
    "<b>Weaknesses/limitations:</b> training was intentionally scaled down (128x128 "
    "resolution, a few hundred images per domain, 40 epochs) to fit within a single GPU "
    "session's time budget, so FID/KID scores are far from publication-quality CycleGAN "
    "results and visible artifacts remain in the sample grids above."
)
body(
    "<b>What we'd try next:</b> scale up to the full ~1,000 Monet / ~6,000 photo training "
    "pools at 256x256 resolution with more epochs on GPU Lab hardware, and complete the "
    "pending Kaggle submission and blinded human audit to get the two remaining required "
    "metrics."
)

# ============================================================================================
# References
# ============================================================================================
story.append(PageBreak())
h1("References")
refs = [
    "Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS 2017.",
    "Eldan, R. & Li, Y. (2023). TinyStories: How Small Can Language Models Be and Still Speak "
    "Coherent English? arXiv:2305.07759.",
    "Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired Image-to-Image "
    "Translation using Cycle-Consistent Adversarial Networks. ICCV 2017.",
    "Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text "
    "Classification (source of the Yelp Polarity dataset). NeurIPS 2015.",
]
story.append(ListFlowable([ListItem(Paragraph(r, styles["Body"])) for r in refs], bulletType="1"))

doc = SimpleDocTemplate(OUT_PATH, pagesize=letter, topMargin=0.6*inch, bottomMargin=0.6*inch,
                         leftMargin=0.55*inch, rightMargin=0.55*inch)
doc.build(story)
print(f"Wrote {OUT_PATH}")
