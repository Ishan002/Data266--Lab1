"""
Shared evaluation utilities for Task 2 (Yelp Polarity sentiment classification).

This module only *measures* trained models -- it contains no model architecture code, so
both members use it to keep their reported numbers directly comparable, while each member's
own model architecture, preprocessing, and training loop remain entirely their own.

Implements the full Task 2 metrics list from the lab spec:
  accuracy; precision/recall/F1 (macro, micro, weighted); confusion matrix; ROC-AUC; PR-AUC;
  MCC; Brier score; expected calibration error; 95% bootstrap CI (accuracy, macro-F1, MCC);
  paired McNemar test; per-slice macro-F1 and error rate; param count; training time;
  examples/sec; peak memory.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    brier_score_loss,
)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = y_true[mask].mean()
        conf_bin = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def bootstrap_ci(y_true, y_pred, y_prob, metric_fn, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    stats = np.sort(stats)
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[int((1 - alpha / 2) * n_boot) - 1]
    return float(lo), float(hi)


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """Paired McNemar test comparing two models' correctness on the same test set."""
    y_true = np.asarray(y_true)
    correct_a = (np.asarray(y_pred_a) == y_true)
    correct_b = (np.asarray(y_pred_b) == y_true)
    n01 = int(np.sum(correct_a & ~correct_b))  # a right, b wrong
    n10 = int(np.sum(~correct_a & correct_b))  # a wrong, b right
    from scipy.stats import chi2
    if n01 + n10 == 0:
        stat, p = 0.0, 1.0
    else:
        stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)  # continuity-corrected
        p = float(1 - chi2.cdf(stat, df=1))
    return {"n01_a_only_correct": n01, "n10_b_only_correct": n10, "statistic": float(stat), "p_value": p}


def length_slice(lengths, q1=0.33, q2=0.66):
    """Bucket examples into short/medium/long by word-count tercile."""
    lengths = np.asarray(lengths)
    t1, t2 = np.quantile(lengths, [q1, q2])
    slices = np.where(lengths <= t1, "short", np.where(lengths <= t2, "medium", "long"))
    return slices


def evaluate_classifier(y_true, y_pred, y_prob, slices=None, n_boot=1000, seed=0):
    """
    y_true, y_pred: 0/1 arrays
    y_prob: predicted probability of class 1
    slices: optional array of slice labels (same length as y_true) for per-slice metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)

    acc_ci = bootstrap_ci(y_true, y_pred, y_prob, lambda t, p: accuracy_score(t, p), n_boot=n_boot, seed=seed)
    f1_ci = bootstrap_ci(
        y_true, y_pred, y_prob,
        lambda t, p: precision_recall_fscore_support(t, p, average="macro", zero_division=0)[2],
        n_boot=n_boot, seed=seed,
    )
    mcc_ci = bootstrap_ci(y_true, y_pred, y_prob, lambda t, p: matthews_corrcoef(t, p) if len(set(t)) > 1 else 0.0,
                           n_boot=n_boot, seed=seed)

    result = {
        "accuracy": float(acc),
        "precision_macro": float(p_macro), "recall_macro": float(r_macro), "f1_macro": float(f1_macro),
        "precision_micro": float(p_micro), "recall_micro": float(r_micro), "f1_micro": float(f1_micro),
        "precision_weighted": float(p_weighted), "recall_weighted": float(r_weighted), "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "mcc": float(mcc),
        "brier_score": float(brier),
        "expected_calibration_error": float(ece),
        "accuracy_ci95": acc_ci,
        "macro_f1_ci95": f1_ci,
        "mcc_ci95": mcc_ci,
    }

    if slices is not None:
        slices = np.asarray(slices)
        per_slice = {}
        for s in sorted(set(slices)):
            mask = slices == s
            if mask.sum() == 0:
                continue
            _, _, f1_s, _ = precision_recall_fscore_support(y_true[mask], y_pred[mask], average="macro", zero_division=0)
            err_rate = float((y_pred[mask] != y_true[mask]).mean())
            per_slice[s] = {"macro_f1": float(f1_s), "error_rate": err_rate, "n": int(mask.sum())}
        result["per_slice"] = per_slice

    return result
