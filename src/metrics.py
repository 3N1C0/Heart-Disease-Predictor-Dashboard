"""Threshold metrics on a probability vector (exam Part 7)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def predictions_at_threshold(y_prob, threshold: float) -> np.ndarray:
    return (np.asarray(y_prob) >= threshold).astype(int)


def threshold_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = predictions_at_threshold(y_prob, threshold)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": threshold,
        "n_sent": int(y_pred.sum()),
        "missed": int(fn),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def recall_by_sex(y_true, y_pred, sex) -> dict:
    sex = np.asarray(sex)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    women = sex == "female"
    return {
        "recall_women": float(recall_score(y_true[women], y_pred[women], zero_division=0)),
        "recall_men": float(recall_score(y_true[~women], y_pred[~women], zero_division=0)),
        "n_women_disease": int(y_true[women].sum()),
    }


def roc_points(y_true, y_prob) -> dict:
    fpr, tpr, cutoffs = roc_curve(y_true, y_prob)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "cutoffs": cutoffs,
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


def delta_vs_half(metrics_t: dict, metrics_half: dict) -> dict:
    return {
        "more_sent": metrics_t["n_sent"] - metrics_half["n_sent"],
        "fewer_missed": metrics_half["missed"] - metrics_t["missed"],
    }
