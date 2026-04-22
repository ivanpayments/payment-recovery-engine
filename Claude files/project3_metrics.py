"""Shared metric + preprocessing utilities for Project 3 training scripts.

Hand-rolled rather than pulled from scikit-learn so the training pipeline keeps
its dependency surface narrow (numpy + pandas only for the math bits).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def parse_bool_series(series: pd.Series) -> pd.Series:
    mapped = series.fillna(False).astype(str).str.lower().map({"true": True, "false": False})
    return mapped.fillna(series).astype(bool)


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())


def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2)) / (n_pos * n_neg)
    return float(auc)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    tp, fp, _, fn = confusion(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.50, 46):
        preds = (y_prob >= threshold).astype(int)
        _, _, f1 = precision_recall_f1(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, best_f1


def metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (y_prob >= threshold).astype(int)
    precision, recall, f1 = precision_recall_f1(y_true, preds)
    return {
        "auc": roc_auc_score_manual(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": float(preds.mean()),
    }
