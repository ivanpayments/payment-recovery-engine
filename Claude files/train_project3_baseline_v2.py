from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "project3_modeling_table.csv"
REPORT = ROOT / "project3_baseline_v2_evaluation.md"
SEGMENTS = ROOT / "project3_baseline_v2_segment_metrics.csv"
PREDICTIONS = ROOT / "project3_baseline_v2_predictions_sample.csv"

TARGET = "target_recovered_by_retry"
MLFLOW_DIR = ROOT / "mlruns"
DEFAULT_MLFLOW_TRACKING_URI = "file:///C:/Users/ivana/.codex/memories/project3-mlruns"


@contextlib.contextmanager
def maybe_mlflow_run(run_name: str):
    try:
        import mlflow
    except ImportError:
        yield None
        return

    os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
    mlflow.set_experiment("project3-payment-recovery")
    with mlflow.start_run(run_name=run_name):
        yield mlflow

LEAKY_COLS = {
    "transaction_id",
    "timestamp",
    "auth_status",
    "is_retry",
    "original_transaction_id",
    "original_transaction_id_x",
    "original_transaction_id_y",
    "retry_attempt_num",
    "retry_reason",
    "hours_since_original",
    "has_any_retry",
    "retry_count",
    "first_retry_status",
    "first_retry_attempt_num",
    "hours_to_first_retry",
    "recovery_attempt_num",
    "hours_to_recovery",
    "target_recovered_by_retry",
    "target_first_retry_approved",
    "approved_amount",
    "auth_code",
    "captured_amount",
    "refunded_amount",
    "captured_at",
    "refunded_at",
    "voided_at",
    "settled_at",
    "chargeback_reason_code",
}

IDENTIFIER_COLS = {
    "merchant_id",
    "psp_transaction_id",
    "psp_reference",
    "gateway_id",
    "network_transaction_id",
    "stan",
    "rrn",
    "arn",
    "session_id",
    "correlation_id",
    "trace_id",
    "device_fingerprint",
    "subscription_id",
    "wallet_token",
    "merchant_descriptor",
    "dynamic_descriptor",
    "soft_descriptor",
    "terminal_id",
    "billing_zip",
    "billing_city",
    "billing_state",
    "issuer_bank_name",
    "issuer_bank_bin_range",
    "stored_credential_id",
    "bin_first6",
    "acquirer_bin",
}


def parse_bool(series: pd.Series) -> pd.Series:
    mapped = series.fillna(False).astype(str).str.lower().map({"true": True, "false": False})
    return mapped.fillna(series).astype(bool)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


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
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    return {
        "auc": roc_auc_score_manual(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": float(y_pred.mean()),
    }


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET, engine="python", on_bad_lines="skip")
    for col in df.columns:
        if df[col].dtype == object:
            lowered = df[col].dropna().astype(str).str.lower()
            if not lowered.empty and lowered.isin(["true", "false"]).all():
                df[col] = parse_bool(df[col])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["event_hour"] = df["timestamp"].dt.hour.fillna(0).astype(int)
    df["event_dayofweek"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
    df["event_month"] = df["timestamp"].dt.month.fillna(0).astype(int)
    df["is_weekend"] = df["event_dayofweek"].isin([5, 6])
    return df


def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    candidates = []
    for col in df.columns:
        if col in LEAKY_COLS or col in IDENTIFIER_COLS:
            continue
        if "original_transaction_id" in col:
            continue
        if col == "timestamp":
            continue
        candidates.append(col)

    bool_cols = [c for c in candidates if pd.api.types.is_bool_dtype(df[c])]
    numeric_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c]) and c not in bool_cols]
    categorical_cols = [c for c in candidates if c not in bool_cols and c not in numeric_cols]
    return numeric_cols, bool_cols, categorical_cols


def fit_numeric_scaler(train_df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        series = pd.to_numeric(train_df[col], errors="coerce")
        median = float(series.median(skipna=True) if not series.dropna().empty else 0.0)
        filled = series.fillna(median)
        mean = float(filled.mean())
        std = float(filled.std(ddof=0))
        if std == 0 or np.isnan(std):
            std = 1.0
        stats[col] = {"median": median, "mean": mean, "std": std}
    return stats


def apply_numeric_scaler(df: pd.DataFrame, numeric_cols: List[str], stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    arr = np.zeros((len(df), len(numeric_cols)), dtype=float)
    for i, col in enumerate(numeric_cols):
        series = pd.to_numeric(df[col], errors="coerce").fillna(stats[col]["median"])
        arr[:, i] = (series.to_numpy(dtype=float) - stats[col]["mean"]) / stats[col]["std"]
    return arr


def fit_target_encoding(
    train_df: pd.DataFrame,
    categorical_cols: List[str],
    target_col: str,
    smoothing: float = 20.0,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    global_mean = float(train_df[target_col].mean())
    encoders: Dict[str, Dict[str, float]] = {}
    y = train_df[target_col].astype(float)

    for col in categorical_cols:
        tmp = pd.DataFrame({"x": train_df[col].fillna("MISSING").astype(str), "y": y})
        stats = tmp.groupby("x")["y"].agg(["mean", "count"])
        enc = ((stats["mean"] * stats["count"]) + global_mean * smoothing) / (stats["count"] + smoothing)
        encoders[col] = enc.to_dict()
    return encoders, global_mean


def apply_target_encoding(
    df: pd.DataFrame,
    categorical_cols: List[str],
    encoders: Dict[str, Dict[str, float]],
    global_mean: float,
) -> np.ndarray:
    arr = np.zeros((len(df), len(categorical_cols)), dtype=float)
    for i, col in enumerate(categorical_cols):
        values = df[col].fillna("MISSING").astype(str)
        mapping = encoders[col]
        arr[:, i] = values.map(mapping).fillna(global_mean).to_numpy(dtype=float)
    return arr


def build_matrix(
    df: pd.DataFrame,
    numeric_cols: List[str],
    bool_cols: List[str],
    categorical_cols: List[str],
    numeric_stats: Dict[str, Dict[str, float]],
    encoders: Dict[str, Dict[str, float]],
    global_mean: float,
) -> Tuple[np.ndarray, List[str]]:
    parts = []
    names = []

    if numeric_cols:
        parts.append(apply_numeric_scaler(df, numeric_cols, numeric_stats))
        names.extend(numeric_cols)

    if bool_cols:
        parts.append(df[bool_cols].fillna(False).astype(int).to_numpy(dtype=float))
        names.extend(bool_cols)

    if categorical_cols:
        parts.append(apply_target_encoding(df, categorical_cols, encoders, global_mean))
        names.extend([f"{c}__target_encoded" for c in categorical_cols])

    X = np.concatenate(parts, axis=1) if parts else np.zeros((len(df), 0))
    return X, names


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rate: float = 0.05,
    epochs: int = 1200,
    l2: float = 0.002,
) -> Tuple[np.ndarray, float]:
    weights = np.zeros(X_train.shape[1], dtype=float)
    bias = 0.0
    best_weights = weights.copy()
    best_bias = bias
    best_val_loss = float("inf")
    patience = 70
    patience_left = patience

    for _ in range(epochs):
        preds = sigmoid(X_train @ weights + bias)
        error = preds - y_train
        grad_w = (X_train.T @ error) / len(X_train) + l2 * weights
        grad_b = float(error.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        val_loss = log_loss(y_val, sigmoid(X_val @ weights + bias))
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_weights = weights.copy()
            best_bias = bias
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    return best_weights, best_bias


def rule_baseline_probs(df: pd.DataFrame) -> np.ndarray:
    probs = np.zeros(len(df), dtype=float)
    soft = df["is_soft_decline"].fillna(False).astype(bool).to_numpy()
    probs[soft] = 0.30
    probs[~soft] = 0.01

    amount = pd.to_numeric(df["amount_usd"], errors="coerce").fillna(0).to_numpy()
    high_value = amount > 150
    probs[soft & high_value] += 0.08

    codes = df["response_code"].fillna("").astype(str)
    probs[codes.eq("91").to_numpy()] += 0.10
    probs[codes.eq("51").to_numpy()] += 0.05
    probs[codes.isin(["54", "62", "14"]).to_numpy()] = 0.0
    return np.clip(probs, 0, 0.95)


def segment_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    records = []
    segment_specs = [
        ("merchant_country", 8),
        ("processor_name", 8),
        ("response_code", 8),
        ("merchant_vertical", 6),
    ]
    for col, limit in segment_specs:
        top_values = df[col].fillna("MISSING").astype(str).value_counts().head(limit).index.tolist()
        for value in top_values:
            mask = df[col].fillna("MISSING").astype(str).eq(value).to_numpy()
            count = int(mask.sum())
            if count < 25:
                continue
            seg_y = y_true[mask]
            seg_p = y_prob[mask]
            seg_pred = (seg_p >= threshold).astype(int)
            precision, recall, f1 = precision_recall_f1(seg_y, seg_pred)
            records.append(
                {
                    "segment_column": col,
                    "segment_value": value,
                    "rows": count,
                    "positive_rate": float(seg_y.mean()),
                    "auc": roc_auc_score_manual(seg_y, seg_p),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return pd.DataFrame(records).sort_values(["segment_column", "rows"], ascending=[True, False])


def top_weights(weights: np.ndarray, feature_names: List[str], top_n: int = 15) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    pairs = list(zip(feature_names, weights))
    positive = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_n]
    negative = sorted(pairs, key=lambda x: x[1])[:top_n]
    return positive, negative


def build_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: List[str],
    threshold: float,
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    rule_metrics: Dict[str, float],
    pos_weights: List[Tuple[str, float]],
    neg_weights: List[Tuple[str, float]],
    segment_df: pd.DataFrame,
) -> str:
    lines = [
        "# Project 3 Baseline Evaluation V2",
        "",
        "## Setup",
        "- Model: custom numpy logistic regression baseline with target encoding for categorical features",
        "- Why this version: less noisy than wide one-hot encoding, better fit for a self-contained runtime, and easier to inspect for segment stability",
        "- Split: temporal 70/15/15 using original decline timestamp",
        f"- Training rows: `{len(train_df)}`",
        f"- Validation rows: `{len(val_df)}`",
        f"- Test rows: `{len(test_df)}`",
        f"- Feature count after preprocessing: `{len(feature_names)}`",
        f"- Decision threshold selected on validation set: `{threshold:.2f}`",
        "",
        "## Validation Metrics",
        f"- AUC: `{val_metrics['auc']:.3f}`",
        f"- Log loss: `{val_metrics['log_loss']:.3f}`",
        f"- Brier score: `{val_metrics['brier']:.3f}`",
        f"- Precision: `{val_metrics['precision']:.3f}`",
        f"- Recall: `{val_metrics['recall']:.3f}`",
        f"- F1: `{val_metrics['f1']:.3f}`",
        "",
        "## Test Metrics",
        f"- AUC: `{test_metrics['auc']:.3f}`",
        f"- Log loss: `{test_metrics['log_loss']:.3f}`",
        f"- Brier score: `{test_metrics['brier']:.3f}`",
        f"- Precision: `{test_metrics['precision']:.3f}`",
        f"- Recall: `{test_metrics['recall']:.3f}`",
        f"- F1: `{test_metrics['f1']:.3f}`",
        "",
        "## Rule Baseline on Test Set",
        f"- AUC: `{rule_metrics['auc']:.3f}`",
        f"- Log loss: `{rule_metrics['log_loss']:.3f}`",
        f"- Brier score: `{rule_metrics['brier']:.3f}`",
        f"- Precision: `{rule_metrics['precision']:.3f}`",
        f"- Recall: `{rule_metrics['recall']:.3f}`",
        f"- F1: `{rule_metrics['f1']:.3f}`",
        "",
        "## Readout",
        "- This V2 baseline is meant to be more realistic and less overfit than the first wide one-hot pass.",
        "- If it still beats the rules baseline clearly, that is a good sign that the dataset is sufficient for the next modeling stage.",
        "",
        "## Strongest Positive Features",
    ]
    for name, value in pos_weights:
        lines.append(f"- `{name}`: `{value:.3f}`")
    lines.extend(["", "## Strongest Negative Features"])
    for name, value in neg_weights:
        lines.append(f"- `{name}`: `{value:.3f}`")

    lines.extend(["", "## Segment Stability Snapshot"])
    preview = segment_df.head(16)
    for _, row in preview.iterrows():
        lines.append(
            f"- `{row['segment_column']}={row['segment_value']}`: rows `{int(row['rows'])}`, positive rate `{row['positive_rate']:.2%}`, AUC `{row['auc']:.3f}`, F1 `{row['f1']:.3f}`"
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "- Keep the current modeling table and proceed to a proper tree-based model when the runtime permits it.",
            "- Use this V2 run as the defensible baseline for the case study.",
            "- Only expand the synthetic dataset if a stronger model still shows weak segment stability or obvious undercoverage.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    with maybe_mlflow_run("project3_baseline_v2") as mlflow_client:
        df = load_data()
        train_df, val_df, test_df = temporal_split(df)
        numeric_cols, bool_cols, categorical_cols = feature_lists(df)

        numeric_stats = fit_numeric_scaler(train_df, numeric_cols)
        encoders, global_mean = fit_target_encoding(train_df, categorical_cols, TARGET)

        X_train, feature_names = build_matrix(
            train_df, numeric_cols, bool_cols, categorical_cols, numeric_stats, encoders, global_mean
        )
        X_val, _ = build_matrix(
            val_df, numeric_cols, bool_cols, categorical_cols, numeric_stats, encoders, global_mean
        )
        X_test, _ = build_matrix(
            test_df, numeric_cols, bool_cols, categorical_cols, numeric_stats, encoders, global_mean
        )

        y_train = train_df[TARGET].astype(int).to_numpy()
        y_val = val_df[TARGET].astype(int).to_numpy()
        y_test = test_df[TARGET].astype(int).to_numpy()

        weights, bias = train_logistic_regression(X_train, y_train, X_val, y_val)
        val_probs = sigmoid(X_val @ weights + bias)
        test_probs = sigmoid(X_test @ weights + bias)

        threshold, _ = find_best_threshold(y_val, val_probs)
        val_metrics = metrics(y_val, val_probs, threshold)
        test_metrics = metrics(y_test, test_probs, threshold)

        rule_probs = rule_baseline_probs(test_df)
        rule_metrics = metrics(y_test, rule_probs, threshold)

        pos_weights, neg_weights = top_weights(weights, feature_names)
        segment_df = segment_metrics(test_df, y_test, test_probs, threshold)
        segment_df.to_csv(SEGMENTS, index=False)

        sample = test_df[
            ["timestamp", "merchant_country", "processor_name", "response_code", "amount_usd", TARGET]
        ].copy()
        sample["predicted_recovery_probability"] = test_probs
        sample["rule_baseline_probability"] = rule_probs
        sample.sort_values("predicted_recovery_probability", ascending=False).head(200).to_csv(PREDICTIONS, index=False)

        REPORT.write_text(
            build_report(
                train_df,
                val_df,
                test_df,
                feature_names,
                threshold,
                val_metrics,
                test_metrics,
                rule_metrics,
                pos_weights,
                neg_weights,
                segment_df,
            ),
            encoding="utf-8",
        )

        if mlflow_client is not None:
            mlflow_client.log_params({
                "model_family": "numpy_logistic_regression",
                "variant": "baseline_v2",
                "target": TARGET,
                "feature_count": len(feature_names),
                "numeric_feature_count": len(numeric_cols),
                "bool_feature_count": len(bool_cols),
                "categorical_feature_count": len(categorical_cols),
                "decision_threshold": threshold,
            })
            mlflow_client.log_metrics({f"val_{k}": float(v) for k, v in val_metrics.items()})
            mlflow_client.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})
            mlflow_client.log_artifact(str(REPORT))
            mlflow_client.log_artifact(str(SEGMENTS))
            mlflow_client.log_artifact(str(PREDICTIONS))

    print(REPORT)
    print(SEGMENTS)
    print(PREDICTIONS)


if __name__ == "__main__":
    main()
