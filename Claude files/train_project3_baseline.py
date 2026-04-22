from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "project3_modeling_table.csv"
REPORT = ROOT / "project3_baseline_evaluation.md"
PREDICTIONS = ROOT / "project3_baseline_predictions_sample.csv"

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

DERIVED_AND_LEAKY_COLS = {
    "transaction_id",
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

# PCI DSS scope guard: these columns represent cardholder data (CHD) or
# sensitive authentication data (SAD). If any of them appear in the loaded
# dataframe, the pipeline refuses to train — processing them here would
# extend PCI scope to the entire ML training environment, artefact store,
# and downstream inference endpoint. The training dataset must stay
# synthetic (BIN, last4, categorical flags, amounts, decline codes only).
PCI_FORBIDDEN_COLS = frozenset({
    "pan", "card_number", "primary_account_number", "full_pan", "cardnumber",
    "cvv", "cvc", "cvv2", "cvc2", "cvn", "security_code",
    "track1", "track2", "track_data", "magstripe",
    "cardholder_name", "card_holder_name", "name_on_card",
    "pin", "pin_block",
})


def _fail_if_cardholder_data(df: pd.DataFrame) -> None:
    cols_lower = {c.lower() for c in df.columns}
    bad = sorted(PCI_FORBIDDEN_COLS & cols_lower)
    if bad:
        raise RuntimeError(
            f"PCI DSS guard: training dataset contains cardholder-data columns {bad}. "
            "Refusing to train. This pipeline accepts synthetic data only — BIN, last4, "
            "categorical verification flags, amounts, and decline codes are fine, but "
            "PAN / CVV / track / cardholder name / PIN must never enter training. "
            "Remove the offending columns or point DATASET at a synthetic file."
        )


def parse_bool_series(series: pd.Series) -> pd.Series:
    mapped = series.fillna(False).astype(str).str.lower().map({"true": True, "false": False})
    return mapped.fillna(series).astype(bool)


def load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DATASET, engine="python", on_bad_lines="skip")
    _fail_if_cardholder_data(df)
    for col in df.columns:
        if df[col].dtype == object:
            lowered = df[col].dropna().astype(str).str.lower()
            if not lowered.empty and lowered.isin(["true", "false"]).all():
                df[col] = parse_bool_series(df[col])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["event_hour"] = out["timestamp"].dt.hour.fillna(0).astype(int)
    out["event_dayofweek"] = out["timestamp"].dt.dayofweek.fillna(0).astype(int)
    out["event_month"] = out["timestamp"].dt.month.fillna(0).astype(int)
    out["is_weekend"] = out["event_dayofweek"].isin([5, 6])
    return out


def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def pick_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    candidate_cols = []
    for col in df.columns:
        if col in DERIVED_AND_LEAKY_COLS or col in IDENTIFIER_COLS:
            continue
        if "original_transaction_id" in col:
            continue
        if col == "timestamp":
            continue
        candidate_cols.append(col)

    bool_cols = [c for c in candidate_cols if pd.api.types.is_bool_dtype(df[c])]
    numeric_cols = [
        c for c in candidate_cols
        if pd.api.types.is_numeric_dtype(df[c]) and c not in bool_cols
    ]
    categorical_cols = [c for c in candidate_cols if c not in bool_cols and c not in numeric_cols]
    return numeric_cols, bool_cols, categorical_cols


def fit_category_maps(train_df: pd.DataFrame, categorical_cols: List[str], top_n: int = 12) -> Dict[str, List[str]]:
    maps: Dict[str, List[str]] = {}
    for col in categorical_cols:
        values = train_df[col].fillna("MISSING").astype(str)
        top_values = values.value_counts().head(top_n).index.tolist()
        maps[col] = top_values
    return maps


def prepare_matrix(
    df: pd.DataFrame,
    numeric_cols: List[str],
    bool_cols: List[str],
    categorical_cols: List[str],
    category_maps: Dict[str, List[str]] | None = None,
    medians: Dict[str, float] | None = None,
    means: Dict[str, float] | None = None,
    stds: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, Dict[str, List[str] | Dict[str, float]]]:
    work = df.copy()
    feature_names: List[str] = []
    matrices: List[np.ndarray] = []

    if medians is None:
        medians = {}
    if means is None:
        means = {}
    if stds is None:
        stds = {}
    if category_maps is None:
        category_maps = {}

    if numeric_cols:
        numeric_frame = pd.DataFrame(index=work.index)
        for col in numeric_cols:
            series = pd.to_numeric(work[col], errors="coerce")
            med = medians.get(col, float(series.median(skipna=True) if not series.dropna().empty else 0.0))
            filled = series.fillna(med)
            mean = means.get(col, float(filled.mean()))
            std = stds.get(col, float(filled.std(ddof=0)))
            if std == 0 or np.isnan(std):
                std = 1.0
            numeric_frame[col] = (filled - mean) / std
            medians[col] = med
            means[col] = mean
            stds[col] = std
            feature_names.append(col)
        matrices.append(numeric_frame.to_numpy(dtype=float))

    if bool_cols:
        bool_frame = work[bool_cols].fillna(False).astype(int)
        feature_names.extend(bool_cols)
        matrices.append(bool_frame.to_numpy(dtype=float))

    if categorical_cols:
        cat_parts: List[np.ndarray] = []
        for col in categorical_cols:
            values = work[col].fillna("MISSING").astype(str)
            known_values = category_maps.get(col)
            if known_values is None:
                known_values = values.value_counts().head(12).index.tolist()
                category_maps[col] = known_values
            capped = values.where(values.isin(known_values), "OTHER")
            dummies = pd.get_dummies(capped, prefix=col)
            expected_columns = [f"{col}_{value}" for value in known_values] + [f"{col}_OTHER"]
            for expected in expected_columns:
                if expected not in dummies.columns:
                    dummies[expected] = 0
            dummies = dummies[expected_columns]
            feature_names.extend(expected_columns)
            cat_parts.append(dummies.to_numpy(dtype=float))
        if cat_parts:
            matrices.append(np.concatenate(cat_parts, axis=1))

    X = np.concatenate(matrices, axis=1) if matrices else np.zeros((len(work), 0))
    return X, {
        "feature_names": feature_names,
        "category_maps": category_maps,
        "medians": medians,
        "means": means,
        "stds": stds,
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rate: float = 0.05,
    epochs: int = 900,
    l2: float = 0.001,
) -> Tuple[np.ndarray, float]:
    weights = np.zeros(X_train.shape[1], dtype=float)
    bias = 0.0
    best_weights = weights.copy()
    best_bias = bias
    best_val_loss = float("inf")
    patience = 50
    patience_left = patience

    for _ in range(epochs):
        preds = sigmoid(X_train @ weights + bias)
        error = preds - y_train
        grad_w = (X_train.T @ error) / len(X_train) + l2 * weights
        grad_b = float(error.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        val_preds = sigmoid(X_val @ weights + bias)
        val_loss = log_loss(y_val, val_preds)
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


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-12
    probs = np.clip(y_prob, eps, 1 - eps)
    return float(-(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs)).mean())


def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos = y_true == 1
    n_pos = pos.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2)) / (n_pos * n_neg)
    return float(auc)


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


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


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


def rule_baseline_probs(df: pd.DataFrame) -> np.ndarray:
    probs = np.zeros(len(df), dtype=float)
    soft = df["is_soft_decline"].fillna(False).astype(bool).to_numpy()
    probs[soft] = 0.30
    probs[~soft] = 0.01

    high_value = pd.to_numeric(df["amount_usd"], errors="coerce").fillna(0).to_numpy() > 150
    probs[soft & high_value] += 0.08

    code = df["response_code"].fillna("").astype(str)
    probs[code.eq("91").to_numpy()] += 0.10
    probs[code.eq("51").to_numpy()] += 0.05
    probs[code.isin(["54", "62", "14"]).to_numpy()] = 0.0
    return np.clip(probs, 0, 0.95)


def metrics_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    return {
        "auc": roc_auc_score_manual(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": float(y_pred.mean()),
    }


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
    positive_weights: List[Tuple[str, float]],
    negative_weights: List[Tuple[str, float]],
) -> str:
    lines = [
        "# Project 3 Baseline Evaluation",
        "",
        "## Setup",
        "- Model: custom numpy logistic regression baseline",
        "- Reason: bundled runtime did not include `scikit-learn` or `lightgbm`, so this run establishes a credible baseline with no external installs",
        "- Split: temporal 70/15/15 using the original decline timestamp",
        f"- Training rows: `{len(train_df)}`",
        f"- Validation rows: `{len(val_df)}`",
        f"- Test rows: `{len(test_df)}`",
        f"- Feature count after preprocessing and one-hot expansion: `{len(feature_names)}`",
        f"- Decision threshold selected on validation set: `{threshold:.2f}`",
        "",
        "## Validation Metrics",
        f"- AUC: `{val_metrics['auc']:.3f}`",
        f"- Log loss: `{val_metrics['log_loss']:.3f}`",
        f"- Brier score: `{val_metrics['brier']:.3f}`",
        f"- Precision: `{val_metrics['precision']:.3f}`",
        f"- Recall: `{val_metrics['recall']:.3f}`",
        f"- F1: `{val_metrics['f1']:.3f}`",
        f"- Predicted positive rate: `{val_metrics['positive_rate']:.3f}`",
        "",
        "## Test Metrics",
        f"- AUC: `{test_metrics['auc']:.3f}`",
        f"- Log loss: `{test_metrics['log_loss']:.3f}`",
        f"- Brier score: `{test_metrics['brier']:.3f}`",
        f"- Precision: `{test_metrics['precision']:.3f}`",
        f"- Recall: `{test_metrics['recall']:.3f}`",
        f"- F1: `{test_metrics['f1']:.3f}`",
        f"- Predicted positive rate: `{test_metrics['positive_rate']:.3f}`",
        "",
        "## Rule Baseline on Test Set",
        "- Baseline rule: retry soft declines by default, boost certain recoverable codes and higher-value transactions, never retry obviously hard declines",
        f"- AUC: `{rule_metrics['auc']:.3f}`",
        f"- Log loss: `{rule_metrics['log_loss']:.3f}`",
        f"- Brier score: `{rule_metrics['brier']:.3f}`",
        f"- Precision: `{rule_metrics['precision']:.3f}`",
        f"- Recall: `{rule_metrics['recall']:.3f}`",
        f"- F1: `{rule_metrics['f1']:.3f}`",
        "",
        "## Interpretation",
        "- This baseline is meant to validate that the richer original-decline feature set carries predictive signal before we invest in a more advanced model stack.",
        "- If the learned model beats the soft-decline rule baseline cleanly, that supports staying with the current dataset for the next stage.",
        "- If the learned model only marginally improves on the rule baseline, the next lever may be better feature engineering or a stronger tree-based model rather than immediately generating more data.",
        "",
        "## Strongest Positive Signals",
    ]
    for name, weight in positive_weights:
        lines.append(f"- `{name}`: `{weight:.3f}`")
    lines.extend(["", "## Strongest Negative Signals"])
    for name, weight in negative_weights:
        lines.append(f"- `{name}`: `{weight:.3f}`")
    lines.extend(
        [
            "",
            "## Recommendation",
            "- Proceed to a stronger training implementation next.",
            "- Preferred next model: LightGBM or another tree-based tabular learner once the runtime supports it.",
            "- Keep the current modeling table as the training foundation.",
            "- Revisit data generation only after comparing this baseline with a stronger model and checking segment stability.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    with maybe_mlflow_run("project3_baseline_v1") as mlflow_client:
        df = add_time_features(load_dataframe())
        train_df, val_df, test_df = temporal_split(df)

        numeric_cols, bool_cols, categorical_cols = pick_feature_columns(df)

        X_train, prep = prepare_matrix(train_df, numeric_cols, bool_cols, categorical_cols)
        X_val, _ = prepare_matrix(
            val_df,
            numeric_cols,
            bool_cols,
            categorical_cols,
            category_maps=prep["category_maps"],
            medians=prep["medians"],
            means=prep["means"],
            stds=prep["stds"],
        )
        X_test, _ = prepare_matrix(
            test_df,
            numeric_cols,
            bool_cols,
            categorical_cols,
            category_maps=prep["category_maps"],
            medians=prep["medians"],
            means=prep["means"],
            stds=prep["stds"],
        )

        y_train = train_df[TARGET].astype(int).to_numpy()
        y_val = val_df[TARGET].astype(int).to_numpy()
        y_test = test_df[TARGET].astype(int).to_numpy()

        weights, bias = train_logistic_regression(X_train, y_train, X_val, y_val)
        val_probs = sigmoid(X_val @ weights + bias)
        test_probs = sigmoid(X_test @ weights + bias)

        threshold, _ = find_best_threshold(y_val, val_probs)
        val_metrics = metrics_dict(y_val, val_probs, threshold)
        test_metrics = metrics_dict(y_test, test_probs, threshold)

        rule_probs = rule_baseline_probs(test_df)
        rule_metrics = metrics_dict(y_test, rule_probs, threshold)

        positive_weights, negative_weights = top_weights(weights, prep["feature_names"])
        REPORT.write_text(
            build_report(
                train_df,
                val_df,
                test_df,
                prep["feature_names"],
                threshold,
                val_metrics,
                test_metrics,
                rule_metrics,
                positive_weights,
                negative_weights,
            ),
            encoding="utf-8",
        )

        sample = test_df[["timestamp", "merchant_country", "processor_name", "response_code", "amount_usd", TARGET]].copy()
        sample["predicted_recovery_probability"] = test_probs
        sample["rule_baseline_probability"] = rule_probs
        sample = sample.sort_values("predicted_recovery_probability", ascending=False).head(200)
        sample.to_csv(PREDICTIONS, index=False)

        if mlflow_client is not None:
            mlflow_client.log_params({
                "model_family": "numpy_logistic_regression",
                "variant": "baseline_v1",
                "target": TARGET,
                "feature_count": len(prep["feature_names"]),
                "numeric_feature_count": len(numeric_cols),
                "bool_feature_count": len(bool_cols),
                "categorical_feature_count": len(categorical_cols),
                "decision_threshold": threshold,
            })
            mlflow_client.log_metrics({f"val_{k}": float(v) for k, v in val_metrics.items()})
            mlflow_client.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})
            mlflow_client.log_artifact(str(REPORT))
            mlflow_client.log_artifact(str(PREDICTIONS))

    print(REPORT)
    print(PREDICTIONS)


if __name__ == "__main__":
    main()
