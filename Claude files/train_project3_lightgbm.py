from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "project3_modeling_table.csv"
FEATURE_POLICY = ROOT / "project3_feature_policy.json"
REPORT = ROOT / "project3_lightgbm_evaluation.md"
SEGMENTS = ROOT / "project3_lightgbm_segment_metrics.csv"
FEATURE_IMPORTANCE = ROOT / "project3_lightgbm_feature_importance.csv"
PREDICTIONS = ROOT / "project3_lightgbm_predictions_sample.csv"
MODEL_TXT = ROOT / "project3_lightgbm_model.txt"
MODEL_META = ROOT / "project3_lightgbm_model_metadata.json"

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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def parse_bool(series: pd.Series) -> pd.Series:
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


def load_feature_policy() -> Dict[str, List[str]]:
    return json.loads(FEATURE_POLICY.read_text(encoding="utf-8"))


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


def select_features(df: pd.DataFrame, policy: Dict[str, List[str]]) -> List[str]:
    allowlist = set(policy["allowlist"])
    present = [col for col in df.columns if col in allowlist]
    return present


def coerce_feature_types(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    work = df[feature_cols].copy()
    bool_cols: List[str] = []
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in feature_cols:
        if pd.api.types.is_bool_dtype(work[col]):
            work[col] = work[col].fillna(False).astype(int)
            bool_cols.append(col)
        elif pd.api.types.is_numeric_dtype(work[col]):
            work[col] = pd.to_numeric(work[col], errors="coerce")
            numeric_cols.append(col)
        else:
            work[col] = work[col].fillna("MISSING").astype(str)
            categorical_cols.append(col)

    return work, numeric_cols, bool_cols, categorical_cols


def rule_baseline_probs(df: pd.DataFrame) -> np.ndarray:
    probs = np.zeros(len(df), dtype=float)
    soft = df["is_soft_decline"].fillna(False).astype(bool).to_numpy()
    probs[soft] = 0.30
    probs[~soft] = 0.01

    amount = pd.to_numeric(df["amount_usd"], errors="coerce").fillna(0).to_numpy()
    probs[soft & (amount > 150)] += 0.08

    code = df["response_code"].fillna("").astype(str)
    probs[code.eq("91").to_numpy()] += 0.10
    probs[code.eq("51").to_numpy()] += 0.05
    probs[code.isin(["54", "62", "14"]).to_numpy()] = 0.0
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


def build_report(
    feature_cols: List[str],
    numeric_cols: List[str],
    bool_cols: List[str],
    categorical_cols: List[str],
    threshold: float,
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    rule_metrics: Dict[str, float],
    segment_df: pd.DataFrame,
    top_features: pd.DataFrame,
) -> str:
    lines = [
        "# Project 3 LightGBM Evaluation",
        "",
        "## Setup",
        "- Model: LightGBM binary classifier",
        "- Training objective: predict whether an original declined transaction is eventually recoverable by retry",
        "- Split: temporal 70/15/15",
        f"- Allowed features from policy: `{len(feature_cols)}`",
        f"- Numeric features: `{len(numeric_cols)}`",
        f"- Boolean features: `{len(bool_cols)}`",
        f"- Categorical features: `{len(categorical_cols)}`",
        f"- Decision threshold chosen on validation set: `{threshold:.2f}`",
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
        "## Top Feature Importances",
    ]

    for _, row in top_features.head(20).iterrows():
        lines.append(f"- `{row['feature']}`: `{row['importance']}`")

    lines.extend(["", "## Segment Stability Snapshot"])
    for _, row in segment_df.head(16).iterrows():
        lines.append(
            f"- `{row['segment_column']}={row['segment_value']}`: rows `{int(row['rows'])}`, positive rate `{row['positive_rate']:.2%}`, AUC `{row['auc']:.3f}`, F1 `{row['f1']:.3f}`"
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "- Compare this LightGBM run directly against the V2 logistic baseline.",
            "- If the gain is meaningful, promote LightGBM as the canonical model in the project story.",
            "- Next after that: SHAP explanations and a business decision layer.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit(
            "LightGBM is not installed in this runtime. "
            "This script is ready to run once `lightgbm` is available."
        ) from exc

    with maybe_mlflow_run("project3_lightgbm") as mlflow_client:
        policy = load_feature_policy()
        df = load_data()
        feature_cols = select_features(df, policy)
        train_df, val_df, test_df = temporal_split(df)

        X_train_df, numeric_cols, bool_cols, categorical_cols = coerce_feature_types(train_df, feature_cols)
        X_val_df, _, _, _ = coerce_feature_types(val_df, feature_cols)
        X_test_df, _, _, _ = coerce_feature_types(test_df, feature_cols)

        for col in categorical_cols:
            cats = sorted(X_train_df[col].fillna("MISSING").astype(str).unique().tolist())
            X_train_df[col] = pd.Categorical(X_train_df[col], categories=cats)
            X_val_df[col] = pd.Categorical(X_val_df[col].fillna("MISSING").astype(str), categories=cats)
            X_test_df[col] = pd.Categorical(X_test_df[col].fillna("MISSING").astype(str), categories=cats)

        y_train = train_df[TARGET].astype(int).to_numpy()
        y_val = val_df[TARGET].astype(int).to_numpy()
        y_test = test_df[TARGET].astype(int).to_numpy()

        params = {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "min_data_in_leaf": 40,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": 42,
        }

        train_set = lgb.Dataset(X_train_df, label=y_train, categorical_feature=categorical_cols, free_raw_data=False)
        val_set = lgb.Dataset(X_val_df, label=y_val, categorical_feature=categorical_cols, reference=train_set, free_raw_data=False)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=1000,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)],
        )

        val_probs = model.predict(X_val_df, num_iteration=model.best_iteration)
        test_probs = model.predict(X_test_df, num_iteration=model.best_iteration)
        threshold, _ = find_best_threshold(y_val, val_probs)

        val_metrics = metrics(y_val, val_probs, threshold)
        test_metrics = metrics(y_test, test_probs, threshold)
        rule_metrics = metrics(y_test, rule_baseline_probs(test_df), threshold)

        importance = pd.DataFrame(
            {
                "feature": model.feature_name(),
                "importance": model.feature_importance(importance_type="gain"),
            }
        ).sort_values("importance", ascending=False)
        importance.to_csv(FEATURE_IMPORTANCE, index=False)

        segment_df = segment_metrics(test_df, y_test, test_probs, threshold)
        segment_df.to_csv(SEGMENTS, index=False)

        sample = test_df[
            ["timestamp", "merchant_country", "processor_name", "response_code", "amount_usd", TARGET]
        ].copy()
        sample["predicted_recovery_probability"] = test_probs
        sample["rule_baseline_probability"] = rule_baseline_probs(test_df)
        sample.sort_values("predicted_recovery_probability", ascending=False).head(200).to_csv(PREDICTIONS, index=False)

        metadata = {
            "target": TARGET,
            "feature_policy_version": policy.get("version", "v1"),
            "feature_columns": feature_cols,
            "numeric_columns": numeric_cols,
            "boolean_columns": bool_cols,
            "categorical_columns": categorical_cols,
            "decision_threshold": threshold,
            "best_iteration": model.best_iteration,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        MODEL_META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        model.save_model(str(MODEL_TXT), num_iteration=model.best_iteration)

        REPORT.write_text(
            build_report(
                feature_cols,
                numeric_cols,
                bool_cols,
                categorical_cols,
                threshold,
                val_metrics,
                test_metrics,
                rule_metrics,
                segment_df,
                importance,
            ),
            encoding="utf-8",
        )

        if mlflow_client is not None:
            mlflow_client.log_params({
                "model_family": "lightgbm",
                "target": TARGET,
                "feature_count": len(feature_cols),
                "learning_rate": params["learning_rate"],
                "num_leaves": params["num_leaves"],
                "feature_fraction": params["feature_fraction"],
                "bagging_fraction": params["bagging_fraction"],
                "min_data_in_leaf": params["min_data_in_leaf"],
                "lambda_l2": params["lambda_l2"],
                "decision_threshold": threshold,
                "best_iteration": model.best_iteration,
            })
            mlflow_client.log_metrics({f"val_{k}": float(v) for k, v in val_metrics.items()})
            mlflow_client.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})
            mlflow_client.log_artifact(str(MODEL_TXT))
            mlflow_client.log_artifact(str(MODEL_META))
            mlflow_client.log_artifact(str(REPORT))
            mlflow_client.log_artifact(str(SEGMENTS))
            mlflow_client.log_artifact(str(FEATURE_IMPORTANCE))
            try:
                mlflow_client.lightgbm.log_model(model, name="model")
            except Exception:
                pass

    print(REPORT)
    print(SEGMENTS)
    print(FEATURE_IMPORTANCE)
    print(PREDICTIONS)


if __name__ == "__main__":
    main()
