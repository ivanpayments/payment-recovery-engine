from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from evaluate_project3_decision_policy import (
    DEFAULT_FRICTION_COST_USD,
    DEFAULT_MARGIN_RATE,
    DEFAULT_RETRY_COST_USD,
    decision_from_prob_and_value,
    expected_retry_value,
    optimize_threshold_for_net_value,
)
from train_project3_lightgbm import MODEL_META, MODEL_TXT, load_data, temporal_split


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = ROOT / "project3_recovery_simulator_output.csv"
DEFAULT_OUTPUT_MD = ROOT / "project3_recovery_simulator_report.md"


def load_model_and_metadata():
    import lightgbm as lgb

    if not MODEL_TXT.exists() or not MODEL_META.exists():
        raise SystemExit("Model artifacts missing. Run train_project3_lightgbm.py first.")

    booster = lgb.Booster(model_file=str(MODEL_TXT))
    metadata = json.loads(MODEL_META.read_text(encoding="utf-8"))
    return booster, metadata


def prepare_features(df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
    feature_cols = metadata["feature_columns"]
    categorical_cols = set(metadata["categorical_columns"])
    work = df[feature_cols].copy()
    train_df, _, _ = temporal_split(load_data())

    for col in feature_cols:
        if col in categorical_cols:
            cats = sorted(train_df[col].fillna("MISSING").astype(str).unique().tolist())
            work[col] = pd.Categorical(df[col].fillna("MISSING").astype(str), categories=cats)
        elif pd.api.types.is_bool_dtype(df[col]):
            work[col] = df[col].fillna(False).astype(int)
        else:
            work[col] = pd.to_numeric(df[col], errors="coerce")
    return work


from project3_runtime import business_phrase  # noqa: E402  (re-exported for callers)


def score_batch(df: pd.DataFrame, booster, metadata: Dict, threshold: float) -> pd.DataFrame:
    import shap

    X = prepare_features(df, metadata)
    probs = booster.predict(X, num_iteration=metadata.get("best_iteration"))
    amount = pd.to_numeric(df["amount_usd"], errors="coerce").fillna(0).to_numpy()
    exp_value = expected_retry_value(probs, amount)
    decisions = decision_from_prob_and_value(probs, exp_value, threshold)

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X)
    shap_matrix = np.array(shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values[0] if isinstance(shap_values, list) else shap_values)
    if shap_matrix.shape[1] == X.shape[1] + 1:
        shap_matrix = shap_matrix[:, :-1]

    rows: List[Dict] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        row_shap = shap_matrix[idx]
        top_feature_idx = np.argsort(np.abs(row_shap))[-3:][::-1]
        top_features = []
        top_explanations = []
        for feat_idx in top_feature_idx:
            feature = X.columns[feat_idx]
            value = row[feature]
            shap_value = float(row_shap[feat_idx])
            top_features.append(feature)
            top_explanations.append(business_phrase(feature, value, shap_value))

        rows.append(
            {
                "timestamp": row["timestamp"],
                "merchant_country": row["merchant_country"],
                "processor_name": row["processor_name"],
                "merchant_vertical": row["merchant_vertical"],
                "response_code": row["response_code"],
                "amount_usd": row["amount_usd"],
                "predicted_recovery_probability": float(probs[idx]),
                "expected_retry_value_usd": float(exp_value[idx]),
                "decision": decisions[idx],
                "top_feature_1": top_features[0] if len(top_features) > 0 else "",
                "top_feature_2": top_features[1] if len(top_features) > 1 else "",
                "top_feature_3": top_features[2] if len(top_features) > 2 else "",
                "explanation_1": top_explanations[0] if len(top_explanations) > 0 else "",
                "explanation_2": top_explanations[1] if len(top_explanations) > 1 else "",
                "explanation_3": top_explanations[2] if len(top_explanations) > 2 else "",
                "actual_recovered_by_retry": row.get("target_recovered_by_retry", ""),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["decision", "expected_retry_value_usd", "predicted_recovery_probability"],
        ascending=[True, False, False],
    )


def build_report(scored: pd.DataFrame, sample_size: int, source_desc: str, threshold: float, threshold_basis: str) -> str:
    retry_df = scored[scored["decision"] == "retry"].copy()
    skip_df = scored[scored["decision"] == "do_not_retry"].copy()
    recovered_col = "actual_recovered_by_retry"
    retry_recovered = int(retry_df[recovered_col].fillna(False).astype(bool).sum()) if recovered_col in retry_df.columns else 0

    lines = [
        "# Project 3 Recovery Simulator Report",
        "",
        "## Run Setup",
        f"- Source: {source_desc}",
        f"- Rows scored: `{len(scored)}`",
        f"- Decision threshold used: `{threshold:.2f}`",
        f"- Threshold basis: `{threshold_basis}`",
        f"- Retry cost assumption: `${DEFAULT_RETRY_COST_USD:.2f}`",
        f"- Margin rate assumption: `{DEFAULT_MARGIN_RATE:.0%}`",
        f"- Friction cost assumption: `${DEFAULT_FRICTION_COST_USD:.2f}`",
        "",
        "## Summary",
        f"- Recommended `retry`: `{len(retry_df)}` declines",
        f"- Recommended `do_not_retry`: `{len(skip_df)}` declines",
        f"- Aggregate expected retry value of recommended retries: `${retry_df['expected_retry_value_usd'].sum():,.2f}`",
    ]
    if recovered_col in retry_df.columns:
        lines.append(f"- Actual recovered-in-hindsight within recommended retries: `{retry_recovered}`")

    lines.extend(["", "## Top Retry Candidates"])
    for _, row in retry_df.head(sample_size).iterrows():
        lines.append(
            f"- `{row['timestamp']}` `{row['merchant_country']}` `{row['processor_name']}` code `{row['response_code']}` amount `${float(row['amount_usd']):,.2f}` -> `retry` (p=`{row['predicted_recovery_probability']:.3f}`, expected value=`${row['expected_retry_value_usd']:.2f}`)"
        )
        lines.append(f"  - {row['explanation_1']}")
        lines.append(f"  - {row['explanation_2']}")
        lines.append(f"  - {row['explanation_3']}")

    lines.extend(["", "## Top Do-Not-Retry Examples"])
    for _, row in skip_df.head(min(5, sample_size)).iterrows():
        lines.append(
            f"- `{row['timestamp']}` `{row['merchant_country']}` `{row['processor_name']}` code `{row['response_code']}` amount `${float(row['amount_usd']):,.2f}` -> `do_not_retry` (p=`{row['predicted_recovery_probability']:.3f}`, expected value=`${row['expected_retry_value_usd']:.2f}`)"
        )
        lines.append(f"  - {row['explanation_1']}")
        lines.append(f"  - {row['explanation_2']}")
        lines.append(f"  - {row['explanation_3']}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score declined transactions with the Project 3 recovery engine.")
    parser.add_argument("--input-csv", type=str, default="", help="Optional CSV of original declines to score.")
    parser.add_argument("--sample-size", type=int, default=10, help="How many top retry candidates to include in the markdown report.")
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV), help="Path to scored output CSV.")
    parser.add_argument("--output-md", type=str, default=str(DEFAULT_OUTPUT_MD), help="Path to markdown summary report.")
    args = parser.parse_args()

    booster, metadata = load_model_and_metadata()
    if args.input_csv:
        df = pd.read_csv(args.input_csv, engine="python", on_bad_lines="skip")
        source_desc = f"user-provided CSV `{args.input_csv}`"
        threshold = float(metadata["decision_threshold"])
        threshold_basis = "model metadata default"
    else:
        full_df = load_data()
        _, val_df, test_df = temporal_split(full_df)
        df = test_df.copy()
        source_desc = "temporal test split from Project 3 modeling table"
        X_val = prepare_features(val_df, metadata)
        val_probs = booster.predict(X_val, num_iteration=metadata.get("best_iteration"))
        val_amount = pd.to_numeric(val_df["amount_usd"], errors="coerce").fillna(0).to_numpy()
        threshold, _ = optimize_threshold_for_net_value(
            val_df["target_recovered_by_retry"].astype(int).to_numpy(),
            val_probs,
            val_amount,
        )
        threshold_basis = "maximize validation realized_net_value_usd"

    scored = score_batch(df, booster, metadata, threshold)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    scored.to_csv(output_csv, index=False)
    output_md.write_text(
        build_report(scored, args.sample_size, source_desc, threshold, threshold_basis),
        encoding="utf-8",
    )

    print(output_csv)
    print(output_md)


if __name__ == "__main__":
    main()
