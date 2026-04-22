from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from train_project3_lightgbm import (
    MODEL_META,
    MODEL_TXT,
    TARGET,
    load_data,
    rule_baseline_probs,
    select_features,
    coerce_feature_types,
    load_feature_policy,
    temporal_split,
)


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "project3_decision_policy_evaluation.md"
DECISIONS_CSV = ROOT / "project3_decision_policy_sample.csv"
SUMMARY_CSV = ROOT / "project3_decision_policy_summary.csv"


DEFAULT_RETRY_COST_USD = 0.12
DEFAULT_MARGIN_RATE = 0.35
DEFAULT_FRICTION_COST_USD = 0.03


def load_model_and_metadata():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit(
            "LightGBM is not installed in this runtime. Install it before running decision evaluation."
        ) from exc

    if not MODEL_TXT.exists() or not MODEL_META.exists():
        raise SystemExit(
            "Model artifacts are missing. Run train_project3_lightgbm.py first."
        )

    booster = lgb.Booster(model_file=str(MODEL_TXT))
    metadata = json.loads(MODEL_META.read_text(encoding="utf-8"))
    return booster, metadata


def prepare_features_for_scoring(df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
    feature_cols = metadata["feature_columns"]
    X_df, _, _, categorical_cols = coerce_feature_types(df, feature_cols)
    for col in categorical_cols:
        cats = sorted(X_df[col].fillna("MISSING").astype(str).unique().tolist())
        X_df[col] = pd.Categorical(X_df[col], categories=cats)
    return X_df


def expected_retry_value(prob: np.ndarray, amount_usd: np.ndarray) -> np.ndarray:
    recovered_margin = prob * amount_usd * DEFAULT_MARGIN_RATE
    return recovered_margin - DEFAULT_RETRY_COST_USD - DEFAULT_FRICTION_COST_USD


def decision_from_prob_and_value(prob: np.ndarray, exp_value: np.ndarray, threshold: float) -> np.ndarray:
    return np.where((prob >= threshold) & (exp_value > 0), "retry", "do_not_retry")


def summarize_policy(
    df: pd.DataFrame,
    probs: np.ndarray,
    decisions: np.ndarray,
    policy_name: str,
) -> Dict[str, float | str]:
    target = df[TARGET].astype(int).to_numpy()
    amount = pd.to_numeric(df["amount_usd"], errors="coerce").fillna(0).to_numpy()
    retry_mask = decisions == "retry"
    recovered_mask = retry_mask & (target == 1)
    wasted_retry_mask = retry_mask & (target == 0)

    retry_count = int(retry_mask.sum())
    recovered_count = int(recovered_mask.sum())
    wasted_count = int(wasted_retry_mask.sum())
    recovered_volume = float(amount[recovered_mask].sum())
    wasted_retry_cost = float(wasted_count * DEFAULT_RETRY_COST_USD)
    total_friction_cost = float(retry_count * DEFAULT_FRICTION_COST_USD)
    gross_margin_recovered = float((amount[recovered_mask] * DEFAULT_MARGIN_RATE).sum())
    realized_net_value = gross_margin_recovered - wasted_retry_cost - total_friction_cost
    avg_prob = float(probs[retry_mask].mean()) if retry_count else 0.0

    return {
        "policy": policy_name,
        "rows": len(df),
        "retry_count": retry_count,
        "retry_rate": retry_count / len(df),
        "recovered_count": recovered_count,
        "recovered_rate": recovered_count / len(df),
        "recovered_volume_usd": recovered_volume,
        "gross_margin_recovered_usd": gross_margin_recovered,
        "wasted_retry_count": wasted_count,
        "wasted_retry_cost_usd": wasted_retry_cost,
        "friction_cost_usd": total_friction_cost,
        "realized_net_value_usd": realized_net_value,
        "average_retry_probability": avg_prob,
    }


def optimize_threshold_for_net_value(
    target: np.ndarray,
    probs: np.ndarray,
    amount_usd: np.ndarray,
) -> tuple[float, float]:
    best_threshold = 0.19
    best_net_value = -float("inf")
    expected_values = expected_retry_value(probs, amount_usd)
    temp_df = pd.DataFrame({TARGET: target, "amount_usd": amount_usd})
    for threshold in np.linspace(0.05, 0.50, 46):
        decisions = decision_from_prob_and_value(probs, expected_values, float(threshold))
        summary = summarize_policy(temp_df, probs, decisions, "candidate")
        if summary["realized_net_value_usd"] > best_net_value:
            best_net_value = float(summary["realized_net_value_usd"])
            best_threshold = float(threshold)
    return best_threshold, best_net_value


def build_report(summary_df: pd.DataFrame, sample_df: pd.DataFrame, threshold: float, threshold_basis: str) -> str:
    ml = summary_df.loc[summary_df["policy"] == "ml_policy"].iloc[0]
    rules = summary_df.loc[summary_df["policy"] == "rules_policy"].iloc[0]
    delta_net = ml["realized_net_value_usd"] - rules["realized_net_value_usd"]
    delta_recovered = ml["recovered_count"] - rules["recovered_count"]
    delta_retry = ml["retry_count"] - rules["retry_count"]

    lines = [
        "# Project 3 Decision Policy Evaluation",
        "",
        "## Setup",
        "- Holdout: temporal test split from the Project 3 modeling table",
        f"- Retry threshold used for ML policy: `{threshold:.2f}`",
        f"- Threshold basis: `{threshold_basis}`",
        f"- Retry cost assumption: `${DEFAULT_RETRY_COST_USD:.2f}` per retry",
        f"- Margin rate assumption: `{DEFAULT_MARGIN_RATE:.0%}` of amount_usd",
        f"- Friction cost assumption: `${DEFAULT_FRICTION_COST_USD:.2f}` per retry",
        "",
        "## Policy Comparison",
        f"- ML policy retries `{int(ml['retry_count'])}` declines and recovers `{int(ml['recovered_count'])}`",
        f"- Rules policy retries `{int(rules['retry_count'])}` declines and recovers `{int(rules['recovered_count'])}`",
        f"- Net value delta (ML - rules): `${delta_net:,.2f}`",
        f"- Recovered decline delta (ML - rules): `{int(delta_recovered)}`",
        f"- Retry volume delta (ML - rules): `{int(delta_retry)}`",
        "",
        "## ML Policy",
        f"- Retry rate: `{ml['retry_rate']:.2%}`",
        f"- Recovered volume: `${ml['recovered_volume_usd']:,.2f}`",
        f"- Gross margin recovered: `${ml['gross_margin_recovered_usd']:,.2f}`",
        f"- Wasted retry cost: `${ml['wasted_retry_cost_usd']:,.2f}`",
        f"- Friction cost: `${ml['friction_cost_usd']:,.2f}`",
        f"- Realized net value: `${ml['realized_net_value_usd']:,.2f}`",
        "",
        "## Rules Policy",
        f"- Retry rate: `{rules['retry_rate']:.2%}`",
        f"- Recovered volume: `${rules['recovered_volume_usd']:,.2f}`",
        f"- Gross margin recovered: `${rules['gross_margin_recovered_usd']:,.2f}`",
        f"- Wasted retry cost: `${rules['wasted_retry_cost_usd']:,.2f}`",
        f"- Friction cost: `${rules['friction_cost_usd']:,.2f}`",
        f"- Realized net value: `${rules['realized_net_value_usd']:,.2f}`",
        "",
        "## Interpretation",
        "- This layer turns model probability into an operational recommendation by combining recoverability with simple economic assumptions.",
        "- These numbers are synthetic and depend on the chosen cost assumptions, but they make the product story much more concrete than model metrics alone.",
        "- The next refinement should be configurable business thresholds and SHAP-backed explanations for why a specific decline is recommended for retry.",
        "",
        "## Sample High-Confidence Decisions",
    ]

    preview = sample_df.head(12)
    for _, row in preview.iterrows():
        lines.append(
            f"- `{row['timestamp']}` `{row['merchant_country']}` `{row['processor_name']}` code `{row['response_code']}` amount `${row['amount_usd']:.2f}` -> `{row['ml_decision']}` (p=`{row['ml_probability']:.3f}`, expected value=`${row['ml_expected_value_usd']:.2f}`)"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    booster, metadata = load_model_and_metadata()
    policy = load_feature_policy()

    df = load_data()
    _, val_df, test_df = temporal_split(df)
    feature_cols = select_features(test_df, policy)
    X_test_df = prepare_features_for_scoring(test_df, metadata)

    ml_probs = booster.predict(X_test_df, num_iteration=metadata.get("best_iteration"))
    amount = pd.to_numeric(test_df["amount_usd"], errors="coerce").fillna(0).to_numpy()

    X_val_df = prepare_features_for_scoring(val_df, metadata)
    val_probs = booster.predict(X_val_df, num_iteration=metadata.get("best_iteration"))
    val_amount = pd.to_numeric(val_df["amount_usd"], errors="coerce").fillna(0).to_numpy()
    threshold, _ = optimize_threshold_for_net_value(
        val_df[TARGET].astype(int).to_numpy(),
        val_probs,
        val_amount,
    )

    ml_expected_value = expected_retry_value(ml_probs, amount)
    ml_decisions = decision_from_prob_and_value(ml_probs, ml_expected_value, threshold)

    rule_probs = rule_baseline_probs(test_df)
    rule_expected_value = expected_retry_value(rule_probs, amount)
    rule_decisions = decision_from_prob_and_value(rule_probs, rule_expected_value, threshold)

    summary_records = [
        summarize_policy(test_df, ml_probs, ml_decisions, "ml_policy"),
        summarize_policy(test_df, rule_probs, rule_decisions, "rules_policy"),
    ]
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    sample_df = test_df[
        ["timestamp", "merchant_country", "processor_name", "response_code", "amount_usd", TARGET]
    ].copy()
    sample_df["ml_probability"] = ml_probs
    sample_df["ml_expected_value_usd"] = ml_expected_value
    sample_df["ml_decision"] = ml_decisions
    sample_df["rules_probability"] = rule_probs
    sample_df["rules_expected_value_usd"] = rule_expected_value
    sample_df["rules_decision"] = rule_decisions
    sample_df = sample_df.sort_values(["ml_expected_value_usd", "ml_probability"], ascending=False)
    sample_df.head(200).to_csv(DECISIONS_CSV, index=False)

    REPORT.write_text(
        build_report(summary_df, sample_df, threshold, "maximize validation realized_net_value_usd"),
        encoding="utf-8",
    )
    print(REPORT)
    print(SUMMARY_CSV)
    print(DECISIONS_CSV)


if __name__ == "__main__":
    main()
