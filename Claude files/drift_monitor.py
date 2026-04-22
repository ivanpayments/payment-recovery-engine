from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from train_project3_lightgbm import load_data, load_feature_policy, select_features, temporal_split


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "project3_drift_report.md"
CSV_OUT = ROOT / "project3_drift_metrics.csv"


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return float("nan")
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    eps = 1e-6
    expected_counts, _ = np.histogram(expected, bins=quantiles)
    actual_counts, _ = np.histogram(actual, bins=quantiles)
    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), eps, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), eps, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def top_categories(series: pd.Series, limit: int = 10) -> dict[str, float]:
    values = series.fillna("MISSING").astype(str)
    shares = values.value_counts(normalize=True).head(limit)
    return {str(k): float(v) for k, v in shares.items()}


def jensen_shannon_from_top_counts(base: dict[str, float], ref: dict[str, float]) -> float:
    universe = sorted(set(base) | set(ref))
    eps = 1e-6
    p = np.array([base.get(k, eps) for k in universe], dtype=float)
    q = np.array([ref.get(k, eps) for k in universe], dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def build_metrics(baseline_df: pd.DataFrame, reference_df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in features:
        if col not in reference_df.columns:
            rows.append({"feature": col, "feature_type": "missing", "psi": np.nan, "ks_stat": np.nan, "p_value": np.nan, "js_divergence": np.nan})
            continue
        if pd.api.types.is_numeric_dtype(baseline_df[col]) or pd.api.types.is_bool_dtype(baseline_df[col]):
            if pd.api.types.is_bool_dtype(baseline_df[col]) or pd.api.types.is_bool_dtype(reference_df[col]):
                base = baseline_df[col].fillna(False).astype(int)
                ref = reference_df[col].fillna(False).astype(int)
            else:
                base = pd.to_numeric(baseline_df[col], errors="coerce")
                ref = pd.to_numeric(reference_df[col], errors="coerce")
            ks_stat, p_value = ks_2samp(base.dropna(), ref.dropna()) if not base.dropna().empty and not ref.dropna().empty else (np.nan, np.nan)
            rows.append(
                {
                    "feature": col,
                    "feature_type": "numeric",
                    "psi": psi(base, ref),
                    "ks_stat": float(ks_stat) if pd.notna(ks_stat) else np.nan,
                    "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                    "js_divergence": np.nan,
                }
            )
        else:
            base_top = top_categories(baseline_df[col])
            ref_top = top_categories(reference_df[col])
            rows.append(
                {
                    "feature": col,
                    "feature_type": "categorical",
                    "psi": np.nan,
                    "ks_stat": np.nan,
                    "p_value": np.nan,
                    "js_divergence": jensen_shannon_from_top_counts(base_top, ref_top),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute drift metrics against a reference CSV.")
    parser.add_argument("--reference-csv", required=False, default=str(ROOT / "project3_modeling_table.csv"))
    args = parser.parse_args()

    baseline_df = load_data()
    train_df, _, _ = temporal_split(baseline_df)
    reference_df = pd.read_csv(args.reference_csv, engine="python", on_bad_lines="skip")
    if "timestamp" in reference_df.columns:
        reference_df["timestamp"] = pd.to_datetime(reference_df["timestamp"], errors="coerce")
        reference_df = reference_df.sort_values("timestamp").reset_index(drop=True)
        reference_df["event_hour"] = reference_df["timestamp"].dt.hour.fillna(0).astype(int)
        reference_df["event_dayofweek"] = reference_df["timestamp"].dt.dayofweek.fillna(0).astype(int)
        reference_df["event_month"] = reference_df["timestamp"].dt.month.fillna(0).astype(int)
        reference_df["is_weekend"] = reference_df["event_dayofweek"].isin([5, 6])

    policy = load_feature_policy()
    features = select_features(train_df, policy)
    metric_df = build_metrics(train_df, reference_df, features).sort_values(["feature_type", "feature"])
    metric_df.to_csv(CSV_OUT, index=False)

    top_psi = metric_df[metric_df["feature_type"] == "numeric"].sort_values("psi", ascending=False).head(10)
    top_js = metric_df[metric_df["feature_type"] == "categorical"].sort_values("js_divergence", ascending=False).head(10)
    lines = [
        "# Project 3 Drift Report",
        "",
        f"- Baseline split: training window from `{ROOT / 'project3_modeling_table.csv'}`",
        f"- Reference CSV: `{args.reference_csv}`",
        f"- Features evaluated: `{len(features)}`",
        "",
        "## Highest Numeric Drift (PSI)",
    ]
    for _, row in top_psi.iterrows():
        lines.append(f"- `{row['feature']}`: PSI `{row['psi']:.4f}`, KS `{row['ks_stat']:.4f}`, p-value `{row['p_value']:.4g}`")
    lines.extend(["", "## Highest Categorical Drift (JS Divergence)"])
    for _, row in top_js.iterrows():
        lines.append(f"- `{row['feature']}`: JS divergence `{row['js_divergence']:.4f}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "- PSI above ~0.2 is a useful yellow flag for numeric features in this synthetic setup.",
            "- Larger JS divergence on categorical features indicates a meaningful shift in the mix of top categories.",
            "- This script is meant for offline monitoring and model-review workflows, not inline inference.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)
    print(CSV_OUT)


if __name__ == "__main__":
    main()
