# Project 3 Isotonic Calibration Report

## Setup
- Base model: LightGBM
- Calibration method: IsotonicRegression fitted on validation predictions
- Threshold reused for downstream metrics: `0.19`

## Test Metrics Before Calibration
- AUC: `0.732`
- Log loss: `0.350`
- Brier score: `0.111`

## Test Metrics After Calibration
- AUC: `0.720`
- Log loss: `0.358`
- Brier score: `0.110`

## Readout
- Brier delta (after - before): `-0.0016`
- Log-loss delta (after - before): `0.0080`
- Use the calibrated score when merchant-facing probability quality matters more than raw ranking purity.

## Calibration Curve Snapshot
- Bucket `1`: raw mean `0.007` vs event `0.000`; calibrated mean `0.081` vs event `0.081`
- Bucket `2`: raw mean `0.007` vs event `0.000`; calibrated mean `0.221` vs event `0.214`
- Bucket `3`: raw mean `0.014` vs event `0.005`; calibrated mean `0.280` vs event `0.213`
- Bucket `4`: raw mean `0.067` vs event `0.109`; calibrated mean `0.328` vs event `0.235`
- Bucket `5`: raw mean `0.103` vs event `0.108`; calibrated mean `0.424` vs event `0.317`
