# Q2 uncertainty-quantification positioning

Date: 2026-08-31
Status: required supporting contribution

## Literature boundary

Battery SOC interval estimation is no longer novel by itself. Recent directly relevant work includes:

- Energy 294 (2024) 130834: TCN-GRU-Attention with quantile-regression interval estimation.
- Journal of Power Sources 666 (2026) 239123: FeatureFormer plus Conformalized Quantile Regression (CQR), emphasizing distribution-free calibration, coverage, interval width and calibration error.

Therefore this paper will **not** claim novelty from quantile regression or conformal prediction alone.

## Role in this manuscript

The point estimator remains the main method contribution: electrical–thermomechanical temporal fusion for dual-FBG assisted SOC estimation.

UQ is the required reliability layer:

1. freeze the point estimator first;
2. construct prediction intervals using source-side calibration only;
3. quantify how calibration behaves as the evaluation protocol moves from interpolation to increasingly difficult operating-condition shift.

The UQ story is therefore:

> calibrated uncertainty for a multimodal SOC estimator under a hierarchy of condition shifts,

not:

> a new conformal-prediction algorithm.

## UQ-U1: symmetric split conformal baseline

For calibration residuals

`r_i = |y_i - yhat_i|`,

use the finite-sample conformal quantile

`q = Quantile_higher(r, ceil((n_cal + 1)*(1-alpha))/n_cal)`.

The test interval is

`[yhat - q, yhat + q]`, clipped to the physical SOC range `[0,1]` only when reporting a physically bounded interval.

Evaluate 90% and 95% nominal coverage.

## UQ-U2: CQR

Train lower/upper quantile heads on source-side training data, then conformalize the quantile interval using an independent source-side calibration subset.

CQR survives into the paper only if it provides meaningfully better adaptivity (lower MPIW / interval score) without meaningful undercoverage relative to UQ-U1.

## Required metrics

- empirical coverage / PICP
- coverage error = PICP - nominal coverage
- MPIW
- PINAW
- mean interval score / Winkler-style interval score
- conditional coverage by SOC region
- conditional coverage by load-intensity region

Point-estimation metrics and UQ metrics must not be conflated.

## Shift-aware reporting

Report UQ separately under:

- T1 mixed-condition interpolation
- T2 same-rate unseen-profile
- T3 cross-rate
- T4 cross-rate + unseen-profile

Standard split-conformal finite-sample coverage relies on exchangeability. T3/T4 deliberately challenge that assumption, so a nominal 95% interval must **not** be described as guaranteed 95% coverage under target-domain shift. Instead report the empirical coverage degradation honestly.

This is scientifically useful: the paper can show whether thermo-mechanical fusion improves not only point accuracy but also the stability of calibrated uncertainty as operating conditions move out of distribution.

## Optional UQ-U3 only if needed

If UQ-U1/CQR shows severe and systematic undercoverage under T3/T4, add one lightweight source-only adaptive calibration control, for example calibration stratified by observable load-intensity and/or predicted-SOC regime. Test strata must be selected from observable inputs or model predictions, never from target SOC labels.

Do not add UQ-U3 unless baseline results justify it.

## Leakage boundary

Forbidden for UQ calibration/model selection:

- final test SOC labels
- test residuals
- test-derived interval scaling
- test-derived calibration strata thresholds
- any use of `dis_cap` or absolute `Time_s`

All UQ hyperparameters and calibration quantiles must be frozen from the source side before final test evaluation.
