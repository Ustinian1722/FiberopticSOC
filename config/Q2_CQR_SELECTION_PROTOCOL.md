# Q2 CQR training-side selection protocol

Status: preregistered before final CQR selection results are observed.

## Purpose
Decide whether conformalized quantile regression (CQR) adds enough value to the frozen Q2 point-estimator feature set to remain in the paper. This gate must not use the formal T1 test blocks or any T4 target labels.

## Frozen backbone identity
- Point-estimator family: IUW-TCN (raw W1/W2 representation).
- The optical representation and TCN depth/width are already frozen by the representation-aware equal-budget screen.
- The only feature-mode uncertainty allowed at CQR runtime is whether the previously preregistered causal log1p(delta-t) gate was KEEP or DROP.
- No ETMF, TF, whitening, or alternate architecture may re-enter through the UQ stage.

## Data boundary
1. Construct the already frozen T1 blocked split with the existing guard-gap implementation.
2. The formal T1 validation, calibration, and test buckets are not used for CQR keep/drop selection.
3. Take only the existing T1 **train** segments. These segments are already separated from the formal T1 held-out buckets by `window-1` guard rows.
4. For each physical trajectory, sort its T1-train segments by original segment start and assign whole segments to inner `fit`, `validation`, `calibration`, and `selection` roles with a deterministic phase-rotated pattern. No SOC label, prediction, residual, or uncertainty statistic may influence this assignment.
5. Normalization is fitted on the inner fit set only.

## Candidate and baseline
- Candidate: frozen raw-W TCN encoder with a four-quantile head for q={0.025, 0.05, 0.95, 0.975}, trained with pinball loss plus a soft crossing penalty, then conformalized on the inner calibration set.
- Simple UQ baseline: the same frozen point-estimator feature mode with symmetric split conformal intervals obtained from absolute calibration residuals.
- Both methods are evaluated on the same untouched inner selection segments.

## Interval levels
- 90% interval (alpha=0.10)
- 95% interval (alpha=0.05)

## CQR KEEP rule
CQR is retained only if all of the following hold on the training-side selection set:
1. PICP at 90% is at least 0.87.
2. PICP at 95% is at least 0.92.
3. CQR mean interval score is no worse than residual split conformal at both nominal levels.
4. The average relative mean-interval-score improvement of CQR over residual split conformal across the two nominal levels is at least 2%.
5. CQR mean interval width is not more than 10% wider than residual split conformal at either nominal level.

If any criterion fails, the decision is DROP. A DROP result is a valid negative ablation and the formal T1 test must not be inspected to rescue CQR.

## Post-selection rule
Only after the KEEP/DROP decision is frozen may a final UQ model be fit using the normal frozen T1 train/validation/calibration protocol and evaluated on the formal T1 test blocks. If CQR is DROP, no formal-test CQR result may be used to reverse the decision.
