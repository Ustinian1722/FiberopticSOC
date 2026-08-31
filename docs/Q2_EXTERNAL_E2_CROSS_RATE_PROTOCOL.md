# Q2 external E2 same-cell cross-rate FBG SOC protocol

Status: **PRE-REGISTERED SECONDARY EXTERNAL TEST; NO ARCHITECTURE SEARCH**

E1 has already been frozen as a negative cross-cell result. E2 is not a rescue of E1 and cannot alter the E1 conclusion.

## Scientific question

Does the pre-declared zero-point-aligned raw surface-FBG response help SOC estimation across discharge rate **when the physical cell and FBG installation are held fixed**?

This separates rate shift from the sensor/cell transfer problem that E1 showed to be nontrivial.

## Data direction fixed from coverage

All four external cells A1/A2/P1/P2 contain eligible constant-current data at 0.2C, 0.5C and 1C. Therefore E2 uses the common ordered stress direction:

- source rates: **0.2C + 0.5C**;
- held-out target rate: **1C**.

The direction is fixed because 1C is the highest common constant-current rate, not because of any 1C model accuracy.

For each physical cell independently:

- train only on that cell's eligible 0.2C and 0.5C full discharge segments;
- test only on the same cell's eligible 1C full discharge segments;
- never include a 1C sample in fitting, normalization, epoch selection, or preprocessing design.

Cells are reported as four independent paired replications, not pooled as interchangeable sensors.

## Labels and predictors

Use the already frozen V2 external SOC reconstruction and the same 52 frozen physical segments as E1.

Predictors:

1. `VI-TCN`: Voltage, Current.
2. `VI-S5rel-TCN`: Voltage, Current, `S5_rel`.

`S5_rel(t) = lambda_S5(t) - lambda_S5(discharge_start)`.

No absolute S5, sign flip, scale alignment, adhesive/cell label, C-rate label, Pt100, absolute time, cumulative current, capacity or SOC history is a predictor.

## Architecture/training freeze

Exactly the same external TCN template as E1:

- hidden width 24;
- three causal residual blocks, dilations 1/2/4;
- 64-sample causal window;
- train stride 4;
- test stride 1;
- AdamW, MSE;
- fixed 20 epochs;
- seed 42;
- same batch size and clipping rule as E1.

Normalization statistics are fit using only the same-cell 0.2C+0.5C source segments.

No model, window, optical preprocessing, seed, epoch or hyperparameter search is permitted from E2 target results.

## Pre-registered E2 support gate

Cell-level metrics are equal weighted across A1/A2/P1/P2. `VI-S5rel-TCN` supports a robust same-cell cross-rate FBG benefit only if **all** conditions hold:

1. mean cell-level MAE is lower than VI-TCN;
2. S5rel wins MAE in at least 3 of 4 cells;
3. mean cell-level RMSE is no worse;
4. mean cell-level Q95-AE is no worse;
5. worst single-cell relative MAE increase is <=10%.

Otherwise the decision is `DOES_NOT_SUPPORT_ROBUST_SAME_CELL_CROSS_RATE_FBG_BENEFIT`.

If the initial seed-42 E2 passes all five gates, a later multi-seed confirmation may be run without changing the design. If it fails, no target-guided rescue is allowed.

## Relation to E1

E1 asks whether S5rel is cell-invariant and found that it is not robustly transferable across physical cells. E2 asks a different, narrower question: whether optical dynamics help rate extrapolation once the sensor/cell identity is fixed. E2 cannot overwrite or soften the E1 negative finding.