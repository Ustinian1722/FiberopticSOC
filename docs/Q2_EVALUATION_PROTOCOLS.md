# Q2 evaluation protocols

Status: FROZEN DRAFT FOR IMPLEMENTATION

This file defines the intended hierarchy of evaluation tasks for the Q2 manuscript. The protocols deliberately separate interpolation accuracy from true operating-condition generalization. No random-row split is permitted.

## Common rules

- Inputs are causal observed measurements only.
- Forbidden predictors: `SOC`, `dis_cap`, absolute `Time_s`, terminal/full-cycle capacity, future statistics.
- Normalization is fitted on the training partition only.
- Sequence windows may not cross a split boundary.
- Whenever different partitions touch the same source trajectory, remove a guard region of at least `window-1` rows on each relevant boundary so overlapping raw observations cannot appear in different partitions.
- Model/hyperparameter/epoch selection uses training-side validation only.
- Final test labels are used only for reporting metrics.

## T1 — Mixed-condition interpolation benchmark

Purpose: provide the conventional headline-accuracy experiment while avoiding random-window leakage.

For each of the 12 trajectories independently:

1. Divide the trajectory into a fixed number of contiguous row blocks (default 12 blocks; implementation may use near-equal blocks).
2. Assign blocks to train/validation/test using one pre-registered repeating pattern so every partition samples multiple SOC regions rather than only the beginning/end of discharge.
3. Construct causal windows only inside each block; windows never cross a block boundary.
4. Apply a guard of at least `window-1` observations at boundaries between blocks assigned to different partitions.
5. Pool train blocks from all six profiles and both rates; likewise pool validation and test blocks.

This is explicitly an **interpolation benchmark**, not an unseen-condition generalization claim.

Headline metrics: MAE, RMSE, R2, MaxAE, Q95-AE, per-SOC-bin MAE/RMSE.

## T2 — Same-rate unseen-profile LOPO

Purpose: test driving-profile generalization without changing nominal rate.

For each rate separately (1C and 2C):

- choose one profile as final test;
- use the other five profiles at the same rate as source data;
- source-only profile-level validation/epoch selection;
- repeat for all six held-out profiles.

Total: 12 final profile-level tests.

## T3 — Cross-rate generalization

Purpose: isolate discharge-rate shift while preserving profile coverage.

Two directions:

- all six 1C profiles -> corresponding six 2C profiles;
- all six 2C profiles -> corresponding six 1C profiles.

Model selection/calibration uses source-rate data only. Results must be reported per profile and aggregated across profiles; pooling all target rows without per-profile reporting is insufficient.

## T4 — Cross-rate + unseen-profile

Purpose: hardest generalization experiment.

For each held-out profile:

- 1C -> 2C: train/validate using the other five 1C profiles; test only the held-out profile at 2C.
- 2C -> 1C: train/validate using the other five 2C profiles; test only the held-out profile at 1C.

Repeat for all six profiles and both directions: 12 strict splits.

T4 is a robustness/extrapolation experiment and should not be numerically compared directly with papers reporting conventional mixed-condition interpolation accuracy.

## Source-only epoch selection

For T2-T4, model-specific epoch budgets are selected from source profiles only. Preferred protocol:

- profile-level inner CV on the source profiles;
- early stopping within each fold;
- frozen robust summary (e.g. median fold-best epoch) used for final retraining;
- final target trajectory is never involved in epoch selection.

Development seed42 may be used to select/freeze protocol choices. Publication statistics use independent final seeds (target 3-5 seeds, preferably 5 for T4).

## Feature ablation matrix

Use the same split and training protocol for:

- VI
- IUT
- IUF
- IUTF
- IUW (I/U/W1/W2)

W and TF are alternative representations of the same dual-FBG degrees of freedom and must not be described as independent modalities.

## Fusion ablation matrix

Minimum:

- single-stream IUTF TCN
- heterogeneous direct electrical-TCN + thermo-mechanical temporal encoder
- ETMF adaptive cross-gated fusion

Parameter counts and training budgets must be reported.

## UQ calibration protocol

UQ is required but is applied only after the point estimator is frozen.

### T1
Reserve calibration blocks from the training-side/validation allocation. Calibration windows obey the same block/guard rules. Test blocks are never used to calibrate interval width.

### T2-T4
Use only source-domain profiles for calibration. A source profile/fold can act as a pseudo-held-out calibration domain; the final target profile/rate remains untouched until evaluation.

Primary interval levels: 90% and 95%.

Primary UQ metrics:

- empirical coverage / PICP
- coverage error from nominal
- MPIW and PINAW
- mean interval score
- conditional coverage by SOC bin
- conditional coverage by load-intensity bin

UQ-U1 is symmetric split conformal using absolute residuals. UQ-U2 is CQR and survives only if it improves interval adaptivity/width or interval score without meaningful undercoverage.

## Noise robustness

Evaluate the frozen point model under predefined sensor perturbations. Noise levels must be chosen before inspecting final-test labels. Compare at least:

- clean
- mild
- moderate
- severe optical/thermo-mechanical perturbation

Raw-W and T/F perturbations must be described carefully because they are different coordinate representations of the same dual-FBG sensing system.

## Reporting hierarchy

The paper should report results in this order:

1. T1 headline interpolation accuracy.
2. Feature and fusion ablations.
3. T2 unseen-profile generalization.
4. T3 cross-rate generalization.
5. T4 cross-rate + unseen-profile robustness.
6. Noise robustness and error stratification.
7. UQ calibration and interval efficiency under T1-T4.

This ordering intentionally matches the taste of a strong Q2 battery-SOC paper: attractive conventional accuracy first, then progressively harder evidence rather than presenting only the hardest extrapolation task.
