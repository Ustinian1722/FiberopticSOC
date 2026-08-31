# Q2 TF-CrossFormer source-side confirmation protocol

Status: **PRE-REGISTERED CONFIRMATION; NO NEW ARCHITECTURE SEARCH**

The final 1C->2C development screen showed that the pre-registered raw-W EO-CrossFormer fails its retention gate, while the matched T/F variant had the best aggregate MAE/RMSE/Q95 but only 3/6 profile wins. The T/F model is therefore not promoted from target-domain evidence alone.

This confirmation asks whether the exact already-screened `EO-CrossFormer-TF` model also shows stable value on **source-side same-rate profile transfer**.

## Data protocol

- rate: 1C only
- six leave-one-profile-out folds: HWFET, LA92, NEDC, NYCC, US06, WLTC
- train on five 1C profiles; validate on the held-out 1C profile
- no 2C trajectory is loaded for model fitting, validation, normalization, or decision
- seed: 42
- fixed 20 epochs
- window: 64
- train stride: 4
- validation stride: 1
- batch size: 256
- optimizer/loss: same repository AdamW/MSE utility
- predictors are leakage-safe; no SOC/discharge-capacity/time predictor

## Models

Exactly two models are allowed:

1. `IUW-TCN`: compact V/I + raw W1/W2 strong baseline.
2. `EO-CrossFormer-TF`: exact 76,753-parameter architecture already used in the final development screen, with V/I electrical branch and physics-decoupled T/F optical branch.

No raw CrossFormer, Transformer variant, Mamba variant, new gate, changed hidden size, changed window, or hyperparameter search is allowed.

## KEEP criteria for EO-CrossFormer-TF

All criteria are required:

1. overall mean delta MAE (`TF - IUW`) < 0;
2. median delta MAE < 0;
3. TF wins MAE in at least 4 of 6 held-out 1C profiles;
4. overall mean RMSE is lower than IUW-TCN;
5. overall mean Q95-AE is no worse than IUW-TCN.

If any criterion fails: **DROP EO-CrossFormer-TF** and stop proposed-architecture development.

If all pass: **KEEP EO-CrossFormer-TF**, freeze the T/F representation and architecture, then perform source-only epoch selection for the frozen model before any new reverse cross-rate result is inspected.

## Interpretation constraint

A KEEP result does not imply that T/F contains additional information beyond W1/W2. T/F is an invertible physical coordinate of the same two optical degrees of freedom. A performance difference is interpreted as representation geometry/inductive-bias interaction with cross-modal fusion.

A DROP result means the small aggregate target-development gain was not sufficiently stable to justify the additional complexity.