# Frozen epoch protocol for the final benchmark

This file records the provenance of `config/frozen_epoch_plan.csv` and fixes the model-selection boundary for the final SOC benchmark.

## Purpose

The earlier six-epoch experiments are retained only as short-budget robustness audits. They are not the final model comparison because optimization maturity differed materially across VI, raw-FBG, and whitened-FBG representations.

## Development-only epoch selection

- Development seed: **42**.
- Final statistical seeds: **0, 1, 2, 3, 4**; none of these seeds were used to select training epochs.
- Main split: for each direction (`1C_to_2C` or `2C_to_1C`) and each final held-out drive profile, the five source-rate profiles are the only data available to model selection.
- Inner validation: five leave-one-source-profile-out folds.
- Candidate models: `VI`, `VI+W`, `VI+W-white`.
- Maximum inner-training budget: 60 epochs.
- Minimum epochs before stopping: 12.
- Early-stopping patience: 8 epochs.
- Minimum MAE improvement: 5e-5.
- Frozen epoch for each `(direction, held_out_profile, model)` is the median of the five fold-best epochs.
- No final held-out-profile SOC labels are used to choose epochs.

The source workflow was GitHub Actions run **33331374593** (`Train-only profile-CV early stopping audit`). Its aggregate artifact generated the exact 36-row plan committed as `config/frozen_epoch_plan.csv`.

## Sanity checks before freezing

- Expected rows: 2 directions × 6 held-out profiles × 3 models = **36**.
- Observed rows: **36**.
- Duplicate `(direction, held_out_profile, model)` keys: **0**.
- Frozen selected-epoch range: **24–59**.
- Frozen selected epochs equal to the 60-epoch ceiling: **0**.

Some individual inner folds, especially optical models in the `2C_to_1C` direction, reached the 60-epoch audit ceiling. This is retained as an optimization-stability caveat. The frozen budget nevertheless uses the profile-CV median fold-best epoch rather than the maximum observed epoch, and no final selected epoch equals 60.

## Final test boundary

After this file and `frozen_epoch_plan.csv` are fixed, the final benchmark:

1. retrains each expert from scratch with seeds 0–4 using only its frozen epoch;
2. recomputes normalization, FBG whitening, current support envelopes, and NestedCal99 thresholds from the five source profiles only;
3. evaluates the held-out target-rate profile once;
4. never uses `SOC`, `dis_cap`, or absolute `Time_s` as predictors;
5. never uses target SOC labels to select epochs, whitening parameters, support thresholds, or gates.

Any later experiment that changes the epoch plan must be reported as a new protocol rather than silently replacing the frozen final benchmark.
