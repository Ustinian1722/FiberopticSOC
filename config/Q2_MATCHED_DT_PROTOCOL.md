# Q2 matched causal delta-t protocol

Status: **PRE-REGISTERED; NOT YET RELEASED**

This gate is evaluated only after the final source-only epoch plan for the frozen raw-W `IUW-TCN` has been committed. It cannot use any opposite-rate target metric.

## Scientific question

Does the causal sampling-interval feature `log(1 + delta_t)` provide a sufficiently stable source-domain validation gain to justify adding it to the otherwise frozen raw-W point estimator?

`delta_t[k] = Time_s[k] - Time_s[k-1]`, with the first row of each trajectory set to zero. Absolute `Time_s` is forbidden. Negative increments are invalid. Delta-t normalization statistics are fitted on the five source-training profiles only.

## Matched arms

For each rate/profile fold:

- baseline: frozen `IUW-TCN`, inputs `[I, U, W1, W2]`;
- candidate: the same TCN depth, width and prediction head, inputs `[I, U, W1, W2, log1p(delta_t)]`.

The candidate therefore differs only by the additional causal elapsed-time input coefficients in the first input projection. No ETMF, T/F coordinate, whitening, gating or other architecture change is allowed in this gate.

## Source-only folds

Evaluate all 12 same-rate profile-level LOPO folds:

- six folds at 1C: train on five 1C profiles and validate on the sixth 1C profile;
- six folds at 2C: train on five 2C profiles and validate on the sixth 2C profile.

For validation profile `P` at a given source rate, use the final frozen epoch selected for the corresponding T4 outer split whose held-out profile is `P`. That epoch was selected from the other five same-rate source profiles only, so the source-side validation profile used here did not participate in its epoch selection.

Seed is development seed42. Window64, train stride4, validation stride1, batch256 and AdamW learning rate 1e-3 remain unchanged.

## Keep / drop rule

Let `delta_MAE = MAE_dt - MAE_no_dt` on each of the 12 source-validation folds. **Keep delta-t only if every condition below is satisfied:**

1. overall mean `delta_MAE < 0`;
2. overall median `delta_MAE < 0`;
3. delta-t wins MAE on at least 8 of 12 folds;
4. mean `delta_MAE < 0` separately for the six 1C folds and the six 2C folds;
5. overall mean `delta_Q95_AE <= 0`.

If any condition fails, delta-t is dropped and the simpler frozen `[I,U,W1,W2]` estimator remains the point-model mainline. There is no rescue based on target-domain performance or a favorable subset of profiles.

The decision artifact must record all five booleans, fold-level paired deltas, rate-level aggregates and the final `KEEP`/`DROP` result.

## Release condition

The workflow must not be triggered until `config/q2_frozen_epoch_plan.csv` exists and the raw-W epoch extension has no unresolved ceiling-bound source-validation folds. The matched-delta-t workflow is released only by committing its dedicated plan/latch file after that condition is met.
