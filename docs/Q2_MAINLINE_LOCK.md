# Q2 publication mainline lock — evidence-frozen version

Date: 2026-08-31
Status: ACTIVE / DESIGN FROZEN BEFORE FORMAL T4

This document supersedes the earlier ETMF-centric Q2 plan. The final publication mainline follows the evidence obtained under the registered development protocol rather than preserving a more complex model because of prior implementation effort.

## 1. Final paper target

The paper targets a solid SCI Q2 battery-SOC contribution built around a simple but strongly validated result:

> Direct dual-FBG wavelength observations can provide useful internal-state information for SOC estimation under operating-condition shift, but the benefit is most reliable when the raw optical coordinates are retained and modeled by a compact causal temporal network rather than by forced thermo-mechanical decoupling, whitening, adaptive fusion, or elapsed-time augmentation.

The core story is therefore **representation choice + leakage-safe cross-condition validation + compact temporal modeling + calibrated reliability**, not architectural complexity.

## 2. Dataset and frozen predictors

Core dataset:
- SiC-18 implanted dual-FBG lithium-ion battery dataset.
- Six dynamic profiles × two rates (1C/2C), 12 discharge trajectories.

Final point-estimator predictors:
- Voltage_V
- Current_A
- Wavelength_1
- Wavelength_2

Forbidden estimator inputs remain:
- SOC or any SOC-derived quantity
- dis_cap
- absolute Time_s
- terminal/test capacity information
- test-set normalization statistics
- future-window statistics

All normalization is fitted on the allowed source/training data only.

## 3. Final point model

Frozen model identity: **IUW-TCN**.

Implementation identity:
- raw W1/W2 optical coordinates;
- V/I + W1/W2 input;
- compact PairTCN with causal left-padded convolutions;
- 64-sample causal window;
- 11,545 trainable parameters in the representation-aware screen configuration.

The final architecture is intentionally compact. No Transformer, Mamba, dual-branch ETMF fusion, whitening transform, or explicit delta-t channel is part of the publication point model.

## 4. Why raw W1/W2 replaced the ETMF/decoupled mainline

A representation-aware equal-training-budget development screen compared eight raw/decoupled/whitened representations and simple/ETMF architectures under the same split, seed, epoch budget, window, stride, optimizer family, and evaluation metrics.

The frozen winner was IUW-TCN.

Relative to the closest complex candidate ETMF-TF:
- mean MAE was about 13.3% lower;
- mean RMSE was about 17.9% lower;
- mean Q95 absolute error was about 17.2% lower;
- parameter count was about 0.47× as large;
- IUW-TCN beat ETMF-TF on 5 of the 6 held-out development profiles.

Whitened raw-W was also rejected as the mainline because its average MAE was about 22.6% worse than raw-W and its cross-profile variability was substantially larger, despite improved numerical conditioning.

Consequences:
- **IUW-TCN/raw-W = final model**.
- **ETMF-TF = negative complexity/fusion ablation**.
- **decoupled T/F = representation ablation**.
- **whitening = conditioning/representation ablation**.

The paper must not claim that the physics-derived T/F coordinate is intrinsically superior. The actual evidence supports the opposite design choice for this SOC task.

## 5. Interpretation of the FBG channels

W1/W2 and the derived thermo-mechanical T/F coordinates are two views generated from the same dual-FBG sensing degrees of freedom. T/F therefore does not create new information; it changes feature geometry according to a sensitivity-matrix model.

The empirical result is that the compact TCN learns more robust cross-profile SOC features from the observed wavelength coordinates directly. The decoupling equations remain useful for sensor interpretation and negative ablation, but they are not the estimator's preferred coordinate system.

This distinction should be maintained throughout the manuscript to avoid over-claiming physical identifiability from the derived coordinate.

## 6. Frozen evaluation hierarchy

T1. **Blocked mixed-condition interpolation**
- deterministic contiguous blocks;
- train/validation/calibration/test separation;
- `window-1` guard gaps across set boundaries;
- purpose: conventional within-dataset accuracy and UQ evaluation.

T2. **Same-rate unseen profile**
- five complete profiles train;
- sixth complete profile test;
- repeated separately at 1C and 2C.

T3. **Cross-rate with seen profile identities**
- all six source-rate profiles train;
- opposite-rate trajectories test.

T4. **Cross-rate + unseen profile**
- five source-rate profiles train;
- held-out profile identity absent from source training;
- corresponding opposite-rate profile is test;
- 2 directions × 6 held-out profiles = 12 strict splits.

T4 is the formal cross-condition generalization benchmark and receives the final seeds 0–4 replication only after all design choices below are frozen.

## 7. Source-only epoch freeze

Epoch selection for formal T4 is source-only. No opposite-rate target metric is allowed to select epochs.

A previous 60-epoch source-only inner-profile audit was reused. Only the 21/60 inner folds that reached the original ceiling were extended to 100 epochs. All 21 resolved before the new ceiling, so no further extension was required.

Frozen epochs:

| Direction | HWFET | LA92 | NEDC | NYCC | US06 | WLTC |
|---|---:|---:|---:|---:|---:|---:|
| 1C→2C | 29 | 40 | 30 | 32 | 42 | 36 |
| 2C→1C | 58 | 56 | 44 | 65 | 51 | 62 |

The plan is stored in `config/q2_frozen_epoch_plan.csv` and explicitly marks `source_only=True` and `target_metrics_computed=False`.

## 8. Time handling — delta-t is a negative ablation

Absolute Time_s is forbidden.

A matched source-only ablation tested causal `log(1 + delta-t)` using the same frozen raw-W TCN family and the same source-validation structure. The preregistered KEEP rule required improvement in overall mean and median MAE, at least 8/12 source-validation wins, improvement within both source rates, and no degradation of mean Q95 absolute error.

Observed delta-t minus no-delta-t results:
- mean delta MAE = +0.0022051;
- median delta MAE = +0.0019449;
- wins = 1/12;
- mean delta Q95-AE = +0.0042641;
- 1C mean delta MAE = +0.0016249, 1/6 wins;
- 2C mean delta MAE = +0.0027852, 0/6 wins.

Decision: **DROP delta-t**.

Therefore the final model remains V/I/W1/W2 only. Non-uniform sampling is documented as a data property, and the causal delta-t experiment is reported as a negative robustness/design ablation rather than silently ignored.

## 9. UQ policy and final CQR decision

UQ is a lightweight reliability layer after point-model freeze. It must not change the point-model identity.

Baseline/final UQ method: **residual split conformal prediction**.

CQR was tested as an optional refinement using only a secondary deterministic whole-segment partition of the already-defined T1 training bucket. Formal T1 validation, formal T1 calibration, formal T1 test, and all T4 target labels were excluded from the CQR KEEP/DROP decision.

Training-side selection results:

### 90% nominal interval
- CQR PICP = 0.9156, MPIW = 0.03132, mean interval score = 0.03882.
- residual split conformal PICP = 0.9782, MPIW = 0.03674, mean interval score = 0.03794.

### 95% nominal interval
- CQR PICP = 0.9484, MPIW = 0.03758, mean interval score = 0.04582.
- residual split conformal PICP = 0.9958, MPIW = 0.04538, mean interval score = 0.04570.

CQR was narrower and met the preregistered minimum coverage thresholds, but its mean interval score was slightly worse than residual split conformal at both nominal levels. Its average relative MIS improvement was −1.30%, below the required +2% threshold.

Decision: **DROP CQR; retain residual split conformal**.

CQR may be reported as a negative UQ ablation: narrower intervals did not translate into better aggregate interval score. Formal test results are not allowed to reverse this decision.

## 10. Required ablations and evidence hierarchy

The final paper should retain evidence that answers distinct questions rather than presenting a large model zoo as co-equal methods.

Core representation/model ablations:
- V/I electrical baseline;
- V/I + raw W1/W2;
- V/I + derived T/F;
- whitened raw-W / whitened T/F diagnostics;
- simple raw/derived PairTCN variants;
- ETMF variants as complexity/fusion negatives where already available.

Design ablations:
- no-delta-t vs causal delta-t;
- residual split conformal vs CQR;
- raw-W test-time FBG noise perturbation.

Do not re-open architecture search after the formal T4 latch is created.

## 11. Formal T4 publication run

The formal T4 runner is now restricted to exactly the frozen design:
- model: IUW-TCN;
- feature mode: raw_w;
- frozen source-only epochs above;
- publication seeds: 0, 1, 2, 3, 4;
- 12 T4 splits per seed;
- no model/baseline selection inside the publication run.

Formal robustness output also adds independent zero-mean Gaussian noise directly to W1/W2 at 0.5, 1.0, and 2.0 pm per wavelength, with five noise draws for each seed/direction/profile/sigma combination. No target-dependent transform is re-estimated.

Across-seed aggregation will report:
- MAE, RMSE, R², Q95-AE, MaxAE;
- per-profile mean/std across seeds;
- per-seed summaries;
- seed-cluster bootstrap confidence intervals;
- raw-W measurement-noise degradation.

## 12. Development vs confirmatory interpretation

The representation-aware design screen used held-out cross-rate target performance for development model selection. Therefore the later same-direction multi-seed experiment must be described as **design-frozen independent-random-seed replication**, not as a brand-new untouched data holdout.

The reverse 2C→1C direction was not the basis for the initial representation winner and consequently provides stronger confirmatory evidence after the model identity is frozen.

This wording is mandatory in the manuscript; seed independence must not be conflated with data-level holdout independence.

## 13. Final publication narrative

The paper should not be written as “we designed an elaborate thermo-mechanical fusion network and proved it works.” The evidence supports a cleaner story:

1. Dual-FBG sensing provides an optical observation of battery internal response that can complement V/I for SOC estimation.
2. Physics-derived T/F decoupling is interpretable but not automatically optimal as a learning coordinate.
3. A representation-aware comparison shows that direct W1/W2 coordinates with a compact causal TCN generalize more reliably than decoupled/whitened and more complex ETMF alternatives.
4. Source-only epoch selection prevents target-guided training-budget tuning.
5. Causal delta-t and CQR are both evaluated under explicit keep/drop gates and rejected when they fail to add robust value.
6. Final T4 multi-seed and raw-W noise tests quantify the robustness of the frozen design rather than continuing model search.

This evidence-first formulation is the active Q2 manuscript direction.

## 14. Frozen execution order from this point

1. Representation/architecture freeze — COMPLETE.
2. Necessary source-only epoch extension and epoch freeze — COMPLETE.
3. Matched causal delta-t keep/drop — COMPLETE: DROP.
4. CQR keep/drop — COMPLETE: DROP; residual split conformal retained.
5. Update this evidence-frozen mainline document — COMPLETE.
6. Create the auditable formal T4 release latch.
7. Run seeds 0–4 formal T4 and aggregate confirmatory statistics.
8. Complete remaining T1/T2/T3/final UQ reporting and manuscript figures/tables around the frozen design without reopening model selection.

Any later change to the model, representation, feature set, epoch plan, or UQ identity must be treated as a new development cycle rather than retroactively folded into the formal T4 run.
