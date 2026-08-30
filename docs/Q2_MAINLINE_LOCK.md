# High-quality Q2 mainline lock

Date: 2026-08-31
Status: ACTIVE / FROZEN DIRECTION

This document supersedes the previous Q1 expansion plan as the publication target for the current FiberopticSOC paper. The objective is a solid, non-trivial SCI Q2 paper with a familiar battery-SOC research structure, clear method contribution, strict leakage control, complete generalization/robustness evidence, and calibrated uncertainty quantification.

## 1. Paper target

Target contribution style:

> Dual-FBG thermo-mechanical assisted SOC estimation using heterogeneous temporal encoding and adaptive electrical–thermomechanical fusion, validated from conventional mixed-condition accuracy through unseen-profile and cross-rate generalization, with calibrated SOC prediction intervals.

The paper should look and read like a strong recent Energy / Journal of Energy Storage / Journal of Power Sources style battery-AI paper, while avoiding trivial feature concatenation or attention stacking.

## 2. What is in scope

### Core dataset
- SiC-18 public implanted dual-FBG silicon-based lithium-ion battery dataset.
- Six dynamic profiles × two rates (1C/2C), 12 discharge trajectories.
- Inputs derived only from causal observed measurements.

### Core model
Working name: **ETMF-Net** (Electrical–ThermoMechanical Fusion Network).

1. Electrical branch: multi-scale causal TCN for fast load-driven V/I transients.
2. Thermo-mechanical branch: GRU/lightweight temporal encoder for stateful T/F evolution that is weakly coupled to instantaneous current jumps.
3. Adaptive cross-gated latent fusion: learn sample/window-dependent electrical and thermo-mechanical interactions.
4. SOC regression head.

The final method should remain compact. A second large Transformer/Mamba stack is not required.

Data diagnostics do **not** support a blanket claim that T/F simply evolves more slowly than voltage after normalization. The defensible distinction is dynamic-response role: voltage responds strongly to current jumps, whereas T/F is comparatively insensitive to instantaneous load changes and provides complementary internal-state information.

### Required experimental hierarchy

T1. **Mixed-condition benchmark**
- Conventional train/validation/test protocol built at trajectory/window level without row-level leakage.
- Purpose: establish competitive headline accuracy.

T2. **Unseen driving profile**
- Five profiles train, sixth profile test, repeated for all six profiles.
- Same-rate evaluation.

T3. **Cross-rate**
- 1C -> 2C and 2C -> 1C with profile coverage controlled.

T4. **Cross-rate + unseen profile**
- Five source-rate profiles train -> corresponding unseen profile at target rate test.
- 12 strict splits.
- This is the hardest robustness experiment, not the only headline benchmark.

## 3. Required ablations

At minimum:

- V/I electrical baseline
- I/U/T
- I/U/F
- I/U/T/F
- I/U/W1/W2
- single-stream IUTF model
- dual-branch direct fusion
- proposed adaptive ETMF fusion

These ablations answer three questions separately:
1. Does internal sensing help?
2. Which internal component helps?
3. Does heterogeneous fusion outperform simple concatenation?

## 4. Robustness experiments

Required but deliberately lightweight:

- wavelength / thermo-mechanical sensor-noise perturbation
- representation comparison: raw W vs decoupled T/F
- per-SOC-bin error
- high-load vs low-load error
- multiple random seeds for final headline tables

The FBG sensitivity-matrix conditioning result is kept as interpretation/robustness context, not as the central research problem.

## 5. Time handling

The public release is not perfectly uniformly sampled. The final model/protocol must therefore either:

- include causal elapsed-time / delta-t information, or
- use a documented causal fixed-duration preprocessing strategy.

Absolute Time_s is forbidden as a trajectory-progress predictor.

## 6. Leakage rules

Forbidden estimator inputs:

- dis_cap
- SOC or SOC-derived quantities
- absolute Time_s
- terminal capacity / test-set normalization statistics
- future-window statistics

Splits must be made by profile/rate/trajectory before any normalization or model selection.

## 7. Required UQ policy

UQ is a **required component** of the Q2 manuscript, but it is executed only after the point estimator and its hyperparameters are frozen.

### UQ-U1: split conformal baseline
- Use absolute point-prediction residuals as nonconformity scores.
- Calibration samples come only from source-side training/validation data.
- Evaluate at least nominal 90% and 95% coverage.

### UQ-U2: CQR candidate
- Add conformalized quantile regression only if it gives meaningfully more adaptive/narrower intervals while retaining calibrated coverage.
- If CQR does not improve interval quality, retain the simpler split-conformal method in the paper.

Required UQ metrics:
- empirical coverage / PICP
- coverage error
- MPIW
- PINAW
- mean interval score
- conditional coverage by SOC bin and load level
- coverage under T1/T2/T3/T4 operating-condition shift

Test SOC labels are evaluation-only and must never determine calibration quantiles, interval widths, model selection, or UQ hyperparameters.

The UQ contribution should remain lightweight and publication-oriented: a calibrated reliability layer on top of the frozen point model, not a second large uncertainty architecture.

## 8. Second dataset policy

The 12-cell sodium-ion pressure dataset is **not required** for the Q2 paper. It can be added later only if the SiC-18 manuscript is otherwise complete and the adapter is low-cost.

Do not let cross-chemistry validation delay the primary paper.

## 9. Superseded/deferred Q1 work

The following are retained as useful diagnostics or future extensions, but are no longer publication prerequisites:

- cross-chemistry validation
- full sensor-uncertainty propagation
- coordinate-invariant optical encoder
- complex support-complementarity gating
- NestedCal99 as the principal method
- multi-stage sensor-level reliability propagation

Existing runs are not deleted; they can support motivation or supplementary ablations.

## 10. Go/no-go criteria for ETMF-Net

The adaptive dual-branch model survives to the final paper only if it satisfies both:

1. It improves over the electrical-only baseline on the aggregate T1/T2/T3/T4 evidence, not just one profile; and
2. It is competitive with or better than the simple IUTF concatenation baseline, especially on at least one generalization protocol.

If adaptive fusion fails this test, use the simpler heterogeneous direct-fusion model rather than forcing an unnecessary gating story.

## 11. Publication-first execution order

1. Q2-P1: ETMF-Net development screen on strict T4.
2. Q2-P2: source-only epoch selection + 3–5 seeds for surviving models.
3. Q2-T1/T2/T3/T4 complete benchmark suite.
4. Feature/representation/noise ablations.
5. Freeze the point estimator.
6. UQ-U1 split conformal; then UQ-U2 CQR only if justified by interval metrics.
7. Figures, statistics, and manuscript writing around the evidence actually obtained.

This scope is frozen unless a fatal data or methodological flaw is discovered.
