# ETMF-Net novelty boundary for the Q2 manuscript

Date: 2026-08-31
Status: active claim guardrail

## Direct competitor boundary

The manuscript must explicitly acknowledge that the following ideas already exist in directly overlapping battery-SOC literature:

1. Implanted FBG sensing for SOC assistance.
2. Dual-FBG temperature/mechanical cross-sensitivity decoupling.
3. Combining current, voltage, internal temperature and mechanical/stress features as multi-source SOC inputs.
4. CNN/GRU/attention hybrid temporal estimation on the public SiC-18 data.
5. Local-to-global temporal modeling of I/U/T/F and unseen-driving-cycle validation.
6. Quantile-regression and conformalized-quantile SOC uncertainty intervals.

In particular, Journal of Energy Storage 180 (2026) 124293 develops implanted dual-FBG T/F decoupling plus a Local-to-Global Temporal Former (LTF), performs [I,U], [I,U,T], [I,U,F], [I,U,T,F] feature ablations, and validates unseen driving cycles on multi-cell datasets. Therefore none of those ingredients is a standalone novelty claim for ETMF-Net.

The Energy 349 (2026) 140609 SiC-18 paper already combines in-situ FBG features with feature engineering, noise augmentation and a CNN-GRU-Attention model, reporting 0.635% RMSE in its evaluation setting. Its number is not treated as a same-protocol comparator to our strict T4 extrapolation task.

## Q2 method claim

The proposed model contribution is deliberately narrower and defensible:

> **Heterogeneous electrical–thermomechanical temporal encoding with bidirectional cross-gated adaptive fusion for SOC estimation under operating-condition shift.**

### Component A — heterogeneous temporal encoders

Electrical V/I and internal T/F are not passed through one homogeneous backbone.

- V/I: multi-scale causal TCN emphasizes fast load transients and local polarization-related dynamics.
- T/F: compact recurrent encoder emphasizes slower thermo-mechanical state evolution.

The claim is not that TCN or GRU is individually novel. The contribution is the signal-role-specific heterogeneous decomposition and its empirical benefit over architecture-matched stacked fusion and ordinary dual-branch fusion.

### Component B — bidirectional latent cross-gating

The two latent states condition each other before fusion:

- thermo-mechanical state modulates the electrical latent representation;
- electrical state modulates the thermo-mechanical latent representation;
- a sample/window-dependent mixing coefficient then controls their contribution to the fused state.

This is the architectural feature that must survive ablation. If it does not beat ordinary dual-branch fusion after source-only epoch selection and multiple seeds, the paper will simplify rather than retain an unsupported mechanism.

### Component C — hierarchical operating-condition evaluation

The evidence chain is intentionally stronger than a single mixed-condition split:

- T1: blocked mixed-condition interpolation
- T2: same-rate unseen profile
- T3: cross-rate with profile identities seen at source rate
- T4: cross-rate + unseen profile

T4 is the key robustness result, not a claim that the model universally generalizes across cells or chemistries.

### Component D — calibrated UQ as a supporting reliability contribution

The frozen point estimator is followed by source-calibrated split conformal and, if justified, CQR intervals. UQ novelty is not claimed from conformal prediction itself. The supporting question is how empirical coverage and interval efficiency evolve from T1 to T4 condition shift.

## Claims to avoid

Do not claim:

- first FBG-assisted SOC method;
- first dual-FBG T/F decoupling for SOC;
- first multi-source I/U/T/F SOC estimator;
- first local/global temporal SOC model;
- first unseen-driving-cycle FBG SOC validation;
- first conformal or quantile SOC UQ;
- cross-cell or cross-chemistry generalization from SiC-18;
- that W and T/F are independent modalities.

## Manuscript-safe contribution structure

A suitable three-contribution Q2 structure is:

1. Develop a compact heterogeneous electrical–thermomechanical temporal fusion network that assigns fast electrical and slower internal thermo-mechanical measurements to signal-role-specific encoders and performs bidirectional cross-gated adaptive fusion.
2. Establish a leakage-safe hierarchical evaluation protocol from mixed-condition interpolation through unseen-profile, cross-rate and combined cross-rate/unseen-profile extrapolation, with matched feature/fusion/noise ablations.
3. Add source-calibrated SOC uncertainty intervals and evaluate point accuracy together with coverage, interval efficiency and conditional reliability as operating-condition shift becomes more severe.

This claim set is intentionally publication-oriented and should remain fixed unless final evidence invalidates one component.
