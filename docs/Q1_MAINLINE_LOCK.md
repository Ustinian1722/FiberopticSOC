# Q1 research mainline lock

Date: 2026-08-31

This document freezes the intended paper direction. Previous OOD selectors, whitening diagnostics and representation studies remain reproducible diagnostics, but they no longer define the paper method.

## Target paper taste

The target is a Q1-style battery state-estimation paper with a familiar and defensible structure rather than a sensor-inverse-problem paper. The work should combine:

1. a clear battery/measurement motivation;
2. a modest but coherent model contribution;
3. hierarchical generalization experiments;
4. a second mechanical-sensing dataset if available;
5. robustness and calibrated uncertainty as supporting evidence.

The model should remain simpler than a stack of unrelated modules. Every added component must have an ablation and a role in the paper story.

## Working thesis

> Internal mechanical sensing can complement terminal electrical measurements under dynamic and shifted operating conditions, but naive permanent fusion can introduce negative transfer. A battery SOC estimator should preserve a strong electrical path and use thermo-mechanical information as a bounded, adaptive correction.

For SiC-18, the FBG-derived T/F channels are accepted as the public authors' calibrated thermo-mechanical representation. Raw W1/W2 are retained as an alternative optical view and an ablation; the paper will not claim that W and T/F are four independent sensing modalities.

## Method roadmap

### Stage M1 — heterogeneous dual-branch estimator

Electrical path:

`[V, I] -> multi-scale causal TCN -> h_e -> base SOC`

Thermo-mechanical path:

`[T, F] -> GRU/temporal encoder -> h_m`

Fusion:

`SOC_hat = SOC_e + g * Delta_SOC_m`

where `g in [0,1]` and the mechanical residual is explicitly bounded. This guarantees that the mechanical branch acts as a correction rather than replacing the electrical estimator.

Controls required at M1:

- VI electrical-only backbone;
- single-stream `[V,I,T,F]` TCN;
- heterogeneous dual-branch direct fusion;
- bounded residual fusion using T/F;
- bounded residual fusion using raw W1/W2.

### Stage M2 — raw/decoupled multi-view consistency

Only if M1 is useful, introduce a training-time relation between raw optical W and decoupled T/F latent representations. The goal is not to claim extra sensing information, but to exploit two coordinate views of the same FBG observations for regularization/robustness.

Possible form:

`L = L_SOC + lambda_cons * L_cons(z_W, z_TF)`

The consistency term must be validated independently and must not use target-domain labels.

### Stage M3 — reliability-aware fusion

Only after M1/M2 are stable, allow the mechanical correction gate to depend on source-domain reliability/support features. Previous AnyOOD/NestedCal99 results are useful diagnostics but are not automatically adopted as the final mechanism.

### Stage M4 — uncertainty quantification

Add post-hoc split conformal or CQR after the point estimator is frozen. Report coverage and interval width. UQ is a reliability layer, not a substitute for the point-estimation contribution.

## Hierarchical experimental protocol

### T1. Conventional multi-condition benchmark

Purpose: demonstrate competitive accuracy under the standard multi-condition setting used by related SOC papers.

The split must avoid direct `dis_cap`, absolute `Time_s`, and cumulative-Ah leakage. Window leakage between train/test partitions must be controlled explicitly.

### T2. Same-rate unseen-profile generalization

For each rate separately: five driving profiles train, the sixth profile test; repeat six times.

### T3. Cross-rate generalization

Train all 1C, test all 2C; reverse.

### T4. Cross-rate + unseen-profile extrapolation

For each held-out profile: train on the other five source-rate profiles and test the corresponding held-out profile at the target rate. Repeat six profiles in each direction.

This is the strongest SiC-18 protocol and should be reported separately from the conventional benchmark rather than compared numerically with pooled/random-split literature results.

## Second dataset requirement for Q1 positioning

SiC-18 contains only one physical cell. A stronger Q1 paper should therefore add at least one independent mechanical-sensing battery dataset.

Preferred second validation route:

- the existing multi-cell sodium-ion pressure SOC dataset, if it can be represented through a common electrical + mechanical interface;
- otherwise a public battery expansion/pressure/strain dataset suitable for SOC estimation.

The second dataset does not need identical FBG physics. Its role is to test whether the electrical–mechanical fusion principle transfers beyond one implanted FBG cell.

## Required robustness and analysis

- feature ablation: VI / VI+T / VI+F / VI+TF / VI+W;
- raw-vs-decoupled optical representation;
- Gaussian sensor-noise stress tests;
- FBG calibration/sensitivity perturbation as a supplemental robustness experiment;
- per-SOC-bin metrics;
- low-load vs high-load metrics;
- per-profile metrics;
- 5 independent seeds for final neural-network results;
- paired split-level statistics and seed-aware uncertainty;
- model parameter count and inference cost.

## Leakage rules

Forbidden estimator inputs:

- `SOC`;
- `dis_cap`;
- absolute `Time_s`;
- any cumulative-current/Ah feature reconstructed from the complete trajectory;
- terminal capacity or any quantity that uses future samples.

Timestamp differences may be investigated as a causal sampling-context feature because the release is not uniformly sampled, but absolute elapsed trajectory time remains prohibited.

## Literature boundary

Recent direct competitors already cover:

- mechanical-stress + TCN/SE SOC estimation (Energy, 2025);
- implanted dual-FBG + T/F + CNN/Transformer-style fusion (JES, 2025–2026);
- implanted dual-FBG + Local-to-Global Temporal Former with unseen-cycle validation (JES, 2026);
- optical-strain + adaptive GPR-UKF at pack level (Applied Energy, 2026);
- conformalized quantile regression for SOC UQ (JPS, 2026).

Therefore the paper must not claim novelty from FBG sensing, T/F decoupling, TCN/GRU/Transformer blocks, adaptive weighting, or conformal UQ individually. The intended differentiator is the integrated problem formulation: heterogeneous electrical–mechanical representation, bounded correction rather than naive permanent fusion, hierarchical condition-shift validation, and cross-dataset verification.

## Decision rule

The project advances in layers. Do not add M2, M3 or M4 merely because they sound novel. M1 must first beat or meaningfully stabilize strong electrical and simple-fusion baselines under T2–T4. If it does not, redesign M1 before adding complexity.
