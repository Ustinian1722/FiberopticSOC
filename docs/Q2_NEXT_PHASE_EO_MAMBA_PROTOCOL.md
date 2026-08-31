# Q2 next phase: representation-aware electrical-optical Mamba

Status: DEVELOPMENT TRACK

This track starts from the frozen publication baseline at commit `5747b32b26a4ca851bb4494c43b4da09d71b49d6`. The completed IUW-TCN T4 benchmark is preserved unchanged and becomes a strong lightweight baseline rather than being overwritten.

## Research positioning

The SiC-18 data are public and the companion work already demonstrates FBG-assisted SOC estimation. Therefore the next paper must not claim dataset novelty, sensing-system novelty, or novelty merely from replacing one sequence model with another.

The paper question is instead:

> Under compound operating-condition shift (simultaneous C-rate shift and unseen drive-profile shift), which optical representation and fusion strategy best preserve SOC generalization?

The intended story is:

`raw optical representation -> modality-specific multi-scale encoding -> electrical-conditioned optical gating -> selective state-space temporal modeling -> robust SOC estimation`

## Representation policy

Main representation: `Voltage_V, Current_A, Wavelength_1, Wavelength_2`.

The physics-decoupled `temperature` / `force` coordinates remain an important matched representation baseline. They are not treated as additional sensing degrees of freedom because the released data are related to W1/W2 by an invertible 2x2 linear sensitivity transform.

Interpretation:

- physical thermo-mechanical decoupling is legitimate for state interpretation;
- it is not assumed to be the optimal predictive coordinate under distribution shift;
- raw W and decoupled T/F must be compared as alternative two-dimensional optical representations;
- `W1,W2,T,F` must never be counted as four independent sensing modalities.

## Leakage constraints

Forbidden predictors remain:

- `SOC`
- `dis_cap`
- absolute `Time_s`

All feature normalization is fitted only on source-training trajectories for each split.

## Architecture-development protocol

The first architecture screen intentionally uses the already-development-exposed direction only:

- train rate: 1C
- target rate: 2C
- six leave-one-profile-out compound-shift splits
- train on five 1C profiles
- test on the held-out profile at 2C
- profiles: HWFET, LA92, NEDC, NYCC, US06, WLTC
- seed: 42
- causal window: 64
- train stride: 4
- test stride: 1
- equal epoch budget across candidates

This stage is DEVELOPMENT EVIDENCE, not a new untouched holdout. It may be used to choose the proposed architecture because the same 1C->2C target profiles were already used by the earlier representation screen.

The reverse 2C->1C direction is not used for architecture search. It is reserved for later confirmatory evaluation after architecture, representation, and source-only epoch selection are frozen.

## Candidate ladder

The first screen compares a deliberately small ladder:

1. `IUW-TCN`: frozen strong lightweight raw-W baseline.
2. `VI-Mamba`: electrical-only selective state-space model.
3. `VIW-Mamba`: naive concatenated V/I/W1/W2 Mamba.
4. `DualMS-Mamba`: modality-specific multi-scale electrical/optical encoding, no optical gate.
5. `EO-Gated-TCN`: multi-scale dual encoders + electrical-conditioned optical gate + TCN backbone.
6. `EO-Gated-Mamba`: full raw-W proposed candidate.
7. `EO-Gated-Mamba-TF`: exact same full architecture with T/F replacing W1/W2; representation control.

This ladder isolates the contribution of optical sensing, selective state-space modeling, modality-specific multi-scale encoding, electrical-conditioned optical gating, and raw-vs-decoupled optical representation.

## Model principle

The Mamba candidate uses a pure-PyTorch implementation of the selective state-space core so that CPU GitHub Actions remain reproducible. Each block uses:

- causal depthwise local convolution;
- input-dependent step size `delta`;
- learned negative diagonal state matrix `A`;
- input-dependent `B` and `C` state projections;
- recurrent selective scan with linear sequence complexity;
- gated output projection and residual feed-forward block.

The implementation is intended to preserve the core selective SSM mechanism rather than depend on CUDA-specific fused kernels.

## What counts as a useful result

Primary reporting metrics:

- MAE
- RMSE
- R2
- Q95 absolute error
- MaxAE
- profile-wise win/loss counts
- cross-profile MAE standard deviation
- parameter count

The full model is not automatically retained because it is more complex. It should show a credible advantage over IUW-TCN and over naive VIW-Mamba, preferably through both lower mean error and better profile stability.

If `EO-Gated-Mamba-TF` wins, the representation conclusion must be revised rather than forcing raw W. If raw W remains stronger, T/F becomes the physics-interpretable representation ablation in the paper.

## After the architecture screen

Only after a candidate is selected from development evidence:

1. perform source-only epoch selection for that candidate;
2. freeze architecture and representation;
3. run 5-seed compound-shift evaluation in both directions;
4. compare against IUW-TCN and reimplemented literature baselines (including the companion CNN-GRU-Attention family) under the identical split protocol;
5. retain raw-W measurement-noise robustness and residual split-conformal UQ as reliability analyses;
6. investigate a second open FBG dataset for external validation without claiming cell-to-cell generalization from SiC-18 alone.
