# Q2 top-conference-inspired TCN extension protocol

Status: **USER-DIRECTED EXPLORATORY REOPEN; SOURCE-SIDE SELECTION ONLY**

The previous architecture search was closed after Mamba and CrossFormer families failed stability gates. The user explicitly requested one further exploration stage based on transferable modeling ideas from recent top time-series conferences rather than model-name chasing. This document defines that stage before its results are inspected.

## Literature ideas transferred

The extension is informed by the following recent top-conference themes:

- **ModernTCN, ICLR 2024:** modernize convolution using large-kernel depthwise temporal mixing and separate temporal/feature/variable mixing rather than stacking traditional small-kernel TCN blocks.
- **iTransformer, ICLR 2024 / TimeXer, NeurIPS 2024:** treat variable relations explicitly instead of irreversibly blending heterogeneous physical measurements at every timestamp.
- **TimeMixer / Pathformer, ICLR 2024:** temporal patterns exist at different effective scales; local and macroscopic variations should not be forced through the same operator.
- **SAN, NeurIPS 2023; FAN and DDN, NeurIPS 2024; TimeBridge, ICML 2025:** non-stationarity should be handled locally, while useful slow/level information should be preserved rather than globally removed.

The present screen does not reproduce those forecasting models. It transfers their inductive biases to causal many-to-one SOC estimation.

## Candidate A — MC-TCN

A compact **Modern Causal TCN** operating on the final raw predictor set V/I/W1/W2:

1. variable-specific scalar-to-feature embedding;
2. large-kernel **causal depthwise** convolution for temporal mixing;
3. pointwise grouped feature mixing within each physical variable;
4. pointwise grouped variable mixing across V/I/W1/W2 for each latent feature;
5. residual connections throughout;
6. last-time-step causal readout.

The temporal and cross-variable operations are deliberately separated. No self-attention, Mamba, target-domain adaptation, or physics-decoupled T/F view is used.

## Candidate B — LR-MC-TCN

A **Level-Residual Modern Causal TCN** using the same MC-TCN dynamic encoder, but with two information paths:

- a locally standardized residual sequence for short-scale dynamics;
- a raw-level summary path containing the source-normalized last value, window mean, window standard deviation, and window endpoint change for each V/I/W1/W2 channel.

The motivation is that cross-rate/profile shift contains local non-stationarity, but SOC estimation also depends on physically meaningful absolute operating level. Therefore local normalization is not allowed to erase the level information; the level path is explicitly retained.

All statistics use only the observed causal input window. No future sample or target label enters the transformation.

## Selection protocol

Architecture selection uses **1C source-side same-rate leave-one-profile-out only**:

- profiles: HWFET, LA92, NEDC, NYCC, US06, WLTC;
- six folds;
- train on five 1C profiles and validate on the sixth 1C profile;
- seed 42;
- fixed 20 epochs;
- window 64;
- train stride 4;
- validation stride 1;
- batch size 256;
- global feature normalization fit on each fold's five source profiles only.

Models compared:

1. IUW-TCN — existing 11,545-parameter raw-W baseline;
2. MC-TCN;
3. LR-MC-TCN.

No 2C label or metric may be used for this architecture decision.

## Retention gate

A new candidate can advance only if **all** conditions hold relative to IUW-TCN:

1. mean delta MAE (candidate - IUW) < 0;
2. median delta MAE < 0;
3. candidate wins MAE in at least 4/6 profiles;
4. mean RMSE < IUW mean RMSE;
5. mean Q95-AE <= IUW mean Q95-AE.

If both new candidates pass, choose the one with lower mean MAE; mean Q95-AE is the tie-breaker. If neither passes, the user-directed architecture extension closes and IUW-TCN remains the final estimator.

Only a candidate passing this source-side gate may subsequently be evaluated on the already-exposed 1C->2C development direction. That later result is validation of cross-rate behavior, not another architecture-selection loop.