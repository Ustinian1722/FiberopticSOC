# Q2 modality-delay residual TCN protocol

Status: **PRE-REGISTERED USER-DIRECTED TOP-CONFERENCE EXTENSION — SOURCE-SIDE ONLY**

The generic ModernTCN-style candidates (`MC-TCN`, `LR-MC-TCN`) failed the 1C source-only gate with 0/6 profile wins. This stage therefore does not continue generic backbone replacement. It transfers a narrower recent time-series idea that is physically aligned with the sensing system: **different variables/modality groups can act on different temporal delays and should not be forced to share one receptive field**.

The motivating recent time-series work is TimePro (ICML 2025), which explicitly formulates a multi-delay issue in multivariate forecasting: distinct variables may influence the target over different temporal intervals. In the present battery setting, electrical voltage/current respond rapidly to load, whereas raw FBG wavelengths contain thermo-mechanical response that can be smoother and delayed. The model below adapts this idea to causal many-to-one SOC estimation rather than reproducing TimePro.

## Candidate — MD-ResTCN

`MD-ResTCN` = **Modality-Delay Residual TCN**.

It preserves the strong raw-W TCN path and adds only a modality-specific temporal correction path:

1. **Base path:** exact causal four-channel V/I/W1/W2 TCN encoder used by IUW-TCN.
2. **Fast electrical path:** V/I encoder with a short causal receptive field.
3. **Slow optical path:** W1/W2 encoder with a substantially longer causal receptive field, spanning most of the 64-sample window.
4. **Interaction correction:** concatenate fast-electrical and slow-optical latent states together with elementwise product and absolute difference, then predict a residual correction.
5. **Residual prediction:** final SOC = base-path prediction + learned modality-delay correction.

The correction head's final layer is initialized to zero so the candidate starts as the strong base TCN and learns only evidence-supported corrections. There is no target-domain lag search, no cross-correlation alignment using future values, and no manually selected target-specific delay.

## Fixed receptive fields

- electrical: kernel 3, dilation 1 and 2 (short/fast response);
- optical: kernel 5, dilation 1, 2 and 4 (long/slow response; effective receptive field approximately covers the 64-sample input window);
- base path: unchanged IUW-TCN encoder.

These values are fixed from causal coverage considerations before results are inspected; they are not tuned per profile.

## Selection protocol

Exactly two models:

- `IUW-TCN` baseline;
- `MD-ResTCN` candidate.

Selection data and budget:

- 1C only;
- leave-one-profile-out across HWFET, LA92, NEDC, NYCC, US06, WLTC;
- six folds;
- seed 42;
- 20 epochs;
- window 64;
- train stride 4;
- validation stride 1;
- batch size 256;
- source-train normalization only;
- predictors V/I/raw W1/raw W2 only;
- no 2C metric is allowed for the decision.

## KEEP gate

All criteria are required relative to IUW-TCN:

1. mean delta MAE (`MD-ResTCN - IUW-TCN`) < 0;
2. median delta MAE < 0;
3. MD-ResTCN wins MAE in at least 4/6 profiles;
4. mean RMSE < IUW-TCN mean RMSE;
5. mean Q95-AE <= IUW-TCN mean Q95-AE.

If any criterion fails: **DROP MD-ResTCN** and close this architecture exploration stage.

If all pass: freeze MD-ResTCN before inspecting any new cross-rate target result, then evaluate the already-exposed 1C->2C direction as a robustness check rather than another model-selection loop.