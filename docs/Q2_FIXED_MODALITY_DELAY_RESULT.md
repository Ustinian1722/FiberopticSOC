# Q2 fixed modality-delay residual TCN result

Status: **DROP — INFORMATIVE NEGATIVE ABLATION**

Workflow: `33357531167`
Artifact: `q2-modality-delay-tcn-source-summary`
Selection protocol: 1C-only same-rate leave-one-profile-out, six profiles, seed 42, 20 epochs, no 2C metric used.

## Aggregate

| Model | Params | MAE | RMSE | Q95-AE | MaxAE |
|---|---:|---:|---:|---:|---:|
| IUW-TCN | 11,545 | **0.005718** | **0.007155** | **0.013761** | 0.027780 |
| MD-ResTCN | 19,226 | 0.005816 | 0.007284 | 0.014150 | **0.026472** |

Pre-registered gate:

- mean delta MAE = `+9.762e-05` -> fail
- median delta MAE = `-3.548e-05` -> pass
- MAE wins = `3/6` -> fail
- mean RMSE lower -> fail
- mean Q95 no worse -> fail

Decision: **DROP**.

## Profile pattern

MD-ResTCN improves:

- HWFET: `0.006460 -> 0.006378`
- LA92: `0.008463 -> 0.007040`
- NEDC: `0.005388 -> 0.004070`

but degrades:

- NYCC: `0.005559 -> 0.005570`
- US06: `0.005127 -> 0.006973`
- WLTC: `0.003312 -> 0.004863`

## Interpretation

A single globally fixed fast-electrical / slow-optical receptive-field assignment is not sufficiently robust. The result is nevertheless informative because the candidate helps three profiles substantially while hurting others. This pattern is consistent with a **condition-dependent delay-scale hypothesis**, not with a universal fixed optical lag.

Accordingly, no fixed lag is tuned or rescued. One final source-side candidate, DMD-ResTCN, is allowed to test sample-adaptive short/medium/long optical temporal weighting while retaining the same strong base TCN. If that candidate fails the same gate, the present top-conference-inspired architecture extension closes.