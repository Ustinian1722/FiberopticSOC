# Q2 final architecture-family development screen

Status: **PRE-REGISTERED DEVELOPMENT SCREEN**

This is the final architecture-family screen before model freeze. It uses only the already-exposed development direction `1C -> 2C` with one unseen drive profile held out at a time. The `2C -> 1C` direction remains uninspected until architecture and optical representation are frozen.

## Motivation

The first Mamba-family screen (run `33348016670`) did not improve over the compact raw-W TCN and its optical gate collapsed near a constant value. Current 2025-2026 SOC literature commonly obtains stronger engineering-paper structure by combining causal/local convolutional extraction with Transformer/global attention and, in some cases, CrossAttention. Fiber-optic SOC work also uses CNN-Transformer architectures. Therefore one final literature-aligned architecture family is tested rather than further tuning Mamba.

## Fixed data protocol

- source rate: 1C
- target rate: 2C
- held-out target profile: HWFET, LA92, NEDC, NYCC, US06, WLTC
- for each fold, source training excludes the same profile
- window: 64
- train stride: 4
- test stride: 1
- seed: 42
- fixed equal budget: 20 epochs
- optimizer/loss: same repository AdamW/MSE training utility
- no target-label early stopping or per-profile hyperparameter search
- predictors remain leakage-safe; no `SOC`, `dis_cap`, or absolute `Time_s`

## Candidate set

1. `IUW-TCN`: frozen compact raw-W strong baseline.
2. `CGA-Matched`: matched-protocol CNN-GRU-temporal-attention literature baseline using V/I/W1/W2. It is a CGA-style comparator, not claimed as an exact reproduction of the companion Energy paper.
3. `VIW-Transformer`: single-stream causal Transformer using V/I/W1/W2.
4. `DualTCN-Transformer`: modality-specific causal TCN encoders followed by Transformer temporal modeling; no cross-attention.
5. `EO-CrossFormer`: modality-specific causal TCN local encoders, bidirectional electrical↔optical **causal cross-attention**, interaction fusion, and causal Transformer global modeling; raw W1/W2 optical input.
6. `EO-CrossFormer-TF`: exact same proposed architecture with physics-decoupled T/F instead of raw W1/W2.

The architecture deliberately preserves the local causal-convolution inductive bias that performed well in the existing benchmark while adding explicit cross-modal interaction and global-context modeling. The paper novelty, if retained, is not claimed to be Transformer or CrossAttention individually; it is the representation-aware electrical-optical fusion under compound rate/profile distribution shift.

## Retention rule

`EO-CrossFormer` is retained as the proposed architecture only if, relative to `IUW-TCN` across the six development profiles, all of the following hold:

1. lower aggregate mean MAE;
2. lower aggregate mean RMSE;
3. wins on MAE in at least 4 of 6 profiles;
4. aggregate mean Q95-AE is no worse.

If these conditions fail, the architecture is not rescued by target-guided tuning. The study will then stop architecture search and reframe around the empirically strongest compact estimator plus representation/generalization evidence.

## Representation decision

`EO-CrossFormer` and `EO-CrossFormer-TF` provide a matched raw-W vs T/F comparison under the same advanced architecture. The preferred representation is determined by aggregate MAE/RMSE and profile consistency. Physical T/F decoupling remains valid for interpretation regardless of predictive selection.

## After this screen

Once the architecture/representation decision is recorded:

1. freeze the model family;
2. perform source-only epoch selection for the selected new model only if it is retained;
3. run matched literature baselines and ablations under the frozen compound-shift protocol;
4. inspect the reverse `2C -> 1C` direction only after all design choices are frozen;
5. pursue external FBG validation as a separate generalization layer.