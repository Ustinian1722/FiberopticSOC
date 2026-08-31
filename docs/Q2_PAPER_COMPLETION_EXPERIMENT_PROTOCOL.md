# Q2 paper-completion experiment protocol

Status: **FROZEN BEFORE NEW TARGET RESULTS**

Purpose: add only the two missing experiment blocks needed to make the manuscript read like a conventional battery-SOC paper. These experiments do not reopen architecture search, optical representation search, UQ selection, external-data rescue, or any target-guided tuning.

## A. Strict cross-condition backbone benchmark

Research question: under the paper's actual main task, which standard temporal backbone best uses the same electrical–optical input?

Protocol:
- task: `cross-rate + unseen-profile`;
- directions: `1C→2C` and `2C→1C`;
- six held-out profiles in each direction: HWFET, LA92, NEDC, NYCC, US06, WLTC;
- input for every model: `V, I, W1, W2` only;
- window: 64 samples;
- train stride: 4; test stride: 1;
- train-only normalization;
- seed: 42 for all benchmark models;
- training epoch for each direction/profile: exactly the source-only `selected_epoch` already frozen in `config/q2_frozen_epoch_plan.csv`;
- no target early stopping and no model-specific target tuning.

Frozen models:
1. CNN;
2. GRU;
3. LSTM;
4. Transformer;
5. CGA-Matched (CNN-GRU-temporal-attention style comparator; not claimed as an exact reproduction of Ling et al.);
6. DualTCN-Transformer;
7. RA-FBG-TCN (the frozen compact raw-W PairTCN).

Primary reporting: direction-level and overall equal-split mean MAE, RMSE, R², Q95-AE, parameter count; per-profile values retained as supporting source data.

No new model family may be added after target metrics are observed.

## B. Strict cross-condition input/representation ablation

Research question: under identical TCN capacity and training budget, what is the contribution of optical information and which optical coordinate is preferable?

Protocol:
- same 12 strict cross-rate + unseen-profile splits as Experiment A;
- seeds: exactly `[0,1,2,3,4]`;
- same `config/q2_frozen_epoch_plan.csv` split-specific epochs for every model and seed;
- window 64, train stride 4, test stride 1, batch size 256, lr 1e-3;
- train-only normalization;
- no target-domain early stopping or tuning.

Frozen matched models:
1. `VI`: parameter-matched 4-channel TCN receiving V/I plus two fixed-zero channels;
2. `VI+TF`: identical PairTCN receiving V/I and decoupled T/F;
3. `VI+W`: identical PairTCN receiving V/I and native W1/W2 (the final RA-FBG-TCN input).

Primary reporting:
- five-seed direction-level mean MAE/RMSE/R²/Q95-AE;
- overall equal-split mean;
- paired `VI - VI+W` and `VI+TF - VI+W` MAE differences over seed×profile pairs;
- number of wins/losses by direction;
- seed-cluster bootstrap 95% CI for paired MAE gain.

Interpretation rules:
- if VI+W improves the strict task, it supports optical complementarity under condition shift;
- if VI+W is better than VI+TF, it supports native-wavelength representation for predictive transfer;
- if either comparison is not favorable, report it as an ablation boundary and do not tune a rescue model.

## Stop rule

After A and B finish, no further numerical experiment is permitted unless a factual inconsistency is discovered or a reviewer explicitly requests it. The manuscript then moves directly to final restructuring, Fig. 3 artwork, and submission formatting.
