# Q2 dynamic multi-delay residual TCN protocol

Status: **PRE-REGISTERED FINAL IDEA-DRIVEN TCN EXTENSION; SOURCE-SIDE SELECTION ONLY**

## Motivation

The fixed modality-delay candidate (`MD-ResTCN`) tested the hypothesis that electrical V/I should use a shorter causal receptive field while raw FBG W1/W2 should use a longer one. In the pre-registered 1C source-only LOPO screen, it improved HWFET, LA92 and NEDC but degraded NYCC, US06 and WLTC. Aggregate MAE/RMSE/Q95 were slightly worse than IUW-TCN, so the fixed-delay model was correctly dropped.

This pattern supports a narrower hypothesis: **the useful optical delay scale is condition dependent rather than globally fixed**.

TimePro (ICML 2025) formulates the multi-delay problem as different variables influencing the target over distinct temporal intervals and addresses it by adaptively focusing on salient time information rather than applying one uniform temporal treatment. The model below transfers that inductive bias to causal SOC estimation without reproducing TimePro's Mamba backbone.

## Candidate — DMD-ResTCN

`DMD-ResTCN` = **Dynamic Multi-Delay Residual TCN**.

The strong IUW-TCN joint path is preserved unchanged. A zero-initialized correction module contains:

1. a short electrical V/I encoder (effective RF ~13 samples);
2. three raw-optical W1/W2 encoders with fixed short / medium / long causal receptive fields:
   - short RF ~13;
   - medium RF ~29;
   - long RF ~57;
3. a per-window softmax selector that receives the electrical state and all three optical-scale states and outputs three nonnegative weights summing to one;
4. a weighted optical state `o = w_short o_short + w_mid o_mid + w_long o_long`;
5. interaction features `[e, o, e*o, |e-o|]` feeding a residual SOC correction;
6. the final correction layer is initialized to zero, so training starts from the strong base predictor rather than replacing it.

The selector is **sample-adaptive**. There is no profile ID, C-rate label, SOC label, target-domain statistic, cross-correlation alignment, or future observation in the selector.

## Fixed temporal scales

Before results are inspected:

- electrical: kernel 3, dilations 1/2 -> RF ~13;
- optical short: kernel 3, dilations 1/2 -> RF ~13;
- optical medium: kernel 3, dilations 1/2/4 -> RF ~29;
- optical long: kernel 5, dilations 1/2/4 -> RF ~57;
- input window: 64.

No receptive-field grid search is allowed.

## Selection protocol

Exactly two models:

1. IUW-TCN;
2. DMD-ResTCN.

Selection uses only 1C same-rate leave-one-profile-out:

- HWFET, LA92, NEDC, NYCC, US06, WLTC;
- six folds;
- seed 42;
- fixed 20 epochs;
- window 64;
- train stride 4;
- validation stride 1;
- batch size 256;
- source-training normalization only;
- predictors V/I/W1/W2 only;
- no 2C metric may influence the decision.

## KEEP gate

All criteria are required relative to IUW-TCN:

1. mean delta MAE (`DMD - IUW`) < 0;
2. median delta MAE < 0;
3. DMD wins MAE in at least 4/6 profiles;
4. mean RMSE < IUW mean RMSE;
5. mean Q95-AE <= IUW mean Q95-AE.

If any criterion fails, **DROP DMD-ResTCN and close the present top-conference-inspired architecture exploration**.

If all pass, freeze DMD-ResTCN before any new target-domain inspection. Gate weights may then be reported descriptively to test whether different input windows actually use different optical temporal scales; they cannot be used for post-hoc architecture tuning.