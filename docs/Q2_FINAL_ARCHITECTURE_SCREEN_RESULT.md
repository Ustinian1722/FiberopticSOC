# Q2 final architecture-family screen result

Status: **RAW EO-CROSSFORMER DROP; TF VARIANT REQUIRES INDEPENDENT SOURCE-SIDE CONFIRMATION**

Canonical workflow: GitHub Actions run `33355158650`, artifact `q2-final-crossformer-screen-summary`, head SHA `3e2d59c6a275a21a5e02706342a2da383d01b13b`.

Protocol: already-exposed development direction `1C -> 2C`, one unseen profile held out at a time, six profiles, seed 42, 20 epochs, window 64, train stride 4, test stride 1. No reverse-direction target metrics were used for this decision.

## Aggregate ranking

| Model | Params | MAE | RMSE | R2 | Q95-AE | MaxAE |
|---|---:|---:|---:|---:|---:|---:|
| **EO-CrossFormer-TF** | 76,753 | **0.013546** | **0.019376** | **0.995621** | **0.039677** | **0.083443** |
| IUW-TCN | 11,545 | 0.013847 | 0.020835 | 0.995231 | 0.048729 | 0.093737 |
| EO-CrossFormer | 76,753 | 0.015190 | 0.025248 | 0.992670 | 0.055202 | 0.131944 |
| DualTCN-Transformer | 64,705 | 0.015658 | 0.023737 | 0.993877 | 0.050330 | 0.100982 |
| VIW-Transformer | 40,753 | 0.019838 | 0.030658 | 0.988947 | 0.067099 | 0.130748 |
| CGA-Matched | 19,314 | 0.025013 | 0.037402 | 0.985002 | 0.083110 | 0.121694 |

## Pre-registered raw-W candidate decision

`EO-CrossFormer` failed every retention criterion relative to IUW-TCN:

- lower aggregate mean MAE: **false** (`0.015190 > 0.013847`)
- lower aggregate mean RMSE: **false** (`0.025248 > 0.020835`)
- MAE wins >=4/6 profiles: **false** (`3/6`)
- mean Q95-AE no worse: **false** (`0.055202 > 0.048729`)

Decision: **DROP raw-W EO-CrossFormer**. It cannot be rescued by target-guided tuning.

## Matched representation result inside CrossFormer

The physics-decoupled T/F coordinate substantially outperformed raw W1/W2 within the exact same 76,753-parameter architecture:

- raw-W CrossFormer MAE: `0.015190`
- T/F CrossFormer MAE: `0.013546`
- raw-W CrossFormer RMSE: `0.025248`
- T/F CrossFormer RMSE: `0.019376`

Thus the matched representation decision inside this architecture favors **physics-decoupled T/F**.

Relative to the compact IUW-TCN, EO-CrossFormer-TF improves aggregate:

- MAE by about 2.18%
- RMSE by about 7.01%
- Q95-AE by about 18.58%
- MaxAE by about 10.98%

However, it wins MAE on only 3/6 profiles (HWFET, LA92, NEDC) and loses on NYCC, US06, WLTC. Its cross-profile MAE standard deviation is about 35% higher than IUW-TCN. Therefore this result is not sufficient to promote the TF model directly.

## Interpretation

W1/W2 and T/F are exactly invertibly related in this dataset, so the T/F advantage does not represent additional sensing information. It demonstrates **architecture-representation interaction**: the CrossFormer fusion operators (cross-attention plus coordinate-wise interaction terms) are not invariant to a non-orthogonal change of optical coordinates. A physically separated thermo-mechanical coordinate can therefore be easier for this particular fusion geometry even though raw W had been superior for the compact TCN family.

This is compatible with the earlier evidence rather than a contradiction:

- compact TCN family -> raw W gave better and more stable cross-profile development performance;
- CrossFormer family -> T/F gives better aggregate development performance, but with profile-dependent gains/losses.

## Next gate

Because EO-CrossFormer-TF emerged as the aggregate leader but was not the pre-registered raw-W retention candidate, it is **not directly frozen as the proposed method**. It receives one independent source-side confirmation only:

- same-rate 1C leave-one-profile-out validation;
- six folds;
- same fixed 20-epoch budget and seed 42;
- compare only IUW-TCN versus EO-CrossFormer-TF;
- no 2C target trajectories in this confirmation;
- no new architecture or hyperparameter candidates.

If the TF model fails this confirmation gate, architecture search stops and IUW-TCN remains the estimator. If it passes, the architecture/representation is frozen as EO-CrossFormer-TF before any new reverse-direction cross-rate metrics are inspected.