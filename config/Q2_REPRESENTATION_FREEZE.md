# Q2 representation / architecture freeze

Status: **FROZEN**

Freeze basis: representation-aware equal-budget development screen, GitHub Actions run `33337345711`, artifact `q2-representation-aware-screen`, seed42, 20 epochs, window64, train stride4, test stride1, batch256. Screen commit: `0099583677425259e5b7e80803368ca2b8497d12`.

## Frozen point-estimator identity

- Final development representation: **raw dual-FBG wavelength coordinates W1/W2**.
- Final development model: **IUW-TCN**, implemented by `PairTCN((2, 3), None)` with electrical I/U stream plus raw W1/W2 optical stream.
- Parameter count: **11,545**.
- Whitening: **not retained** in the mainline.
- T/F decomposition: **not retained** in the mainline.
- ETMF adaptive fusion: **not retained** in the mainline; it remains a negative/complexity ablation.

This freeze is not revisited by later target-domain results. Subsequent epoch, delta-t and UQ decisions must operate on this frozen model family and must use training-side/source-only evidence for keep/drop decisions.

## Equal-budget evidence

Across the six 1C -> 2C unseen-profile development splits:

| Model | MAE mean | MAE std | RMSE mean | R2 mean | Q95-AE mean | MaxAE mean | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| **IUW-TCN** | **0.013847** | **0.005281** | **0.020835** | **0.995231** | **0.048729** | 0.093737 | **11,545** |
| ETMF-TF | 0.015692 | 0.004914 | 0.024558 | 0.993793 | 0.057097 | 0.106039 | 24,434 |
| IUWwhite-TCN | 0.016979 | 0.011183 | 0.024793 | 0.991990 | 0.056262 | **0.093213** | 11,545 |
| ETMF-Wwhite | 0.017007 | 0.004926 | 0.026024 | 0.993135 | 0.058021 | 0.110158 | 24,434 |
| ETMF-TFwhite | 0.018659 | 0.009255 | 0.030419 | 0.989522 | 0.064322 | 0.136583 | 24,434 |
| IUTFwhite-TCN | 0.019602 | 0.009718 | 0.029383 | 0.990451 | 0.066689 | 0.119719 | 11,545 |
| IUTF-TCN | 0.020328 | 0.015327 | 0.030598 | 0.987116 | 0.068351 | 0.124731 | 11,545 |
| ETMF-W | 0.021779 | 0.009658 | 0.035781 | 0.986356 | 0.079914 | 0.155344 | 24,434 |

Relative to the frozen IUW-TCN, the closest aggregate competitor ETMF-TF has 13.3% higher mean MAE, 17.9% higher mean RMSE and 17.2% higher mean Q95-AE while using 2.12x as many parameters. ETMF-TF beats IUW-TCN on only one of six profile-level MAEs and loses on five.

Whitened W improves some individual profiles but is not robust enough to retain: it wins versus raw W on LA92, NEDC and NYCC but loses on HWFET, US06 and WLTC; its mean MAE is 22.6% higher and its cross-profile MAE standard deviation is more than twice that of raw W. Its improved numerical conditioning therefore does not translate into reliable predictive generalization.

The T/F coordinates are strongly ill-conditioned before whitening in the source data (condition number about 107-119 with correlation about -0.982), compared with raw W (condition number about 6.1-7.2, correlation about 0.72-0.76). Whitening fixes numerical conditioning by construction but does not provide a stable accuracy advantage. Conditioning alone is therefore not a reason to retain the transformed coordinate.

## Development-screen disclosure

This screen used the six held-out 2C target profiles in the 1C -> 2C direction to freeze representation and architecture. Those target trajectories are therefore development evidence, not untouched data-level holdouts. Later seeds 0-4 on the same direction are independent-seed robustness statistics after design freeze. Provided the reverse 2C -> 1C target results remain uninspected until all remaining design decisions are frozen, that reverse direction retains the stronger confirmatory role described in `docs/Q2_EVALUATION_PROTOCOLS.md`.

## Next gate

Only the frozen IUW-TCN model is eligible for the remaining design stages, in this order:

1. Reuse the existing source-only raw-W epoch audit wherever its 60-epoch ceiling was not binding; extend only ceiling-bound source-validation folds as needed.
2. Freeze the source-only epoch plan.
3. Run a matched delta-t ablation on IUW-TCN using source-only validation evidence; target labels cannot decide keep/drop.
4. Decide CQR only from training-side/UQ-selection evidence after the point estimator is frozen.
5. Only then release formal T4 seeds 0-4 through `config/START_Q2_FINAL_T4`.
