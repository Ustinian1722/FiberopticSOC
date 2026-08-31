# Q2 numeric claim audit / manuscript source of truth

Status: **FROZEN NUMERIC CLAIM MAP**

All manuscript numbers should be copied from this file or its named source-data file. Do not recalculate a rounded value by hand during English editing.

## Dataset

| Claim | Manuscript value | Source / note |
|---|---:|---|
| Main dynamic profiles | 6 | HWFET, LA92, NEDC, NYCC, US06, WLTC |
| Discharge rates | 2 | 1C, 2C |
| Main trajectories | 12 | 6 profiles × 2 rates |
| Released effective rows | ~68,086 | SiC-18 dataset description / Section 2 |
| Final causal window | 64 samples | frozen model configuration |
| RA-FBG-TCN parameters | 11,545 | `paper/source_data/table3_model_comparison.csv` |

## Fig. 2 representation statistics

Use only the final directly reproducible Pearson statistics in the main manuscript:

| Representation | Rate | Absolute Pearson r |
|---|---|---:|
| Raw W1/W2 | 1C | 0.738 |
| Raw W1/W2 | 2C | 0.655 |
| Decoupled T/F | 1C | 0.982 |
| Decoupled T/F | 2C | 0.985 |

Do not reintroduce the old development-stage condition-number approximations into the main text.

Matched representation transfer:

| Representation/model | MAE (% SOC) | RMSE (% SOC) | Q95-AE (% SOC) |
|---|---:|---:|---:|
| Raw W + compact causal TCN | 1.385 | 2.084 | 4.873 |
| Decoupled T/F + matched TCN | 2.033 | 3.060 | 6.835 |
| ETMF-TF | 1.569 | 2.456 | 5.710 |

Primary source: `paper/source_data/fig2_representation_transfer.csv` plus frozen development summary.

## Conventional blocked-T1 result

RA-FBG-TCN:
- MAE: **0.482% SOC**;
- RMSE: **0.593% SOC**;
- R²: **0.999614**;
- Q95-AE: **1.088% SOC**;
- parameters: **11,545**.

Strong in-distribution electrical baseline:
- VI-TCN MAE: **0.231% SOC**.

Source: `paper/source_data/table3_model_comparison.csv`.

Narrative constraint: never say RA-FBG-TCN is the best conventional model in Table 3.

## Parameter-matched VI versus VI+W cross-rate result

| Direction | VI MAE | VI+W MAE | Relative MAE reduction |
|---|---:|---:|---:|
| 1C → 2C | 2.151% | 1.632% | 24.1% |
| 2C → 1C | 0.866% | 0.814% | 6.0% |

Source: `paper/source_data/fig5_direction_summary.csv`.

Use positive values (24.1%, 6.0%) when wording the improvement as a **reduction/gain**. Internal raw `relative_change` may use the opposite sign; do not mix the conventions.

## Electrical-OOD optical gain

| OOD bin | VI MAE | VI+W MAE | Relative optical gain |
|---|---:|---:|---:|
| ID | 0.735% | 0.873% | −18.75% |
| 0–25% | 1.550% | 1.470% | +5.17% |
| 25–50% | 2.772% | 2.356% | +15.00% |
| 50–75% | 3.665% | 2.930% | +20.05% |
| 75–100% | 5.529% | 2.846% | +48.52% |

Source: `paper/source_data/fig5_ood_bins.csv`.

Main-paper headline: optical gain reaches **48.52%** in the most severe OOD bin. Do not claim universal gain in ID regions.

## Strict five-seed T4

| Direction | MAE | RMSE | R² | Q95-AE |
|---|---:|---:|---:|---:|
| 1C → 2C | 1.795% | 3.037% | 0.985689 | 6.804% |
| 2C → 1C | 0.806% | 0.988% | 0.998480 | 1.830% |

Overall seed-cluster:
- MAE: **1.301% SOC**;
- bootstrap 95% CI: **0.961–1.677% SOC**.

Sources:
- `paper/source_data/fig6_t4_profile_summary.csv`;
- `paper/source_data/fig6_t4_bootstrap.csv`;
- frozen workflow `33346131766`.

## Direct FBG wavelength-noise robustness

At independent Gaussian wavelength noise σ = 2 pm/channel:
- 1C → 2C MAE relative increase: **~1.67%**;
- 2C → 1C MAE relative increase: **~4.57%**.

Source: `paper/source_data/fig7_noise_summary.csv`.

Wording: smooth/small degradation; do not claim invariance to noise.

## 95% residual split conformal

- nominal coverage: **95%**;
- empirical PICP: **95.04%**;
- MPIW: **2.075% SOC**;
- MIS: **0.02441**;
- residual quantile: **~1.089% SOC**.

Source: `paper/source_data/fig7_uq_summary.csv` and fixed blocked-T1 UQ rerun.

Main paper uses the 95% level only. Do not promote the 90% result.

## Optional external-dataset sentence only

If retained in Discussion or supplementary context:
- same-cell 0.2C+0.5C → 1C four-cell mean MAE: VI **14.92%**, VI+S5rel **13.42%**;
- MAE improved on **3/4 cells**;
- tail Q95 robustness did not pass the pre-specified robust-benefit gate.

Do not state cross-cell generalization.

## Rounding policy

- Main MAE/RMSE/Q95 values: normally 3 decimals in tables/text.
- OOD gain: 2 decimals when quoting the maximum 48.52%; one decimal is acceptable in Discussion prose (48.5%).
- R²: 6 decimals in Table 3/Table 5; may use 4–6 decimals in prose according to journal style.
- CI endpoints: 3 decimals.
- PICP/MPIW: 2 and 3 decimals respectively in percent units.

## Claims that are explicitly prohibited

- “FBG improves SOC estimation under all conditions.”
- “RA-FBG-TCN outperforms all baselines.”
- “T/F decoupling is physically invalid.”
- “The method demonstrates universal cross-cell generalization.”
- “The present study is the first FBG-based SOC estimator.”
- Directly comparing the Energy 2026 paper's 0.635% RMSE with strict T4 as if the evaluation protocols were matched.
