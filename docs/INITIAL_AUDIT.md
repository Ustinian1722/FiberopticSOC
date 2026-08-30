# Initial audit of SiC-18 fiber-optic SOC dataset

## Archive provenance

- User-provided archive: `SiC-18.zip`
- SHA-256: `8ebc43eb6d205dfc573dae853d63efcd1a00169be9d2f1e512b9220ea7799bc4`
- Archive size: 5,398,170 bytes
- Files: 12 `.xlsx` files
- Total samples: 68,086
- Profiles: HWFET, LA92, NEDC, NYCC, US06, WLTC
- Rates: 1C and 2C
- Cell identifier: SiC-18 (single cell)

Each workbook contains the same 9 columns: `Time_s`, `Current_A`, `Voltage_V`, `Wavelength_1`, `Wavelength_2`, `temperature_℃`, `force_N`, `dis_cap`, `SOC`. No missing values or duplicated rows were found.

## Critical finding 1: SOC label leakage through discharge capacity

For 11/12 workbooks, the supplied SOC satisfies exactly:

`SOC = 1 - dis_cap / max(dis_cap)`

For `HWFET_1C.xlsx`, the maximum discrepancy from that relationship is only `2.77e-05` near the end of discharge. Therefore `dis_cap` must be excluded from all SOC-model inputs. Absolute elapsed time should also be excluded from the main model because each file is a single monotonic discharge trajectory and `Time_s` can act as a progress proxy.

## Critical finding 2: temperature and force are deterministic transforms of the two FBG channels

Across all 68,086 samples, ordinary least squares gives:

`temperature_℃ = 214.429868819374 * Wavelength_1 - 136.226034308779 * Wavelength_2`

`force_N = -6407.66902119071 * Wavelength_1 + 5247.22502522705 * Wavelength_2`

Both equations have `R² = 1.000000` with numerical residuals on the order of `1e-13`. Hence `Wavelength_1/2`, `temperature`, and `force` are not four independent sensor channels. A paper should compare **raw FBG representation** (`W1,W2`) against **physics-decoupled representation** (`temperature,force`) as alternative representations, rather than concatenate all four and claim four-channel sensing.

## Critical finding 3: strong SOC relation exists, especially for mechanical force

Pooled Pearson / Spearman correlation with SOC:

| Signal | Pearson | Spearman |
|---|---:|---:|
| Voltage | 0.9696 | 0.9840 |
| Wavelength 1 | 0.4961 | 0.4010 |
| Wavelength 2 | 0.9574 | 0.9431 |
| Decoupled temperature | -0.9034 | -0.9390 |
| Decoupled force | 0.9620 | 0.9869 |
| Current | -0.1384 | -0.1785 |

The force-SOC relationship is highly monotonic across this single cell. However, this must not be interpreted as cross-cell generality because the release contains only SiC-18.

## Baseline protocol

A deliberately simple non-sequential `HistGradientBoostingRegressor` was used only as a diagnostic. Forbidden inputs: `dis_cap`, `SOC`, and `Time_s`. The primary diagnostic is leave-one-driving-profile-out within each C-rate: train on five profiles and test on the sixth, repeated for all six profiles and both rates.

### Mean leave-one-profile-out results

| features | MAE | RMSE | R2 | MaxAE |
|---|---:|---:|---:|---:|
| VI+TF | 0.007691 | 0.010874 | 0.998758 | 0.052455 |
| VI | 0.009320 | 0.013430 | 0.998034 | 0.063687 |
| VI+W | 0.009987 | 0.013808 | 0.997536 | 0.056464 |
| TF | 0.020697 | 0.027748 | 0.991866 | 0.124772 |
| W | 0.025389 | 0.033330 | 0.986991 | 0.129876 |
| V | 0.045140 | 0.058556 | 0.962934 | 0.168287 |

`VI+TF` gives the best average MAE (`0.7691%` SOC fraction), improving over `VI` (`0.9320%`). Raw wavelengths (`VI+W`) are not uniformly better than `VI+TF`; their performance is less stable across held-out profiles, especially `NEDC_2C`.

### Cross-rate diagnostic

| train | test | features | MAE | RMSE | R2 | MaxAE |
|---|---|---|---:|---:|---:|---:|
| 1C | 2C | VI+W | 0.015366 | 0.029305 | 0.991858 | 0.197701 |
| 1C | 2C | VI+TF | 0.018753 | 0.032762 | 0.989823 | 0.149499 |
| 1C | 2C | TF | 0.021639 | 0.029211 | 0.991910 | 0.157826 |
| 1C | 2C | VI | 0.025603 | 0.045765 | 0.980143 | 0.200645 |
| 1C | 2C | W | 0.025921 | 0.035099 | 0.988320 | 0.172383 |
| 1C | 2C | V | 0.054961 | 0.082041 | 0.936186 | 0.274346 |
| 2C | 1C | VI+TF | 0.008024 | 0.011031 | 0.998710 | 0.076290 |
| 2C | 1C | VI | 0.011066 | 0.014782 | 0.997683 | 0.063940 |
| 2C | 1C | VI+W | 0.011281 | 0.016286 | 0.997187 | 0.076632 |
| 2C | 1C | TF | 0.020325 | 0.026808 | 0.992378 | 0.148519 |
| 2C | 1C | W | 0.022892 | 0.030427 | 0.990181 | 0.124461 |
| 2C | 1C | V | 0.051424 | 0.065333 | 0.954733 | 0.189230 |

The 1C→2C direction is harder than 2C→1C. Raw wavelengths provide the best 1C→2C MAE in this simple diagnostic, whereas the decoupled temperature/force representation is best for 2C→1C.

## Recommended research framing

1. Treat this release as a **single-cell multi-profile multi-rate generalization dataset**, not a cross-cell benchmark.
2. Main representations: `V,I`; `V,I,W1,W2`; `V,I,T,F`. Do not use `W1,W2,T,F` simultaneously as independent information.
3. Main splits should be group-based: leave-one-profile-out and cross-rate. Never randomly split rows from the same discharge file.
4. Exclude `dis_cap` and absolute `Time_s` from predictors. `dis_cap` reconstructs the label; `Time_s` is a trajectory-progress proxy.
5. Next modeling step should use causal sequence windows and evaluate MAE, RMSE, MaxAE, R², and SOC-bin errors under the group splits above.
6. A useful paper story is **raw optical sensing vs physics-decoupled thermo-mechanical sensing for robust cross-condition SOC estimation**, followed by representation fusion/gating only if it adds information beyond the deterministic transform.
