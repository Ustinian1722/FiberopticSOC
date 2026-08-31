# Q2 publication evidence tables

Status: **FROZEN FOR MANUSCRIPT**

This file contains only the compact evidence intended for the main paper. Full exploratory results remain elsewhere in the repository.

## Table A — Conventional blocked interpolation (T1)

| Model | Params | MAE (%) | RMSE (%) | R2 | Q95-AE (%) |
|---|---:|---:|---:|---:|---:|
| VI-TCN | 11,497 | **0.231** | **0.296** | **0.999904** | **0.581** |
| VI+TF-TCN | 11,545 | 0.311 | 0.412 | 0.999814 | 0.825 |
| GRU | 10,801 | 0.426 | 0.549 | 0.999670 | 1.014 |
| RA-FBG-TCN | 11,545 | 0.482 | 0.593 | 0.999614 | 1.088 |
| LSTM | 14,129 | 0.548 | 0.708 | 0.999450 | 1.303 |
| CNN | 12,081 | 0.564 | 0.738 | 0.999403 | 1.414 |
| Transformer | 20,177 | 0.761 | 0.910 | 0.999090 | 1.584 |

Paper message: all retained models achieve high interpolation accuracy. Electrical-only TCN is strongest in-distribution; the sensing contribution is evaluated under distribution shift rather than claimed as universal interpolation superiority.

## Table B — Matched electrical-OOD optical complementarity

Parameter-matched causal experts:

- VI: `[V, I, 0, 0]`
- VI+W: `[V, I, W1, W2]`

### Cross-rate aggregate

| Direction | VI MAE (%) | VI+W MAE (%) | Relative change |
|---|---:|---:|---:|
| 1C -> 2C | 2.151 | **1.632** | **-24.1%** |
| 2C -> 1C | 0.866 | **0.814** | **-6.0%** |

### 1C -> 2C by electrical-OOD severity

The electrical support envelope is estimated from training current only (0.5th–99.5th percentile). No SOC labels are used in the OOD score.

| Electrical OOD bin | Windows | VI MAE (%) | VI+W MAE (%) | Relative optical gain |
|---|---:|---:|---:|---:|
| ID (0%) | 12,011 | **0.735** | 0.873 | -18.75% |
| OOD 0–25% | 4,296 | 1.550 | **1.470** | +5.17% |
| OOD 25–50% | 2,742 | 2.772 | **2.356** | +15.00% |
| OOD 50–75% | 1,849 | 3.665 | **2.930** | +20.05% |
| OOD 75–100% | 4,464 | 5.529 | **2.846** | **+48.52%** |

Paper message: the usefulness of optical sensing is condition dependent. V/I are sufficient inside the electrical training support, whereas raw FBG observations become increasingly complementary as the operating trajectory moves beyond that support. This is an explanatory diagnostic, not a new gating method claim.

## Table C — Strict cross-rate + unseen-profile T4 (5 seeds)

| Direction | MAE (%) | RMSE (%) | R2 | Q95-AE (%) |
|---|---:|---:|---:|---:|
| 1C -> 2C | 1.795 | 3.037 | 0.985689 | 6.804 |
| 2C -> 1C | **0.806** | **0.988** | **0.998480** | **1.830** |

Overall seed-cluster MAE: **1.301% SOC**, bootstrap 95% CI **0.961–1.677%**.

Paper message: the frozen raw-FBG causal estimator maintains strong accuracy under simultaneous C-rate and unseen-profile shift; extrapolation toward the higher-rate domain is more variable than the reverse direction.

## Table D — Direct wavelength noise robustness

Gaussian perturbation is applied independently to W1 and W2 before source-train normalization; no target-domain transform is re-estimated.

At 2 pm wavelength noise:

- 1C->2C MAE relative increase: approximately **1.67%**;
- 2C->1C MAE relative increase: approximately **4.57%**.

Intermediate 0.5/1 pm levels show small monotonic degradation and should be plotted rather than expanded into another large table.

## Table E — 95% residual split conformal UQ

| Nominal | PICP | MPIW (% SOC) | MIS | Residual quantile (% SOC) |
|---:|---:|---:|---:|---:|
| **95%** | **95.04%** | **2.075%** | 0.02441 | 1.089% |

Paper message: the 95% residual conformal interval achieves empirical coverage essentially equal to the nominal level with a compact average interval width.

## Optional external paragraph only

On the second public multi-cell surface-FBG dataset, under fixed sensor identity and a 0.2C+0.5C -> 1C test, adding S5-relative optical response reduced four-cell mean MAE from **14.92% to 13.42%** and improved **3/4 cells**. Tail-error robustness remained cell dependent.

This is optional supporting evidence only. Do not claim cross-cell generalization.

## Main-paper exclusions

Do not create main-paper tables for Mamba, CrossFormer, ModernTCN, multi-delay models, delta-t, CQR, E1 cross-cell transfer, or E3 WLTP label audit. These are preserved for internal/supplementary use only.