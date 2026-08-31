# Q2 final T1 paper results

Status: **FINAL / PAPER USE**

Canonical workflow: `Q2 final T1 paper benchmark`, run `33362802803`, head SHA `ec3318402c31e8c4b585e65a04e78c8057218354`, aggregate artifact `q2-final-t1-paper-summary` (artifact id `9747384099`).

Protocol: blocked mixed-condition interpolation (`T1_block16_phase_rotated_guard_window`), seed 42, causal window 64, train stride 2, evaluation stride 1, source/training-only normalization, validation-based early stopping, reserved calibration set for UQ.

## Table 3 candidate — conventional SOC model comparison

| Model | Params | Best epoch | MAE (%) | RMSE (%) | R2 | Q95-AE (%) | MaxAE (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **VI-TCN** | 11,497 | 39 | **0.231** | **0.296** | **0.999904** | **0.581** | **1.428** |
| VI+TF-TCN | 11,545 | 29 | 0.311 | 0.412 | 0.999814 | 0.825 | 2.008 |
| GRU | 10,801 | 49 | 0.426 | 0.549 | 0.999670 | 1.014 | 1.706 |
| **RA-FBG-TCN (VI+W1/W2)** | 11,545 | 23 | **0.482** | **0.593** | **0.999614** | **1.088** | 2.438 |
| LSTM | 14,129 | 49 | 0.548 | 0.708 | 0.999450 | 1.303 | 2.202 |
| CNN | 12,081 | 48 | 0.564 | 0.738 | 0.999403 | 1.414 | 2.832 |
| Transformer | 20,177 | 17 | 0.761 | 0.910 | 0.999090 | 1.584 | 2.492 |

All percentages above are the normalized SOC errors multiplied by 100.

### Manuscript interpretation

T1 is a conventional within-dataset/interpolation benchmark, not the experiment used to establish the value of FBG sensing under transfer. The electrical-only TCN is strongest in this relatively easy regime, indicating that V/I already provide highly informative SOC coordinates when the evaluation distribution remains well represented by the training set.

RA-FBG-TCN nevertheless achieves high absolute accuracy (MAE 0.482%, RMSE 0.593%, R2 0.999614) with only 11.5k parameters and outperforms the conventional CNN/GRU/LSTM/Transformer baselines in the retained comparison except for GRU. The principal value of optical sensing is therefore presented in the subsequent electrical-OOD and strict cross-condition transfer experiments rather than claimed as universal in-distribution superiority.

## 95% residual split conformal UQ

The final point model `RA-FBG-TCN` uses the reserved T1 calibration block only after point-model fitting/early stopping.

| Nominal coverage | PICP | MPIW (%) | Mean interval score | Residual quantile (%) | n_cal | n_test |
|---:|---:|---:|---:|---:|---:|---:|
| **95%** | **0.9504** | **2.075** | **0.02441** | **1.089** | 4,238 | 4,174 |

The 95% prediction interval achieves empirical coverage of 95.04%, essentially matching its nominal level, with mean width 2.075% SOC. This is the primary UQ result for the manuscript.

A secondary 90% interval is retained in the artifact/supplementary record (PICP 85.31%) but is not used as the main-paper UQ claim.

## Paper-use rule

- Section 4.2: use the seven-model table above as conventional benchmark evidence.
- Do **not** claim that FBG universally improves same-distribution point accuracy.
- Section 4.3: establish the sensing contribution through parameter-matched electrical-OOD complementarity and representation analysis.
- Section 4.5: use only the 95% residual conformal row as the principal UQ result.
- No new T1 baseline, architecture, epoch, representation, or UQ search is permitted after this result.