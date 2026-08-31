# Q2 paper-completion experiment results

Status: **FROZEN FOR MANUSCRIPT**

Canonical workflow run: `33404899282`

Release commit used by the canonical run: `490b63fccf5ef08bfe70d8f72376527474a01714`

These experiments were added only to complete the conventional battery-paper evidence chain. They do **not** reopen architecture search or change the previously frozen RA-FBG-TCN design.

## 1. Frozen protocol

- Task: simultaneous C-rate shift + unseen-profile shift.
- Profiles: HWFET, LA92, NEDC, NYCC, US06, WLTC.
- Directions: 1C→2C and 2C→1C.
- For each held-out profile, training uses the other five profiles at the source C-rate and testing uses the held-out profile at the opposite C-rate.
- Window: 64 samples; training stride: 4; test stride: 1.
- Normalization: source-training data only.
- Training budget: the same split-specific source-only frozen epoch plan used by formal T4 (`config/q2_frozen_epoch_plan.csv`).
- Target data are not used for normalization, epoch selection, or model rescue.

### Strict backbone benchmark

All models use the same V/I/W1/W2 inputs, split, normalization, source-only frozen epoch, and seed 42.

Models: CNN, GRU, LSTM, Transformer, CGA-Matched, DualTCN-Transformer, RA-FBG-TCN.

### Strict input ablation

The architecture is parameter-matched TCN in all cases (11,545 trainable parameters). Only the input representation changes:

- VI: voltage/current with two zero channels for parameter matching.
- VI+TF: voltage/current + decoupled temperature/force coordinates.
- VI+W: voltage/current + native W1/W2.

Seeds: 0, 1, 2, 3, 4. Each direction therefore contains 30 seed×profile paired observations per input condition.

## 2. Strict backbone benchmark

### 1C→2C

| Model | Parameters | MAE (% SOC) | RMSE (% SOC) | R² | Q95-AE (% SOC) |
|---|---:|---:|---:|---:|---:|
| GRU | 10,801 | **1.059** | **1.367** | **0.998004** | **2.836** |
| LSTM | 14,129 | 1.122 | 1.527 | 0.997533 | 3.329 |
| RA-FBG-TCN | 11,545 | 1.163 | 1.757 | 0.996494 | 4.048 |
| CNN | 12,081 | 1.274 | 1.739 | 0.996584 | 3.553 |
| DualTCN-Transformer | 64,705 | 1.288 | 1.946 | 0.996098 | 4.382 |
| Transformer | 20,177 | 2.025 | 2.944 | 0.989330 | 6.048 |
| CGA-Matched | 19,314 | 2.035 | 3.148 | 0.989292 | 7.106 |

RA-FBG-TCN ranks third by MAE in the harder low-to-high-rate transfer. It remains substantially better than CNN, DualTCN-Transformer, Transformer, and the CGA-style matched comparator while retaining only 11.5k parameters.

### 2C→1C

| Model | Parameters | MAE (% SOC) | RMSE (% SOC) | R² | Q95-AE (% SOC) |
|---|---:|---:|---:|---:|---:|
| RA-FBG-TCN | 11,545 | **0.565** | **0.714** | **0.999346** | **1.404** |
| DualTCN-Transformer | 64,705 | 0.732 | 0.876 | 0.999034 | 1.572 |
| CGA-Matched | 19,314 | 0.749 | 0.935 | 0.998917 | 1.825 |
| CNN | 12,081 | 0.757 | 0.944 | 0.998831 | 1.833 |
| GRU | 10,801 | 0.762 | 0.929 | 0.998835 | 1.719 |
| LSTM | 14,129 | 0.848 | 1.026 | 0.998502 | 1.886 |
| Transformer | 20,177 | 1.128 | 1.361 | 0.997745 | 2.495 |

RA-FBG-TCN ranks first by MAE, RMSE, R², and Q95-AE in this direction.

### Both directions combined

| Model | MAE (% SOC) | RMSE (% SOC) | R² | Q95-AE (% SOC) |
|---|---:|---:|---:|---:|
| RA-FBG-TCN | **0.864** | 1.235 | 0.997920 | 2.726 |
| GRU | 0.910 | **1.148** | **0.998420** | **2.277** |
| LSTM | 0.985 | 1.276 | 0.998018 | 2.607 |
| DualTCN-Transformer | 1.010 | 1.411 | 0.997566 | 2.977 |
| CNN | 1.015 | 1.342 | 0.997708 | 2.693 |
| CGA-Matched | 1.392 | 2.042 | 0.994105 | 4.465 |
| Transformer | 1.576 | 2.152 | 0.993537 | 4.271 |

The correct manuscript claim is therefore **not** that RA-FBG-TCN is best on every metric or every direction. The defensible result is that it gives the **lowest mean MAE across the 12 strict cross-rate/unseen-profile splits**, is best in 2C→1C, remains top-three in the harder 1C→2C direction, and does so with a compact 11.5k-parameter architecture. GRU gives lower pooled RMSE/Q95 and higher pooled R².

## 3. Five-seed strict input ablation

| Direction | Input | MAE (% SOC) | RMSE (% SOC) | R² | Q95-AE (% SOC) |
|---|---|---:|---:|---:|---:|
| 1C→2C | VI | 2.162 | 3.824 | 0.977020 | 8.780 |
| 1C→2C | VI+TF | **1.770** | 3.051 | 0.984593 | **6.751** |
| 1C→2C | VI+W | 1.795 | **3.037** | **0.985689** | 6.804 |
| 2C→1C | VI | **0.464** | **0.578** | **0.999587** | **1.127** |
| 2C→1C | VI+TF | 0.493 | 0.605 | 0.999569 | 1.153 |
| 2C→1C | VI+W | 0.806 | 0.988 | 0.998480 | 1.830 |

### Paired evidence

For 1C→2C, VI+W reduces mean MAE from 2.162% to 1.795%, an absolute gain of 0.367 percentage points and a relative reduction of approximately 17.0%. VI+W wins 20 of 30 paired seed×profile comparisons. The seed-cluster bootstrap 95% CI for the absolute MAE gain is +0.004 to +0.907 percentage points. A paired t-test gives p=0.0288 and Wilcoxon signed-rank test gives p=0.0208.

VI+TF also improves the difficult 1C→2C transfer (1.770% MAE). VI+W and VI+TF are statistically comparable in this formal multi-seed ablation (paired t-test p=0.746; Wilcoxon p=0.903). Therefore the manuscript must not claim that native wavelength coordinates universally or formally outperform T/F.

For 2C→1C, VI is best (0.464% MAE). VI+W loses to VI in 24 of 30 paired comparisons. This direction is consistent with the current-support interpretation: source training at 2C already covers a wider electrical excitation range, reducing the need for the auxiliary optical channels.

## 4. Final interpretation for the paper

The paper should use the following evidence hierarchy:

1. **Conventional accuracy:** all principal models achieve high accuracy under blocked interpolation; electrical-only sensing can be strongest when the operating distribution is well covered.
2. **Main model benchmark:** RA-FBG-TCN is a compact, competitive cross-condition estimator. It gives the lowest pooled MAE across both strict transfer directions and the best 2C→1C result, while remaining top-three in 1C→2C.
3. **FBG ablation:** dual-FBG information significantly improves the difficult 1C→2C low-to-high-rate transfer, but is not universally beneficial in the easier reverse direction.
4. **Representation:** the development-stage matched screen selected native W1/W2; the formal five-seed ablation shows W and T/F are comparable in difficult forward transfer. Native W is retained because it is directly measured, avoids an additional inversion, and was frozen before formal testing—not because it universally dominates T/F.
5. **Mechanism support:** electrical-OOD stratification explains the direction dependence and shows optical gain increasing to 48.52% in the most shifted windows.
6. **Reliability:** multi-seed reporting, pm-scale wavelength-noise tests, and residual conformal calibration complete the engineering evidence chain.

No additional SiC-18 architecture search or target-guided rescue experiment is justified after this result freeze.