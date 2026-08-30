# Causal sequence findings and mechanism diagnostics

This document records the first leakage-safe causal sequence experiments on the public single-cell SiC-18 release. These are **diagnostic results**, not final multi-seed publication numbers.

## Protocol

- Cell: SiC-18 only.
- Profiles: HWFET, LA92, NEDC, NYCC, US06, WLTC.
- Rates: 1C and 2C.
- Causal window: 64 samples; prediction uses only the current and preceding samples.
- Train stride: 4; test stride: 1.
- Normalization: fitted on training-rate data only.
- Forbidden inputs: `SOC`, `dis_cap`, `Time_s`.
- Prototype training: 6 epochs, seed 42.
- Base encoder: small causal TCN (~11.5k trainable parameters for single-view models).

## Cross-rate sequence benchmark

### 1C -> 2C

| Model | MAE | RMSE | R² | MaxAE | Q95 AE |
|---|---:|---:|---:|---:|---:|
| **VI+W** | **0.016316** | **0.024226** | **0.994261** | 0.147449 | **0.053165** |
| RA-TCN v1 | 0.022226 | 0.036300 | 0.987116 | 0.181016 | 0.091429 |
| VI | 0.024261 | 0.046825 | 0.978562 | 0.318418 | 0.107646 |
| VI+W+TF | 0.025755 | 0.035677 | 0.987554 | 0.161992 | 0.077223 |
| VI+TF | 0.028846 | 0.050336 | 0.975226 | 0.251464 | 0.121316 |

Raw dual-FBG coordinates reduce MAE from `2.4261%` SOC for `VI` to `1.6316%`, a relative reduction of approximately **32.7%** in this hard cross-rate direction.

### 2C -> 1C

| Model | MAE | RMSE | R² | MaxAE | Q95 AE |
|---|---:|---:|---:|---:|---:|
| **VI** | **0.006682** | **0.008723** | **0.999179** | **0.042024** | **0.017985** |
| VI+W | 0.008135 | 0.010276 | 0.998860 | 0.042898 | 0.020422 |
| RA-TCN v1 | 0.008323 | 0.010464 | 0.998818 | 0.064844 | 0.020449 |
| VI+TF | 0.010675 | 0.014020 | 0.997879 | 0.063750 | 0.029396 |
| VI+W+TF | 0.013922 | 0.018166 | 0.996438 | 0.092689 | 0.037657 |

The reverse direction is easier and, importantly, adding FBG variables does not improve the clean VI baseline.

## Directional consistency across all six profiles

The cross-rate asymmetry is not driven by a single profile.

For **1C -> 2C**, `VI+W` beats `VI` on all 6/6 held profiles. Approximate relative MAE reductions are:

- HWFET: 51.6%
- LA92: 17.8%
- NEDC: 7.1%
- NYCC: 31.1%
- US06: 40.5%
- WLTC: 40.7%

For **2C -> 1C**, `VI+W` is worse than `VI` on all 6/6 profiles.

This is therefore a systematic **direction-dependent rate-domain effect**, not an average-result artifact.

## First adaptive-gating prototype is rejected as a final method

`RA-TCN v1` used separate electrical, raw-FBG and decoupled-T/F branches and a learned convex raw-vs-physics gate. It did **not** beat the best single-view baseline in either cross-rate direction despite having approximately twice the parameters (`23,498` vs ~`11,545`).

The learned raw-view gate was weakly varying:

- 1C -> 2C: mean raw weight approximately `0.56–0.58` across profiles.
- 2C -> 1C: approximately `0.54`.

Therefore this first gate should be treated as a negative result. The project should not force a novelty claim around this architecture.

## Representation conditioning mechanism

Although `(W1,W2)` and `(T,F)` are exactly invertible coordinate representations, their empirical geometry on the battery trajectories differs strongly.

After feature-wise z-score normalization using training data only:

| Train rate | Representation | Pair correlation | Covariance condition number |
|---|---|---:|---:|
| 1C | W1/W2 | +0.737951 | 6.632 |
| 1C | T/F | -0.982114 | **110.822** |
| 2C | W1/W2 | +0.654631 | 4.791 |
| 2C | T/F | -0.984910 | **131.539** |

Thus the physically interpretable T/F coordinates occupy a highly collinear, poorly conditioned trajectory manifold compared with the raw optical coordinates.

The physical transform itself is also poorly conditioned. For

```text
[T, F]^T = B [W1, W2]^T
```

`cond(B) ≈ 272.15`.

## Train-only whitening experiment

Whitening each optical pair using only the training rate forces the pair covariance condition number to approximately 1 without using target-domain statistics.

### 1C -> 2C

| Model | MAE |
|---|---:|
| **VI+W** | **0.016316** |
| VI+W-white | 0.017160 |
| VI+TF-white | **0.023095** |
| VI+TF | 0.028846 |

Whitening T/F reduces its MAE by approximately **20.0% relative**, but raw W remains clearly stronger.

### 2C -> 1C

| Model | MAE |
|---|---:|
| **VI+TF-white** | **0.008006** |
| VI+W | 0.008135 |
| VI+W-white | 0.008325 |
| VI+TF | 0.010675 |

Whitening T/F reduces its MAE by approximately **25.0% relative** and makes it marginally better than raw W, although it still does not beat the clean `VI` sequence baseline (`0.006682`).

## Interpretation after whitening

The results support, but also refine, the representation-conditioning hypothesis:

1. **Conditioning is real and materially affects performance.** The decoupled T/F coordinates are highly collinear, and train-only whitening produces large gains.
2. **Conditioning is not the whole explanation.** Whitening does not make T/F match raw W in the hard 1C -> 2C direction.
3. The remaining mechanism likely involves a combination of:
   - rate-dependent domain shift;
   - noise propagation through the inverse sensitivity matrix;
   - calibration uncertainty/drift;
   - feature-specific temporal dynamics.
4. In the current data, 1C -> 2C is a stronger extrapolation in current amplitude than 2C -> 1C, helping explain the directional asymmetry.

## Next falsifiable tests

The next experiment is not another network architecture. It is a controlled robustness diagnostic:

- inject matched noise into `W1/W2` before physical decoupling;
- compare raw-W and recomputed-T/F models from exactly the same noisy optical measurements;
- perturb the 2x2 sensitivity calibration matrix and quantify degradation of the T/F branch;
- retain raw W as a calibration-independent reference representation.

Only after these mechanism tests should a final adaptive representation model be designed.
