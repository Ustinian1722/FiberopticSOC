# FBG noise and calibration robustness findings

This diagnostic tests a specific mechanism behind the difference between raw optical coordinates `(W1,W2)` and physically decoupled `(T,F)`. The values below are **controlled stress tests**, not claims about the published interrogator's actual noise or calibration tolerance.

## Physical error propagation through the decoder

The released SiC-18 tables imply the sensitivity relation

```text
[W1, W2]^T = A [T, F]^T

A = [[0.0208, 0.00054],
     [0.0254, 0.00085]]
```

and therefore

```text
[T, F]^T = inv(A) [W1, W2]^T.
```

The inverse sensitivity matrix has condition number approximately `272.15`.

If the two wavelength channels each contain independent Gaussian noise with standard deviation `1 pm = 0.001 nm`, linear propagation through the decoder gives approximately:

- temperature noise standard deviation: **0.254 °C**;
- force noise standard deviation: **8.282 N**.

This shows that decoupling can strongly magnify raw optical perturbations in physical units, especially for force.

## Model-level wavelength-noise sweep

Two parameter-matched causal TCNs were trained on clean data:

- `VI+W`: voltage, current, raw W1/W2;
- `VI+TF`: voltage, current, physically decoupled T/F.

At test time, matched Gaussian noise was added to W1/W2 **before** decoupling. The T/F inputs were then recomputed from exactly the same noisy W signals. The robustness diagnostic uses causal windows of 64 samples and test stride 4 for computational efficiency.

### 1C -> 2C MAE

| W noise (pm) | VI+W | VI+TF |
|---:|---:|---:|
| 0.00 | **0.016293** | 0.028753 |
| 0.10 | **0.016295** | 0.028753 |
| 0.25 | **0.016298** | 0.028760 |
| 0.50 | **0.016318** | 0.028790 |
| 1.00 | **0.016406** | 0.028843 |
| 2.00 | **0.016776** | 0.029177 |

### 2C -> 1C MAE

| W noise (pm) | VI+W | VI+TF |
|---:|---:|---:|
| 0.00 | **0.008175** | 0.010682 |
| 0.10 | **0.008177** | 0.010686 |
| 0.25 | **0.008190** | 0.010699 |
| 0.50 | **0.008228** | 0.010758 |
| 1.00 | **0.008436** | 0.010969 |
| 2.00 | **0.009108** | 0.011791 |

### Noise conclusion

The physical decoder clearly amplifies wavelength noise in physical units, but **ordinary zero-mean wavelength noise is not the main explanation for the large clean-data SOC gap between W and TF**. Across 0.1–2 pm, both SOC models degrade smoothly and by comparable relative amounts. The representation gap already exists at zero added noise.

This falsifies the overly simple hypothesis that decoder noise amplification alone explains why raw W can generalize better.

## Calibration-matrix uncertainty sweep

The stronger robustness issue is systematic calibration uncertainty. Each of the four sensitivity coefficients in `A` was independently perturbed by zero-mean multiplicative Gaussian error, and the test T/F signals were recomputed using the resulting imperfect inverse matrix. The SOC network itself was not retrained.

### 1C -> 2C, VI+TF MAE

| Sensitivity coefficient RMS error | Mean MAE | Std | 90th percentile |
|---:|---:|---:|---:|
| 0% | 0.028753 | — | 0.028753 |
| 0.25% | 0.029559 | 0.000921 | 0.030515 |
| 0.5% | 0.028591 | 0.001262 | 0.029630 |
| 1% | 0.029537 | 0.003155 | 0.033334 |
| 2% | 0.033562 | 0.009237 | 0.040503 |
| 5% | **0.053722** | **0.027275** | **0.080256** |

At 5% perturbation the worst of the ten controlled trials reached MAE `0.115471`.

### 2C -> 1C, VI+TF MAE

| Sensitivity coefficient RMS error | Mean MAE | Std | 90th percentile |
|---:|---:|---:|---:|
| 0% | 0.010682 | — | 0.010682 |
| 0.25% | 0.010177 | 0.000556 | 0.010839 |
| 0.5% | 0.011021 | 0.001220 | 0.012217 |
| 1% | 0.011517 | 0.002686 | 0.014208 |
| 2% | 0.012039 | 0.003928 | 0.018540 |
| 5% | **0.029219** | **0.012862** | **0.038414** |

At 5% perturbation the worst trial reached MAE `0.059695`.

Small perturbation levels are not strictly monotonic because only ten random calibration realizations were used and some coefficient combinations partially cancel. The scientifically relevant observation is the rapidly increasing variance and upper-tail error as calibration uncertainty grows.

## Calibration conclusion

Unlike random wavelength noise, calibration uncertainty introduces a **systematic representation-specific vulnerability**:

- raw `W1/W2` can be consumed directly and is independent of the T/F sensitivity inversion;
- T/F gains physical interpretability but inherits dependence on a calibrated, relatively ill-conditioned inverse transformation;
- modest coefficient error may be tolerable, but larger calibration drift can cause substantial and highly variable SOC degradation.

This creates a concrete engineering trade-off between **physical interpretability** and **calibration robustness**.

## Refined mechanism picture

The accumulated evidence now supports three distinct effects:

1. **Coordinate conditioning:** T/F is much more collinear than W on the observed trajectories; train-only whitening significantly improves T/F.
2. **Systematic calibration sensitivity:** the decoupled representation can become fragile when the sensitivity matrix drifts or is estimated imperfectly.
3. **Rate-domain asymmetry:** even after conditioning correction, raw W remains superior in the hard 1C -> 2C extrapolation, so domain shift is an additional mechanism.

Random wavelength noise alone is not sufficient to explain the representation difference.

## Implication for the final method

A final model should not simply concatenate `W1,W2,T,F`. It should preserve the two-dimensional optical information while explicitly addressing conditioning and calibration reliability. Candidate designs should be parameter-matched and tested against:

- raw W only;
- physical TF only;
- whitened TF;
- electrical VI only;
- calibration perturbation and wavelength-noise sweeps;
- cross-rate and leave-one-profile-out domain shifts.
