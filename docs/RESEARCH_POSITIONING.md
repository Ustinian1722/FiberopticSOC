# Research positioning: representation-aware dual-FBG SOC estimation

## Direct prior work that constrains novelty

### 1. Raw Bragg wavelength directly used for SOC

A 2023 *Batteries* paper, **Fiber-Bragg-Grating-Based Sensor System to Measure Battery State of Charge Based on a Machine Learning Model**, estimated SOC directly from monitored Bragg reflection wavelengths and explicitly discussed avoiding temperature compensation / explicit separation of strain and temperature for the black-box SOC estimator.

DOI: `10.3390/batteries9100508`

Implication: **using raw FBG wavelengths for SOC is not itself novel**.

### 2. Public SiC-18 dataset + CGA hybrid model

Chen Ling et al., **In-situ data-driven high-precision SOC estimation for silicon-based lithium-ion batteries**, *Energy* 349 (2026), 140609, combines in-situ FBG-derived thermo-mechanical sensing with electrical measurements, feature engineering, noise augmentation, and a CNN-GRU-Attention (CGA) model. The paper reports a best SOC RMSE of approximately `0.635%`.

DOI: `10.1016/j.energy.2026.140609`

Implication: **FBG-assisted SOC, feature engineering, noise augmentation, CNN/GRU/attention fusion are already claimed on this public data line**.

### 3. Physical basis of thermo-mechanical decoupling

Chen Ling et al., **Optical fiber sensors reveal in-situ thermo-mechanical behaviors inside silicon-based lithium-ion batteries**, *Journal of Power Sources* 676 (2026), 239852, implants Armored FBG and Bare FBG sensors, performs in-situ temperature/force sensitivity calibration, and uses the differential sensitivities to decouple internal temperature and deformation force.

DOI: `10.1016/j.jpowsour.2026.239852`

Implication: the `temperature` and `force` columns in the SiC-18 release are **physics-decoupled quantities**, not arbitrary learned features.

### 4. Very recent direct competitor: dual-FBG + Local-to-Global Temporal Former

Yuqian Fan et al., **State-of-charge estimation for lithium-ion batteries using implanted dual-Fiber Bragg Grating sensing and Local-to-Global Temporal Former with temperature-stress feature decoupling**, *Journal of Energy Storage* 180 (2026), 124293, uses decoupled internal `T` and `F` with current/voltage and an LTF temporal architecture. It reports multi-temperature and unseen-driving-condition experiments on multi-cell implanted-FBG datasets.

DOI: `10.1016/j.est.2026.124293`

Implication: **"dual-FBG decoupling + advanced temporal model + unseen driving condition" is no longer a sufficient novelty claim**.

## Empirical fact unique to the released SiC-18 tables

For all 68,086 released samples, the two coordinate systems satisfy an essentially exact, nonsingular linear transformation:

```text
T = 214.429868819374 * W1 - 136.226034308779 * W2
F = -6407.66902119071 * W1 + 5247.22502522705 * W2
```

with `R² = 1` and numerical residuals around `1e-13`. The transform determinant is non-zero, so `(W1,W2)` and `(T,F)` are information-equivalent coordinate representations of the same two FBG degrees of freedom.

This means a model using both `W1,W2,T,F` does **not** receive four independent sensing modalities. Any performance difference between raw and decoupled representations must arise from feature geometry, scaling, conditioning, inductive bias, optimization, noise propagation, or domain shift.

## Defensible research gap

The primary working hypothesis is therefore:

> **Physics decoupling is not an information-gain operation; it is a coordinate transformation. Its practical value for SOC estimation should be studied as a representation-conditioning problem under domain shift. A representation-adaptive estimator may exploit the robustness advantages of raw optical coordinates and physics-decoupled thermo-mechanical coordinates without falsely treating them as independent modalities.**

This creates a research direction distinct from both the 2026 CGA paper and the 2026 LTF paper.

## Locked experimental principles

1. `SOC`, `dis_cap`, and absolute `Time_s` are forbidden model inputs.
2. No random row split from the same discharge trajectory is accepted as the main result.
3. Primary public-data stress tests are:
   - same-rate leave-one-driving-profile-out;
   - `1C -> 2C` cross-rate;
   - `2C -> 1C` cross-rate.
4. Core input controls:
   - `VI`;
   - `VI+W1+W2` (raw optical representation);
   - `VI+T+F` (physics-decoupled representation);
   - `VI+W1+W2+T+F` only as an explicit redundancy control;
   - representation-adaptive model using both views with a constrained selector/gate.
5. Normalization statistics are fit on training groups only.
6. Sequence windows are causal: only samples at or before the prediction instant are allowed.
7. Report MAE, RMSE, R², MaxAE, Q95 absolute error, per-profile error, per-SOC-bin error, and parameter count.

## Required controls before any novelty claim

- Parameter-matched baseline against the representation-adaptive model.
- Gate ablation: fixed raw-only, fixed TF-only, fixed 0.5 mixture, learned gate.
- Calibration perturbation: perturb the W->TF sensitivity matrix to test whether raw optical input is more robust to calibration error.
- Sensor-noise perturbation: matched noise injected before decoupling to quantify noise amplification through the inverse sensitivity matrix.
- Window-length sensitivity.
- Multiple random seeds for final results.
- Full leave-one-profile-out testing after the cross-rate prototype is validated.

## Scope limitation

The public `SiC-18` release is a **single-cell** dataset. It can support strong cross-profile and cross-rate conclusions, but it cannot by itself establish cross-cell universality. Any final paper should either keep the claim explicitly single-cell/domain-shift focused or add a genuinely independent multi-cell dataset.
