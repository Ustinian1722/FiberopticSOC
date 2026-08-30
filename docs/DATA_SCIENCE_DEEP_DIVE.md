# SiC-18 battery data-science deep dive and research reset

Date: 2026-08-31

This note deliberately resets the project order of operations. Model/selector experiments already in the repository are retained as diagnostics, but the paper direction is not chosen from those scores. The dataset is first treated as a battery/measurement-system object: label construction, sensing physics, sampling, condition shift, observability and failure modes are audited before method design.

## 1. Dataset scope and scientific boundary

The released SiC-18 dataset contains one silicon-based lithium-ion pouch cell measured under six dynamic drive profiles (HWFET, LA92, NEDC, NYCC, US06, WLTC) and two discharge-rate settings (1C and 2C), for 12 trajectories and 68,086 rows in total. The associated Energy paper describes a 2.5 Ah SiOx/C pouch cell with an Armored FBG and a Bare FBG implanted at the center of the sixth-layer anode surface.

This is therefore a **single-cell, multi-condition** dataset. It can support strong same-cell cross-profile and cross-rate conclusions, but by itself it cannot support universal cross-cell or cross-chemistry claims.

Primary source: Chen Ling et al., *In-situ data-driven high-precision SOC estimation for silicon-based lithium-ion batteries*, Energy 349 (2026) 140609, DOI: 10.1016/j.energy.2026.140609.

Dataset: Mendeley Data, DOI: 10.17632/ft6rtwt8vm.1.

## 2. SOC label construction and leakage boundary

Across 11 of the 12 files,

`SOC = 1 - dis_cap / max(dis_cap)`

holds to numerical precision. HWFET_1C differs only by about 2.77e-5 maximum absolute SOC, consistent with rounding/processing.

Therefore `dis_cap` is a direct target reconstruction variable and must never be used as a predictor. Absolute `Time_s` is also excluded from the main estimator because every workbook is a single monotonic full-discharge trajectory and absolute time is a strong trajectory-progress proxy.

The terminal discharge-capacity values correspond to approximately 1.9800–2.0106 Ah (mean about 1.9993 Ah) when divided by 3600. More importantly, row-to-row discharge-capacity increments satisfy

`Delta dis_cap ~= - I_i * Delta t_i`

with pooled correlation about 0.99982 and R2 about 0.99964. Numerically this means `dis_cap` behaves like ampere-seconds / coulomb-count accumulation rather than Wh. The Mendeley description currently labels `dis_cap` as Wh while also saying SOC is obtained by ampere-hour integration; the numerical data are inconsistent with a Wh interpretation. For this project, `dis_cap` is treated only as the source used to construct the SOC label and is prohibited from estimator inputs.

## 3. Sampling is not uniformly 1 Hz

Although the median timestamp interval is 1 s in every file, the released rows are not uniformly spaced:

- 11.8%–30.9% of intervals, depending on trajectory, are longer than 1 s.
- Maximum observed gaps range up to 40 s.
- A small number of duplicate timestamps (`Delta t = 0`) also occur in several trajectories.
- The capacity increments remain consistent with current multiplied by the actual timestamp gap.

Consequently, a conventional sequence model that treats a fixed **number of rows** as a fixed temporal window is physically inconsistent: a 64-row window represents different elapsed time in different portions of the dataset and in different drive profiles.

This should be handled explicitly in the final protocol. Candidate solutions are causal elapsed-time encoding or a fixed-duration causal representation. Future-value interpolation must not be introduced.

## 4. Dual-FBG physics: W and T/F are two coordinate systems of the same information

The released columns satisfy the following mapping essentially exactly over all 68,086 samples:

`W1 = 0.0208 T + 0.00054 F`

`W2 = 0.0254 T + 0.00085 F`

where W1/W2 are wavelength-shift signals in nm, T is the released decoupled temperature feature and F is the released deformation-force feature in N.

Equivalently,

`[W1, W2]^T = K [T, F]^T`

with

`K = [[0.0208, 0.00054], [0.0254, 0.00085]]`.

The inverse recovered directly from the release is

`T = 214.4298688 W1 - 136.2260343 W2`

`F = -6407.6690212 W1 + 5247.2250252 W2`.

This confirms the physical decoupling interpretation in the companion mechanism paper. It also establishes an important modeling rule: W1/W2 and T/F must not be described as four independent information channels. They span two FBG degrees of freedom.

Primary mechanism source: Chen Ling et al., *Optical fiber sensors reveal in-situ thermo-mechanical behaviors inside silicon-based lithium-ion batteries*, Journal of Power Sources 676 (2026) 239852, DOI: 10.1016/j.jpowsour.2026.239852.

## 5. The decoupling problem is numerically ill-conditioned

The sensitivity matrix has:

- determinant: about 3.964e-6
- 2-norm condition number: about **272.15**
- angle between the temperature-sensitivity and force-sensitivity vectors: only about **6.89 degrees**

The two sensing directions are therefore close to collinear. The inverse transform is mathematically valid but can strongly amplify wavelength noise.

For independent wavelength perturbations with characteristic scale 1 pm (0.001 nm), the row norms of `K^-1` imply approximately:

- 0.254 degC characteristic temperature perturbation
- 8.28 N characteristic force perturbation

The released data show the same order of magnitude. During consecutive 1-s, zero-current observations, the median absolute wavelength step is about 1 pm, while the median absolute step after decoupling is about **0.36 degC** for T and **11.95 N** for F.

This is not a reason to reject T/F. It is a reason to distinguish **physical interpretability** from **numerical conditioning** and to model decoupling uncertainty rather than treating the derived channels as noise-free states.

General FBG literature also treats conditioning of the sensitivity matrix as central to stable temperature/strain discrimination; poorly separated sensitivity directions are known to amplify demodulation noise.

## 6. Why the mechanical signal is scientifically valuable

The force signal is not merely another highly correlated feature. Its dynamic behavior differs qualitatively from terminal voltage.

For each trajectory, a fifth-order SOC-only curve was fitted for descriptive analysis (not for model training). Across the 12 files:

- Voltage SOC-curve R2: about 0.9822
- Force SOC-curve R2: about **0.9937**
- Temperature SOC-curve R2: about 0.9903

After removing the smooth SOC trend, the absolute correlation of the remaining signal residual with instantaneous current is:

- Voltage: about **0.817**
- Force: about **0.059**
- Temperature: about **0.064**

Thus terminal voltage is strongly distorted by instantaneous load/polarization, while the decoupled mechanical response is much more load-invariant on the same trajectories. This gives the FBG mechanical state a defensible role as an **internal SOC anchor/complement**, rather than simply an extra feature concatenated to V/I.

The literature is consistent with this mechanism. Mechanical pressure/expansion/strain has repeatedly been linked to lithium content and SOC, and the Si/SiOx anode is especially relevant because lithiation induces pronounced volume change and stress evolution.

Relevant examples:

- *Lithium-ion battery expansion mechanism and Gaussian process regression based state of charge estimation with expansion characteristics*, Energy 292 (2024) 130541, DOI: 10.1016/j.energy.2024.130541.
- *Mechanical stress-based state-of-charge estimation for lithium-ion batteries via deep learning techniques*, Energy 326 (2025) 136216, DOI: 10.1016/j.energy.2025.136216.
- *In Operando Monitoring the Stress Evolution of Silicon Anode Electrodes during Battery Operation via Optical Fiber Sensors*, Small (2024), DOI: 10.1002/smll.202311299.
- *Real-time electrochemical-strain distribution and state-of-charge mapping via distributed optical fiber for lithium-ion batteries*, Journal of Power Sources (2025), PII S0378775324014782.

## 7. Force is also comparatively stable under the 1C/2C condition shift

For each profile, signals were interpolated on a common SOC grid and the average absolute 1C-vs-2C difference was normalized by the pooled signal span. The normalized rate-shift magnitudes were approximately:

- Force: **2.96%**
- W2: 3.82%
- Temperature: 5.13%
- Voltage: 6.45%
- W1: 8.45%
- Current: 33.38%

This supports a precise claim: in this cell, force is a comparatively rate-stable internal coordinate of discharge state, whereas terminal voltage is more rate-sensitive.

This does **not** prove force is universally invariant across cells, aging states or installation conditions.

## 8. Why W1 looks weak and W2 looks strong

The forward equations explain the observed correlations.

During discharge, the released temperature feature generally rises while deformation force becomes strongly negative. In W1, the positive thermal term and negative mechanical term substantially cancel. In W2, the larger force coefficient produces a more monotonic net shift. This explains why pooled W1-SOC correlation is much weaker than W2-SOC correlation without implying that W1 is a bad or defective sensor.

This cancellation is another reason that raw-wavelength and decoupled physical representations should be compared as alternative coordinate systems rather than treated as independent modalities.

## 9. Literature positioning as of August 2026

The field has moved quickly. A new paper cannot credibly claim novelty from "FBG + deep network" alone.

Directly overlapping recent work includes:

1. Chen Ling et al., Energy 2026: implanted FBG + electrical/thermomechanical features + feature engineering/noise augmentation + CNN-GRU-Attention; reported SOC RMSE 0.635%.
2. *Estimation of state-of-charge for lithium-ion batteries based on simultaneous internal strain and temperature monitoring by fiber optic sensors*, Journal of Energy Storage 133 (2025) 117969: dual-FBG decoupling + CNN-Transformer; reported >40% improvement after adding internal parameters.
3. *State-of-charge estimation for lithium-ion batteries using implanted dual-Fiber Bragg Grating sensing and Local-to-Global Temporal Former with temperature-stress feature decoupling*, Journal of Energy Storage 180 (2026) 124293: implanted dual FBG + T/F + Local-to-Global Temporal Former; includes unseen-driving-cycle validation.
4. *Adaptive estimation of battery pack state of charge with optical fibre strain measurements*, Applied Energy 407 (2026) 127330: optical strain + GPR-based adaptive UKF, explicitly adapting measurement/process covariance according to sensor reliability.
5. *Enhancing reliability in electrified transportation: A conformalized quantile regression framework for battery state-of-charge uncertainty quantification*, Journal of Power Sources 666 (2026) 239123: SOC point model + conformalized quantile regression with distribution-free intervals.

Therefore the project should not be centered on another CNN/GRU/Transformer/Mamba block, a generic filter attachment, or generic conformal UQ. Those can be supporting components only.

## 10. Revised paper thesis suggested by the data

The strongest data-grounded direction is:

> **reliability-aware SOC estimation from an internal mechanical anchor whose information is physically useful but whose dual-FBG decoupling is noise-sensitive.**

The method should preserve an electrical path as the primary estimator and use FBG information as a bounded correction/auxiliary state, with an explicit fallback when the optical information is unreliable or outside its source support.

A defensible method chain is:

1. **Leakage-safe electrical backbone:** only causal V/I history; no `dis_cap`, no absolute `Time_s`, no reconstructed SOC/coulomb-count feature.
2. **Time-consistent preprocessing:** explicitly account for the irregular timestamps instead of silently treating each row as 1 s.
3. **Sensor-physics representation:** compare raw W, direct T/F inversion, and a noise-aware/regularized physical representation. Do not fuse W+T+F as independent information.
4. **Bounded mechanical correction:** optical/mechanical branch corrects the electrical estimate rather than replacing it everywhere; correction can collapse to zero.
5. **Reliability/uncertainty mechanism:** propagate wavelength/decoupling uncertainty or use source-domain calibration to determine when the mechanical correction should be trusted.
6. **Optional post-hoc UQ:** only after the point-estimation/fusion mechanism is justified; UQ should quantify residual SOC uncertainty, not serve as the only novelty.

The central paper story should be the tension between **mechanical observability** and **sensor-decoupling reliability**, not raw model complexity.

## 11. Recommended experimental protocol

Primary task: **cross-rate + unseen-profile generalization**.

For each direction (1C -> 2C and 2C -> 1C), hold out one drive profile completely. Train on the five remaining source-rate profiles and test only on the corresponding held-out profile in the target rate. Repeat for all six profiles, producing 12 strict splits.

All normalization, whitening, regularization coefficients, early stopping, reliability thresholds and UQ calibration must be determined from source-domain training/validation data only.

Required controls:

- VI electrical baseline
- optical-only diagnostic (not necessarily a production estimator)
- VI + raw W
- VI + direct T/F
- VI + conditioned/regularized optical representation
- bounded fusion without reliability control
- reliability-aware fusion
- irregular-time ablation
- wavelength-noise perturbation at realistic pm-scale amplitudes
- sensitivity-matrix perturbation / calibration uncertainty stress test
- per-SOC-bin error and high-load-vs-low-load error
- paired split-level and seed-level statistics

The existing NestedCal99/AnyOOD experiments remain useful as diagnostics of source-support gating, but should not define the paper before the above sensor-physics analyses are completed.

## 12. Claim boundary

What this dataset can support strongly:

- same-cell generalization across drive profiles and discharge rates;
- internal mechanical information can complement electrical signals under dynamic load;
- representation conditioning and dual-FBG decoupling noise matter;
- strict leakage-safe evaluation under condition shift.

What it cannot establish by itself:

- cross-cell universality;
- cross-chemistry universality;
- full-life aging robustness;
- production BMS reliability without additional cells and environmental conditions.
