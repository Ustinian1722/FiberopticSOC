# Q2 paper evidence map

Status: **working factual map; methodological decisions are already frozen**

This file separates claims supported by evidence from claims that must not appear in the manuscript.

## 1. Dataset/sensing novelty

### Supported

- The SiC-18 public dataset provides synchronized electrical and dual implanted-FBG signals over six dynamic profiles and two discharge rates.
- The study reuses an existing public sensing dataset to ask a stricter representation/generalization question.

### Do not claim

- novel dataset;
- novel FBG installation;
- first FBG-assisted SOC estimation;
- first use of these SiC-18 data.

The companion Energy 2026 paper already introduced the in-situ sensing/SOC dataset and CNN-GRU-Attention framework.

## 2. Representation contribution

### Supported

- W1/W2 and physics-decoupled T/F are invertibly related representations of the same two optical degrees of freedom in SiC-18.
- Explicit T/F decoupling is physically interpretable but does not create new sensing information.
- Under matched compact-TCN development, raw W1/W2 provides better and more stable unseen-profile cross-rate performance than T/F.
- Whitening, delta-time, and more complex representation/fusion variants do not provide stable source-side gains.
- A coordinate-sensitive CrossFormer can favor T/F on selected cross-rate profiles, but this aggregate gain fails independent 1C source-only confirmation.

### Manuscript wording

Physical interpretability and predictive transferability are distinct objectives. A physically decoupled coordinate is not automatically the optimal learning coordinate under distribution shift.

### Do not claim

- T/F decoupling is physically wrong;
- raw W contains more physical information than T/F;
- raw W is universally superior for every model/task.

## 3. Estimator/architecture contribution

### Supported

- Compact IUW-TCN (11,545 parameters) is the strongest stable estimator after source-only architecture/inductive-bias screening.
- Mamba, CrossFormer, ModernTCN-style large-kernel redesign, local-residual nonstationarity treatment, fixed modality-delay TCN and dynamic multi-delay residual TCN do not pass pre-registered source-side retention gates.
- Dynamic multi-delay selector weights vary by profile, indicating learnable condition-dependent temporal-scale structure, but this does not translate into stable error reduction.

### Paper role

Architecture is a compact causal modeling vehicle, not the central novelty. Advanced time-series models serve as evidence that additional architectural complexity is not automatically beneficial for this sensing task.

### Do not claim

- TCN itself is novel;
- first Mamba/Transformer/attention battery model;
- architecture superiority based on target-guided search.

## 4. Main generalization contribution — SiC-18

### Frozen formal T4

Five seeds, two rate directions, six unseen profiles per direction, 60 clean formal split results.

- 1C -> 2C: mean MAE `0.017954`, RMSE `0.030368`, R2 `0.985689`, Q95-AE `0.068041`.
- 2C -> 1C: mean MAE `0.008057`, RMSE `0.009876`, R2 `0.998480`, Q95-AE `0.018296`.
- overall seed-cluster mean MAE `0.013005`, 95% CI `[0.009607, 0.016767]`.

### Supported claim

The raw-optical compact causal estimator generalizes across simultaneous rate and unseen-profile shifts, with strong directional asymmetry. High-rate-to-low-rate transfer is substantially easier and more stable than low-rate-to-high-rate extrapolation.

### Integrity qualifier

The original representation development screen used held-out 2C profiles in the 1C->2C direction. Therefore same-direction multi-seed T4 is a fully frozen replication/robustness evaluation, not a pristine never-seen-before representation-selection holdout. Reverse 2C->1C carries stronger confirmatory status for the original representation choice.

## 5. Sensor-noise robustness

### Supported

Direct raw-W test-time Gaussian perturbation at 0.5/1/2 pm produces small monotonic degradation.

At 2 pm:

- 1C->2C MAE relative increase ~1.67%;
- 2C->1C MAE relative increase ~4.57%.

### Claim

The frozen estimator is not highly sensitive to small direct Bragg-wavelength perturbations under the tested sensor-noise model.

## 6. UQ

### Supported

- training-side CQR candidate meets coverage/width conditions but does not improve aggregate interval score over residual split conformal;
- final UQ method remains residual split conformal.

### Claim

A simpler residual conformal layer is retained because more complex quantile modeling does not justify its additional complexity under the pre-registered selection criterion.

## 7. External multi-cell FBG evidence — E1

### Frozen negative result

External Hebenbrock four-cell surface-FBG dataset, S5 fixed a priori, zero-point-aligned `S5_rel`, leave-one-cell-out.

Equal-cell mean:

- VI-TCN MAE `0.096603`;
- VI-S5rel-TCN MAE `0.118696`.

S5rel improves only A1 and degrades A2/P1/P2. All five pre-registered E1 support criteria fail.

### Supported claim

A zero-point correction alone is insufficient to make surface-FBG measurements cell-invariant across physically distinct pouch cells/sensor bonds.

This is consistent with the source publication's evidence that adhesive aging/debonding causes sensor-specific changes in strain-transfer sensitivity and can require recalibration.

### Do not claim

- universal cross-cell FBG benefit;
- cell-invariant optical SOC coordinate;
- E1 external validation success.

## 8. External same-cell cross-rate evidence — E2

Status: **PENDING frozen run `33360769760`**.

Frozen direction:

- same physical cell;
- train 0.2C + 0.5C;
- test 1C;
- A1/A2/P1/P2 as four paired replications;
- VI-TCN vs VI-S5rel-TCN;
- no architecture or preprocessing search.

E2 will determine whether the external surface-FBG signal helps rate extrapolation when sensor identity is fixed. It cannot overwrite the E1 cross-cell conclusion.

## 9. Recommended manuscript thesis

The paper should be framed as a **representation and domain-shift study for in-situ optical SOC estimation**, not as a generic new-network paper:

1. direct optical versus physics-decoupled learning coordinates;
2. compact causal modeling chosen through leakage-safe source-side evidence;
3. simultaneous rate/profile OOD validation with multi-seed and sensor-noise robustness;
4. uncertainty reporting without unnecessary UQ complexity;
5. external multi-cell evidence defining the transfer boundary between same-sensor predictive utility and cross-sensor calibration dependence.

The strongest novelty is the combination of rigorous representation evidence and deliberately hard generalization protocols, together with explicit reporting of where direct optical transfer does and does not hold.