# FiberopticSOC research decision log

This file separates exploratory observations from results that are sufficiently controlled for manuscript use.

## Locked data and leakage policy

- Dataset: 12 dynamic-condition workbooks = 6 profiles × {1C, 2C}.
- Allowed causal inputs: voltage, current, optical wavelengths, and train-only transforms of these signals.
- Forbidden SOC shortcuts: `SOC`, `dis_cap`, `Time_s` as model inputs; no full-test-cycle statistics or target-informed normalization.
- All normalization, whitening, support envelopes, and selector thresholds must be estimated from training data only.

## Validated structural findings

### 1. Same-rate unseen-profile generalization does not justify unconditional optical fusion

Fair-initialization 12-split LOPO audit (6 profiles at each rate):

- VI mean MAE: 0.008137
- OOD-selective VI-or-W: 0.008297
- VI+W: 0.011272
- VI+TF: 0.011594
- VI+TF-white: 0.012211

Interpretation: under ordinary same-rate profile shift, electrical-only VI is the strongest control and optical fusion can cause negative transfer. Do not claim that optical sensing universally improves unseen driving-profile SOC estimation.

### 2. The useful optical regime is asymmetric low-rate → high-rate extrapolation

In the strict protocol `1C five profiles -> 2C unseen sixth profile`, the electrical current distribution substantially leaves the 1C training support. In the reverse `2C -> 1C` direction, the test current remains inside the 2C training support and the support gate remains off.

This supports a conditional-assistance interpretation rather than unconditional multimodal fusion.

### 3. Wavelength and decoupled T/F contain the same two-dimensional FBG information

Train-only whitening followed by orthogonal Procrustes alignment shows whitened W and whitened T/F differ only by a numerical-precision orthogonal transform (relative residual ~1e-14 across the audited splits).

Consequences:

- Do not claim T/F contains more information than raw wavelengths.
- Performance differences between W and T/F representations reflect coordinate conditioning and finite optimization, not information gain.
- A physically interpretable T/F coordinate system can still be useful for interpretation, but it is not an information-theoretic advantage.

### 4. First-generation strict coordinate-invariant optical compression is too destructive

The O(2)-invariant representations based on whitened optical radius plus local step length or consecutive-vector cosine were numerically representation-invariant, but substantially underperformed the full two-channel optical expert.

Interpretation: exact invariance removes orientation/phase information useful to SOC. Keep this experiment as a negative ablation; do not use it as the main method.

## Short-budget five-seed robustness audit — exploratory, not final main table

All numbers below use six training epochs and are therefore retained only as robustness diagnostics until convergence-controlled training is completed.

### Hardest protocol: 1C -> 2C plus unseen profile

Five-seed averages:

- VI: 0.036184 MAE
- VI+W: 0.033814
- VI+W-white: 0.030393
- AnyOOD VI-or-W: 0.030698
- NestedCal99 VI-or-W: 0.030501
- AnyOOD VI-or-W-white: 0.027482
- NestedCal99 VI-or-W-white: 0.027361

`NestedCal99 VI-or-W-white` wins 14 of 30 seed×profile cases. Against VI it gives 22 wins / 8 losses; paired profile-level tests are favorable, but seed-cluster uncertainty remains material with only five seeds.

### Reverse direction: 2C -> 1C plus unseen profile

The support selectors remain off and reduce exactly to VI:

- VI / support-selective variants: 0.009211 mean MAE
- VI+W-white: 0.014079
- VI+W: 0.014911

This asymmetric behavior is desirable and should be retained as a core mechanistic control.

## Selector decisions

### Rejected: Any-OOD as the final gate

Triggering optical assistance when any current point leaves the train envelope is too sensitive to mild profile shift.

### Current leading candidate: NestedCal99 + whitened-FBG expert

For each main training split:

1. perform leave-one-profile-out scoring inside the five source-rate training profiles;
2. estimate the 99th percentile of normal internal current OOD-fraction;
3. activate the whitened-FBG expert only when a test window exceeds this train-only severity threshold.

No test SOC is used to determine the threshold.

### Negative / secondary ablation: electrical-OOD AND optical-ID complementarity gate

Five-seed short-budget averages at 1C -> 2C:

- NestedCal99 + W-white: 0.027361
- AnyOOD + W-white: 0.027482
- Electrical-OOD AND optical-ID + W-white: 0.027498

The additional optical-ID constraint did not improve the five-seed aggregate and is therefore not the current main method.

## Critical training-protocol finding

The original six-epoch budget is under-converged. Train-only nested profile validation already shows examples such as:

- `2C -> 1C / US06`: VI selects epoch 14; W and W-white select epoch 15 (the audit ceiling).
- `2C -> 1C / WLTC`: VI selects epoch 14; W selects 13; W-white selects 15 (ceiling).
- `2C -> 1C / HWFET`: VI selects epoch 14; W and W-white select 15 (ceiling).

Therefore the six-epoch five-seed table must not be treated as the final manuscript comparison. A 30-epoch train-only convergence audit is now required before the definitive selector benchmark.

## Current manuscript-safe research claim

The strongest defensible story at this stage is:

> Implanted dual-FBG sensing is not uniformly beneficial for SOC estimation. Its value emerges primarily when the electrical modality extrapolates beyond the training support, particularly in low-rate-to-high-rate transfer. A train-only support-calibrated selector can conditionally invoke an optical expert and revert to the electrical model when the electrical signal remains in-distribution, reducing negative-transfer risk. Raw/decoupled FBG coordinates contain equivalent information; observed representation differences must therefore be controlled for conditioning and optimization rather than attributed to additional physical information.

The exact performance claim remains provisional until the 30-epoch nested convergence audit and convergence-controlled multi-seed selector benchmark are complete.
