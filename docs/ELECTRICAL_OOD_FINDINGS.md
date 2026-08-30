# Electrical OOD and selective optical assistance

This document records the **corrected parameter-matched** diagnostic relating electrical distribution shift to the benefit of raw dual-FBG input.

## Clean control design

Two causal TCN experts use exactly the same four-channel input width and architecture:

- `VI`: `[V, I, 0, 0]`
- `VI+W`: `[V, I, W1, W2]`

Thus the comparison does not duplicate V/I and does not confound the result with parameter count.

For each training split, an electrical in-distribution envelope is computed from **training current only** using the 0.5th and 99.5th percentiles. For every causal 64-sample test window, `ood_fraction` is the fraction of current samples outside that training envelope. This OOD score requires no SOC labels and no test-domain fitting.

## Cross-rate result

### 1C -> 2C

- 52.64% of test windows contain at least one current sample outside the 1C training envelope.
- Overall `VI` MAE: **0.021507**.
- Overall `VI+W` MAE: **0.016316**.
- Pearson correlation between `ood_fraction` and per-window optical gain (`|e_VI| - |e_VIW|`): **0.4050**.
- Spearman correlation: **0.2879**.

The optical benefit grows strongly with electrical OOD severity:

| Electrical OOD bin | Windows | VI MAE | VI+W MAE | Relative optical gain |
|---|---:|---:|---:|---:|
| ID (0%) | 12,011 | **0.007348** | 0.008726 | **-18.75%** |
| OOD 0–25% | 4,296 | 0.015501 | **0.014700** | +5.17% |
| OOD 25–50% | 2,742 | 0.027723 | **0.023564** | +15.00% |
| OOD 50–75% | 1,849 | 0.036646 | **0.029298** | +20.05% |
| OOD 75–100% | 4,464 | 0.055292 | **0.028464** | **+48.52%** |

The key observation is not merely that FBG helps on average. **Within the training electrical envelope, FBG is harmful; its benefit emerges and increases as the electrical trajectory moves out of distribution.**

### 2C -> 1C

All 41,968 causal test windows are inside the 2C training current envelope:

- `VI` MAE: 0.008657
- `VI+W` MAE: 0.008135

There is no electrical-OOD variation in this direction, so no OOD/gain correlation can be estimated. The rate shift is therefore asymmetric: low-rate training must extrapolate into high-current regions, while high-rate training already covers the lower-rate current envelope.

## Label-free selective-expert sanity check

Using the two already trained experts, a simple deterministic inference rule was evaluated:

```text
if ood_fraction > 0:
    prediction = VI+W expert
else:
    prediction = VI expert
```

No test SOC label is used to choose the expert, and no threshold is tuned on the test domain beyond the fixed training-only 0.5%–99.5% current envelope.

Results:

| Split | VI | Always-on VI+W | Hard OOD-selective expert |
|---|---:|---:|---:|
| 1C -> 2C MAE | 0.021507 | 0.016316 | **0.015663** |
| 2C -> 1C MAE | 0.008657 | **0.008135** | 0.008657 |

In the difficult 1C -> 2C shift, the simple selective rule improves on both single experts by avoiding the negative transfer of FBG in ID windows while preserving its large OOD benefit. In the reverse direction all windows are ID, so the rule correctly falls back to VI; the small advantage of always-on W in this direction remains a trade-off to be tested under broader splits.

A fixed continuous rule `g = ood_fraction` also improves 1C -> 2C over always-on W (MAE `0.015836`), but the hard rule is currently preferred as the primary mechanism test because it has the clearest interpretation and no shape hyperparameter.

## Research implication

The emerging story is **selective optical assistance under electrical distribution shift**, not generic multimodal fusion:

1. Electrical variables are sufficient and sometimes preferable when the operating trajectory is inside the training support.
2. Raw FBG provides a complementary internal-state coordinate that becomes especially valuable when current trajectories leave the training electrical support.
3. Always-on fusion can produce negative transfer in ID regions.
4. A training-only electrical reliability/OOD score can decide when optical information should be trusted without requiring test SOC labels.
5. Raw W is a natural optical expert input because it also avoids the calibration-inversion sensitivity identified in the decoupled T/F robustness study.

## Required next validation

Before this becomes the final method claim, it must pass:

- all 12 same-rate leave-one-profile-out splits;
- multiple random seeds;
- fixed-gate vs soft-gate ablation with any hyperparameters selected without test labels;
- parameter-matched experts;
- per-profile and per-SOC-bin reporting;
- calibration/noise robustness;
- comparison with always-on VI+W, VI+TF, whitened TF and strong sequence baselines.

If selective assistance does not generalize beyond cross-rate extrapolation, the paper should present it specifically as a cross-rate/domain-extrapolation method rather than claim universal gating.