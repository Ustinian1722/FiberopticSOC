# Q2 external E1 cross-cell result

Status: **FROZEN — DOES NOT SUPPORT ROBUST CROSS-CELL FBG BENEFIT**

Canonical workflow: GitHub Actions run `33359316457`, artifact `q2-external-e1-summary`, head SHA `6add401d97e9ed16cc3290dfe571edc8cdd3a510`.

Protocol was frozen before any E1 target metric was inspected:

- four leave-one-cell-out folds: A1, A2, P1, P2;
- 52 physically frozen constant-current discharge segments;
- rates 0.2C, 0.5C, 1C;
- 781,654 valid samples;
- 64-sample causal windows;
- source-cell-only normalization;
- seed 42, fixed 20 epochs;
- exact matched TCN template;
- baseline `VI-TCN` versus `VI-S5rel-TCN`;
- S5 selected a priori as the common central FBG position;
- `S5_rel(t)=lambda_S5(t)-lambda_S5(discharge_start)` removes only sensor-specific additive zero point;
- held-out cell never used for training, normalization, architecture selection, sensor selection, sign alignment, or calibration.

## Main result

Equal-weight mean across the four held-out physical cells:

| Model | MAE | RMSE | R2 | Q95-AE | MaxAE |
|---|---:|---:|---:|---:|---:|
| **VI-TCN** | **0.096603** | **0.136205** | **0.714768** | **0.316855** | **0.815910** |
| VI-S5rel-TCN | 0.118696 | 0.163024 | 0.628782 | 0.346991 | 0.831234 |

Pooled-window MAE is consistent with the cell-equal result: `0.095536` for VI-TCN versus `0.115515` for VI-S5rel-TCN.

## Paired held-out-cell results

| Held-out cell | VI MAE | VI+S5rel MAE | Relative MAE change |
|---|---:|---:|---:|
| A1 | 0.153608 | **0.138943** | **-9.55%** |
| A2 | **0.057570** | 0.120100 | **+108.62%** |
| P1 | **0.047474** | 0.066930 | **+40.98%** |
| P2 | **0.127761** | 0.148812 | **+16.48%** |

S5rel improves only 1/4 held-out physical cells.

## Pre-registered gate

All required criteria fail:

- mean cell-level MAE improves: false;
- S5rel wins at least 3/4 cells: false (`1/4`);
- mean cell-level RMSE non-worse: false;
- mean cell-level Q95-AE non-worse: false;
- worst held-out-cell MAE increase <=10%: false (A2 increases by ~108.6%).

Decision: **DOES_NOT_SUPPORT_ROBUST_CROSS_CELL_FBG_BENEFIT**.

## Rate-level diagnostic

Equal-weight cell means by target rate:

| Rate | VI MAE | VI+S5rel MAE |
|---|---:|---:|
| 0.2C | **0.052253** | 0.060143 |
| 0.5C | **0.100767** | 0.129577 |
| 1.0C | **0.225479** | 0.269795 |

The external cross-cell problem becomes substantially harder with increasing discharge rate, and direct S5rel does not reverse this trend. A1 benefits from S5rel at all three rates, whereas the other physical cells show heterogeneous or adverse behavior.

## Interpretation

This result does **not** invalidate FBG sensing for SOC. It rejects a narrower transfer claim: a single zero-point-aligned surface-FBG coordinate cannot be assumed to form a universal cell-invariant SOC representation across different physical pouch cells/adhesive states.

The frozen external dataset diagnostics already showed strong cell x C-rate interaction in S5rel, including different response signs/shapes across cells and rates. E1 confirms that removing only the additive Bragg zero point is insufficient to make that response universally transferable.

This is consistent with the source publication's own sensor-physics findings (Hebenbrock et al., *Electrochimica Acta*, 2025, DOI `10.1016/j.electacta.2025.146975`):

- A1/A2 form an aged adhesive set that underwent about 5.5 additional months of testing relative to pristine P1/P2;
- the degradation of strain transfer is different for individual fixed FBG sensors;
- A2/S5 is reported as showing critically advanced debonding and non-temperature-correlated irregularities;
- validation analysis requires sensor-specific scaling/update of calibration constants as adhesive strain transfer changes;
- debonding of one fixed sensor can alter the strain distribution seen by neighboring sensors.

Therefore an additive zero-point correction alone cannot be expected to remove sensor-specific sensitivity, adhesive-state and strain-transfer differences. The E1 failure is an engineering boundary of uncalibrated surface-FBG interchangeability rather than evidence that optical sensing contains no SOC-related information.

No target-informed sign flip, scale correction, cell-type calibration, sensor-position search, or architecture rescue is permitted after this result.

## Consequence for E2

E2 remains a separately pre-declared external task. Its purpose is now especially clean: remove the cross-cell sensor mismatch by keeping the physical cell fixed, train on the common lower/mid rates (0.2C + 0.5C), and evaluate the held-out highest rate (1C). This tests whether S5rel contains useful **same-cell cross-rate** information even though E1 rejects universal cross-cell transfer.

E1 target results cannot be used to tune E2 architecture or optical preprocessing.