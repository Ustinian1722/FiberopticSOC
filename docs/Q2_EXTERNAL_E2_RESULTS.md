# Q2 external E2 same-cell cross-rate result

Status: **FROZEN — DOES NOT PASS THE ROBUST-BENEFIT GATE, BUT SHOWS MEAN-ERROR BENEFIT IN 3/4 CELLS**

Canonical workflow: GitHub Actions run `33360769760`, artifact `q2-external-e2-summary`, head SHA `74117c16bda167ec8d906bd13b9ae50449ac1cf8`.

Frozen protocol:

- each physical cell treated independently;
- train on the same cell's 0.2C + 0.5C constant-current discharges;
- test only on the same cell's held-out 1C discharges;
- A1/A2/P1/P2 are four paired replications;
- source-rate-only normalization;
- VI-TCN versus VI-S5rel-TCN;
- fixed S5rel coordinate and external TCN template;
- seed 42, 20 epochs, window 64;
- no 1C target sample used for training, normalization, preprocessing design or hyperparameter/model selection.

## Equal-cell result

| Model | MAE | RMSE | R2 | Q95-AE | MaxAE |
|---|---:|---:|---:|---:|---:|
| VI-TCN | 0.149199 | 0.175529 | 0.518712 | **0.305993** | **0.400971** |
| VI-S5rel-TCN | **0.134241** | **0.164119** | **0.559508** | 0.312989 | 0.442248 |

Adding S5rel lowers mean cell-level MAE by `0.014958` SOC (about 10.0% relative) and mean RMSE by `0.011410` SOC (about 6.5% relative), but slightly worsens mean Q95-AE.

Pooled-window MAE is consistent: `0.152078` for VI-TCN versus `0.135240` for VI-S5rel-TCN.

## Paired cell results

| Cell | VI MAE | VI+S5rel MAE | Relative change |
|---|---:|---:|---:|
| A1 | 0.173643 | **0.125170** | **-27.92%** |
| A2 | 0.152649 | **0.144492** | **-5.34%** |
| P1 | 0.176824 | **0.165054** | **-6.66%** |
| P2 | **0.093679** | 0.102248 | **+9.15%** |

S5rel wins MAE in 3/4 cells. The worst MAE degradation (P2, +9.15%) remains within the pre-registered +10% tolerance.

## Pre-registered support gate

- mean cell MAE improves: **pass**;
- MAE wins >=3/4 cells: **pass**;
- mean cell RMSE non-worse: **pass**;
- mean cell Q95-AE non-worse: **fail** (`0.312989 > 0.305993`);
- worst cell MAE increase <=10%: **pass**.

Decision: **DOES_NOT_SUPPORT_ROBUST_SAME_CELL_CROSS_RATE_FBG_BENEFIT** because all five conditions were required.

No multi-seed confirmation is triggered because the initial frozen gate did not fully pass.

## Interpretation

E2 is materially different from E1:

- E1 cross-cell: S5rel worsens mean MAE and wins only 1/4 cells, showing strong sensor/cell calibration mismatch.
- E2 same-cell cross-rate: S5rel reduces average MAE/RMSE and wins 3/4 cells, showing that once sensor identity is fixed the optical signal carries useful rate-transfer information.

However, the optical model does not improve the high-error tail consistently. P1 is the clearest example: MAE improves but RMSE/Q95 worsen, implying occasional large errors despite better typical predictions. P2 degrades in both central and tail metrics.

Therefore the defensible conclusion is not 'FBG robustly improves same-cell cross-rate SOC' but:

> Direct surface-FBG measurements provide useful same-sensor cross-rate information on average, while their tail-error reliability remains condition/cell dependent.

This result supports the main SiC-18 observation that optical information can aid cross-condition estimation, while the external dataset exposes the limits imposed by sensor installation, calibration drift and rate-dependent optical response.

E2 does not reopen the E1 cross-cell conclusion or any SiC-18 model/representation decision.