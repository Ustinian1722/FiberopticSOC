# Q2 final T4 multiseed results

Status: **FORMAL FROZEN RESULT — REPORTING ONLY**

Source workflow: `Q2 final T4 multiseed benchmark`, run `33346131766`, head SHA `5747b32b26a4ca851bb4494c43b4da09d71b49d6`, artifact `q2-final-t4-summary` (`9742172644`). The run completed successfully with the design frozen before release.

## Integrity counts

- model: `IUW-TCN`
- predictors: Voltage, Current, raw W1, raw W2
- publication seeds: 0, 1, 2, 3, 4
- directions: 1C→2C and 2C→1C
- profiles per direction: HWFET, LA92, NEDC, NYCC, US06, WLTC
- clean formal rows: 60 = 5 seeds × 2 directions × 6 held-out profiles
- FBG-noise evaluations: 900 = 5 seeds × 2 directions × 6 profiles × 3 noise levels × 5 noise draws

No architecture, feature, epoch, representation or UQ choice is changed by these results.

## Across-seed clean performance

| Direction | MAE | RMSE | R² | Q95-AE | MaxAE |
|---|---:|---:|---:|---:|---:|
| 1C→2C | 0.017954 | 0.030368 | 0.985689 | 0.068041 | 0.163266 |
| 2C→1C | 0.008057 | 0.009876 | 0.998480 | 0.018296 | 0.029125 |

The reverse 2C→1C direction is substantially more stable than 1C→2C. In 1C→2C, HWFET is the hardest profile (mean MAE 0.033214) and the seed-level mean varies materially, with seed 0 reaching mean MAE 0.033224 while seeds 1–2 are near 0.010. In 2C→1C, NEDC is the hardest profile (mean MAE 0.018569), while the other five profiles remain below 0.009 mean MAE.

## Seed-cluster uncertainty

The seed-cluster bootstrap uses 20,000 repetitions and treats the training seed as the resampling cluster.

| Scope | Mean MAE | 95% seed-cluster bootstrap interval |
|---|---:|---:|
| Overall | 0.013005 | [0.009607, 0.016767] |
| 1C→2C | 0.017954 | [0.011367, 0.025734] |
| 2C→1C | 0.008057 | [0.006892, 0.009221] |

The interval width confirms that most uncertainty is concentrated in the harder 1C→2C transfer direction.

## Raw-FBG noise robustness

Independent Gaussian test-time noise was applied directly to both raw wavelength channels before source-train normalization, at σ = 0.5, 1.0 and 2.0 pm per wavelength, with five noise draws per formal split.

| Direction | σ (pm) | Clean MAE | Noisy MAE | Absolute MAE increase | Relative increase |
|---|---:|---:|---:|---:|---:|
| 1C→2C | 0.5 | 0.017954 | 0.017972 | 0.000018 | 0.10% |
| 1C→2C | 1.0 | 0.017954 | 0.018026 | 0.000072 | 0.40% |
| 1C→2C | 2.0 | 0.017954 | 0.018253 | 0.000299 | 1.67% |
| 2C→1C | 0.5 | 0.008057 | 0.008078 | 0.000021 | 0.26% |
| 2C→1C | 1.0 | 0.008057 | 0.008144 | 0.000088 | 1.09% |
| 2C→1C | 2.0 | 0.008057 | 0.008425 | 0.000368 | 4.57% |

The degradation is monotonic and small in absolute SOC error across all three noise levels. No profile exhibits a qualitative noise-induced failure mode.

## Paper interpretation

The formal T4 result supports three claims that can be made without reopening development:

1. the compact raw-W TCN remains accurate under strict cross-rate, unseen-profile evaluation across five independent seeds;
2. transfer is directionally asymmetric, with 2C→1C markedly easier and more stable than 1C→2C;
3. the raw-FBG representation is tolerant to small wavelength perturbations up to 2 pm in this controlled test-time robustness study.

The same-direction 1C→2C experiment remains a design-frozen robustness replication because that target direction contributed to the earlier representation-development screen. The reverse 2C→1C direction retains the stronger confirmatory interpretation.
