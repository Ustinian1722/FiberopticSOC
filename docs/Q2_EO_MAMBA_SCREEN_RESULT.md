# Q2 EO-Mamba development-screen result

Status: **DEVELOPMENT EVIDENCE — MAMBA FAMILY NOT RETAINED**

Source workflow: GitHub Actions run `33348016670`, artifact `eo-gated-mamba-screen`, seed 42, 15 epochs, window 64, train stride 4, test stride 1. The screen used the already-designated development direction `1C -> 2C` with one unseen profile held out at a time. It does not inspect the stronger-confirmatory `2C -> 1C` direction.

## Aggregate result

| Model | Params | MAE | RMSE | R2 | Q95-AE | Wins vs IUW-TCN |
|---|---:|---:|---:|---:|---:|---:|
| **IUW-TCN** | 11,545 | **0.014859** | **0.022293** | **0.994504** | **0.051582** | - |
| DualMS-Mamba | 21,825 | 0.019156 | 0.030487 | 0.990482 | 0.068958 | 1/6 |
| EO-Gated-TCN | 25,393 | 0.019696 | 0.033150 | 0.988939 | 0.076188 | 1/6 |
| EO-Gated-Mamba | 23,601 | 0.020125 | 0.031216 | 0.989549 | 0.070338 | 0/6 |
| EO-Gated-Mamba-TF | 23,601 | 0.023357 | 0.038153 | 0.984236 | 0.086303 | 0/6 |
| VIW-Mamba | 18,465 | 0.029746 | 0.053181 | 0.967060 | 0.123496 | 0/6 |
| VI-Mamba | 18,401 | 0.055282 | 0.101949 | 0.880479 | 0.234209 | 0/6 |

The best Mamba-family candidate, `DualMS-Mamba`, is 28.9% worse than IUW-TCN in mean MAE. The full raw-W `EO-Gated-Mamba` is 35.4% worse. This is not a marginal loss that should be rescued by target-guided hyperparameter search.

## Gate diagnostic

The learned optical gate is effectively collapsed around one half. For raw-W EO-Gated-Mamba, profile-level gate means are approximately 0.487-0.495 and within-profile standard deviations only about 0.005-0.007. Therefore the scalar/channel gate does not provide convincing operating-condition adaptation and should not be used as a paper interpretability claim.

## Representation conclusion strengthened

Within the exact same EO-Gated-Mamba architecture:

- raw W1/W2 MAE: `0.020125`
- physics-decoupled T/F MAE: `0.023357`

The raw optical coordinates are about 13.8% better in mean MAE. Together with the earlier TCN representation screen, this provides architecture-diverse evidence that explicit T/F inversion is physically interpretable but not predictively preferable under compound rate/profile shift.

## Decision

1. Do **not** promote Mamba simply for novelty/taste.
2. Retain Mamba-family results as a negative architecture ablation if useful.
3. Retain raw W1/W2 as the preferred optical representation; T/F remains a physics-decoupled representation baseline.
4. Run exactly one final architecture-family development screen motivated by current SOC literature: modality-specific causal local encoding + electrical/optical cross-attention + Transformer global context.
5. After that screen, freeze the proposed architecture. Do not inspect `2C -> 1C` until architecture/representation are frozen.

The next candidate must earn retention on the existing development evidence; architectural sophistication alone is not sufficient.