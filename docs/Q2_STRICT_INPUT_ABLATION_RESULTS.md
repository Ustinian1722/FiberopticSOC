# Formal five-seed strict input/representation ablation

Status: **COMPLETE — frozen paper-completion experiment**

Protocol: `docs/Q2_PAPER_COMPLETION_EXPERIMENT_PROTOCOL.md`. The experiment extends the already frozen formal T4 protocol without changing its splits, source-only selected epochs, windowing, normalization, or optimizer budget. The `VI+W` results reproduce the original formal T4 results to numerical precision.

## Direction-level five-seed results

All values below are equal-weight averages over 5 seeds × 6 held-out profiles (30 paired evaluations per direction).

| Direction | Input | MAE (% SOC) | RMSE (% SOC) | R² | Q95-AE (% SOC) |
|---|---|---:|---:|---:|---:|
| 1C→2C | VI | 2.162 | 3.824 | 0.977020 | 8.780 |
| 1C→2C | VI+TF | **1.770** | 3.051 | 0.984593 | **6.751** |
| 1C→2C | VI+W | 1.795 | **3.037** | **0.985689** | 6.804 |
| 2C→1C | VI | **0.464** | **0.578** | **0.999587** | **1.127** |
| 2C→1C | VI+TF | 0.493 | 0.605 | 0.999569 | 1.153 |
| 2C→1C | VI+W | 0.806 | 0.988 | 0.998480 | 1.830 |

## Paired evidence

### 1C→2C: optical information is beneficial under the difficult low-to-high-rate transfer

For `VI+W` versus matched `VI`:
- mean MAE gain (`VI − VI+W`): **+0.367 SOC percentage points**;
- relative mean-MAE reduction: approximately **17.0%**;
- paired wins/losses over seed×profile pairs: **20 / 10**;
- seed-cluster bootstrap 95% CI for absolute MAE gain: approximately **+0.004 to +0.907 SOC percentage points**.

For `VI+W` versus `VI+TF`:
- mean MAE difference is small: `VI+TF` is lower by approximately **0.025 SOC percentage points**;
- median paired difference slightly favors `VI+W`;
- wins/losses: **16 / 14** for `VI+W`;
- the seed-cluster bootstrap interval for `VI+TF − VI+W` spans zero (approximately **−0.108 to +0.059 SOC percentage points**).

Thus native W and decoupled T/F should be described as **comparable in the difficult low-to-high-rate transfer**, not as a universally ordered pair. `VI+W` gives slightly lower RMSE and higher R², while `VI+TF` gives slightly lower MAE and Q95-AE.

### 2C→1C: optical augmentation is unnecessary when the source electrical support is broader

For `VI+W` versus matched `VI`:
- mean MAE difference (`VI − VI+W`): **−0.342 SOC percentage points**;
- paired wins/losses for `VI+W`: **6 / 24**;
- seed-cluster bootstrap 95% CI: approximately **−0.492 to −0.197 SOC percentage points**.

The electrical-only control therefore dominates in the high-to-low-rate direction. This is consistent with the source-support interpretation: 2C training exposes the estimator to a broader current-excitation range, so additional optical information is not required to compensate for electrical extrapolation.

## Seed-level MAE (% SOC)

| Direction | Seed | VI | VI+TF | VI+W |
|---|---:|---:|---:|---:|
| 1C→2C | 0 | 3.656 | 3.342 | 3.322 |
| 1C→2C | 1 | 1.076 | 0.794 | 0.959 |
| 1C→2C | 2 | 1.090 | 0.941 | 1.010 |
| 1C→2C | 3 | 1.812 | 2.057 | 1.941 |
| 1C→2C | 4 | 3.177 | 1.717 | 1.744 |
| 2C→1C | 0 | 0.553 | 0.535 | 0.723 |
| 2C→1C | 1 | 0.423 | 0.456 | 0.666 |
| 2C→1C | 2 | 0.404 | 0.463 | 0.985 |
| 2C→1C | 3 | 0.494 | 0.508 | 0.695 |
| 2C→1C | 4 | 0.444 | 0.500 | 0.959 |

## Manuscript consequence

The paper should no longer claim that native W1/W2 universally outperforms T/F in formal cross-condition validation. The stronger and cleaner claim is:

1. the development-stage matched representation screen selected native W1/W2 before formal evaluation;
2. in formal five-seed low→high-rate transfer, both optical representations improve substantially over the matched electrical-only control, while W and T/F are statistically comparable in MAE;
3. the main value of FBG sensing is **condition-dependent**, becoming most relevant when the electrical target range extends beyond source support;
4. native W1/W2 remains a practical retained representation because it operates directly on measured coordinates and avoids an extra thermo-mechanical inversion, rather than because it universally minimizes every error metric.

No rescue tuning or representation reselection is permitted after this result.
