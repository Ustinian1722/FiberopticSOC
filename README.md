# FiberopticSOC

Reproducible analysis workspace for the public Mendeley Data dataset **Multi-condition Battery In-situ Sensing Data** (DOI: `10.17632/ft6rtwt8vm.1`). The dataset contains electrical variables and implanted Fiber Bragg Grating (FBG) sensing signals for silicon-based lithium-ion battery SOC estimation.

## Current data status

The original `SiC-18.zip` archive is now committed at the repository root and is consumed directly by `.github/workflows/analyze-committed-sic18.yml`. The workflow stages it into `data/raw/`, verifies its audited SHA-256, extracts the 12 workbooks, reruns the leakage/structure audit, and reruns the group-split baseline.

Audited provenance:

- SHA-256: `8ebc43eb6d205dfc573dae853d63efcd1a00169be9d2f1e512b9220ea7799bc4`
- Size: 5,398,170 bytes
- 12 XLSX workbooks
- 68,086 total samples
- Profiles: HWFET, LA92, NEDC, NYCC, US06, WLTC
- Rates: 1C and 2C
- Cell: SiC-18

See `data/SiC-18_MANIFEST.md` and `docs/INITIAL_AUDIT.md` for the audited structure and findings.

## Data source

- Dataset: Multi-condition Battery In-situ Sensing Data
- Mendeley Data ID: `ft6rtwt8vm`, version `1`
- DOI: `10.17632/ft6rtwt8vm.1`
- License: CC BY 4.0
- Columns in every audited workbook: `Time_s`, `Current_A`, `Voltage_V`, `Wavelength_1`, `Wavelength_2`, `temperature_℃`, `force_N`, `dis_cap`, `SOC`

## Dual-FBG physical decoupling

The companion work uses an implanted **Armored FBG + Bare FBG** pair. Both wavelengths respond to temperature and mechanical action, but with different calibrated sensitivities. This gives a 2×2 sensitivity system that can be inverted to recover internal temperature and deformation force.

The released data are exactly consistent with:

`W1 = 0.0208 * T + 0.00054 * F`

`W2 = 0.0254 * T + 0.00085 * F`

and therefore:

`T = 214.429868819374 * W1 - 136.226034308779 * W2`

`F = -6407.66902119071 * W1 + 5247.22502522705 * W2`

Thus `T,F` should be interpreted as **physics-decoupled thermo-mechanical features derived from the two implanted FBG channels**. They are legitimate physical-state inputs. However, `W1,W2,T,F` together still contain only two independent FBG degrees of freedom because the transform is exactly invertible.

## Research-integrity constraints

1. `SOC` is constructed from discharge capacity: for 11/12 files it is exactly `SOC = 1 - dis_cap / max(dis_cap)`; the remaining file differs only by about `2.77e-05` at the terminal point. Therefore `dis_cap` is forbidden as a model input.
2. Absolute `Time_s` is excluded from the main benchmark because each workbook is a monotonic discharge trajectory and elapsed time can act as a progress/SOC proxy.
3. Raw optical (`W1,W2`) and physics-decoupled (`T,F`) inputs should be compared as alternative representations, not counted as four independent sensing modalities.

## Clean benchmark family

The initial non-sequential diagnostic uses group-based testing rather than random row splits:

- Same-rate leave-one-profile-out: train on five drive profiles and test on the sixth, repeated across all profiles at 1C and 2C.
- Cross-rate: train on all 1C data and test all 2C data, then reverse.

Feature families:

- `V`: voltage only
- `VI`: voltage + current
- `VI+W`: voltage + current + raw Wavelength 1/2
- `VI+TF`: voltage + current + physics-decoupled temperature/force
- `W`: raw wavelength pair only
- `TF`: decoupled temperature/force only

Initial mean leave-one-profile-out MAE:

- `VI+TF`: 0.7691%
- `VI`: 0.9320%
- `VI+W`: 0.9987%

Cross-rate 1C→2C MAE:

- `VI+W`: 1.5366%
- `VI+TF`: 1.8753%
- `VI`: 2.5603%

Cross-rate 2C→1C MAE:

- `VI+TF`: 0.8024%
- `VI`: 1.1066%
- `VI+W`: 1.1281%

Because the raw and decoupled FBG coordinates are mathematically invertible, differences between `VI+W` and `VI+TF` should be interpreted as representation/conditioning/model-inductive-bias effects rather than extra information.

## Repository outputs

- `SiC-18.zip`: committed source archive
- `docs/INITIAL_AUDIT.md`: detailed audit, leakage analysis, dual-FBG decoupling interpretation, baseline results
- `data/SiC-18_MANIFEST.md`: archive hash, file list, sample counts
- `results/initial_baseline_summary.csv`: compact group-split baseline results
- `results/signal_soc_correlations.csv`: pooled Pearson/Spearman associations
- `analysis/run_initial_baseline.py`: leakage-safe diagnostic baseline implementation
- `.github/workflows/analyze-committed-sic18.yml`: reproducible archive verification + audit + baseline workflow

## Next modeling step

The next defensible stage is a causal sequence benchmark under the same group splits. Recommended inputs are `VI`, `VI+W`, and `VI+TF`; `dis_cap` and absolute time remain forbidden. Evaluate MAE, RMSE, R², MaxAE, Q95 absolute error, per-SOC-bin error, and profile/rate robustness.
