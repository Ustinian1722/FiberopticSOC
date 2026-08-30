# FiberopticSOC

Reproducible analysis workspace for the public Mendeley Data dataset **Multi-condition Battery In-situ Sensing Data** (DOI: `10.17632/ft6rtwt8vm.1`). The dataset contains electrical variables and implanted Fiber Bragg Grating (FBG) sensing signals for silicon-based lithium-ion battery SOC estimation.

## Current data status

The `SiC-18.zip` archive was supplied directly for analysis after Mendeley Data began returning Cloudflare HTTP 403 responses to GitHub-hosted runners. Its provenance is fixed by:

- SHA-256: `8ebc43eb6d205dfc573dae853d63efcd1a00169be9d2f1e512b9220ea7799bc4`
- Size: 5,398,170 bytes
- 12 XLSX workbooks
- 68,086 total samples
- Profiles: HWFET, LA92, NEDC, NYCC, US06, WLTC
- Rates: 1C and 2C
- Cell: SiC-18

See `data/SiC-18_MANIFEST.md` and `docs/INITIAL_AUDIT.md` for the audited structure and findings.

The connected GitHub write interface available in this session accepts text/blob content but does not expose a direct local-file upload parameter for the uploaded binary attachment. The raw archive is therefore represented by its exact hash/manifest rather than silently re-encoded. Analysis outputs and reproducible code are committed normally.

## Data source

- Dataset: Multi-condition Battery In-situ Sensing Data
- Mendeley Data ID: `ft6rtwt8vm`, version `1`
- DOI: `10.17632/ft6rtwt8vm.1`
- License: CC BY 4.0
- Columns in every audited workbook: `Time_s`, `Current_A`, `Voltage_V`, `Wavelength_1`, `Wavelength_2`, `temperature_℃`, `force_N`, `dis_cap`, `SOC`

## Research-integrity constraints

Two leakage/redundancy findings are already established from the supplied data:

1. `SOC` is constructed from discharge capacity: for 11/12 files it is exactly `SOC = 1 - dis_cap / max(dis_cap)`; the remaining file differs only by about `2.77e-05` at the terminal point. Therefore `dis_cap` is forbidden as a model input.
2. `temperature_℃` and `force_N` are an exactly invertible linear transform of `Wavelength_1/2` (`R² = 1`, numerical residual about `1e-13`). They are alternative representations of the same two FBG degrees of freedom, not four independent sensing channels.

Absolute `Time_s` is also excluded from the main benchmark because each workbook is a monotonic discharge trajectory and time can act as a progress/SOC proxy.

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

- `docs/INITIAL_AUDIT.md`: detailed audit, leakage analysis, FBG transform equations, baseline interpretation
- `data/SiC-18_MANIFEST.md`: archive hash, file list, sample counts
- `results/initial_baseline_summary.csv`: compact group-split baseline results
- `results/signal_soc_correlations.csv`: pooled Pearson/Spearman associations
- `analysis/run_initial_baseline.py`: leakage-safe diagnostic baseline implementation
- `.github/workflows/fiberoptic-data-audit.yml`: original Mendeley download/audit workflow; current hosted-runner downloads are blocked by Mendeley Cloudflare and should not be treated as a data-quality failure

## Next modeling step

The next defensible stage is a causal sequence benchmark under the same group splits. Recommended inputs are `VI`, `VI+W`, and `VI+TF`; `dis_cap` and absolute time remain forbidden. Evaluate MAE, RMSE, R², MaxAE, Q95 absolute error, per-SOC-bin error, and profile/rate robustness.
