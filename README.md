# FiberopticSOC

Reproducible analysis workspace for the public Mendeley Data dataset **Multi-condition Battery In-situ Sensing Data** (DOI: `10.17632/ft6rtwt8vm.1`). The dataset contains electrical variables and implanted Fiber Bragg Grating (FBG) sensing signals for silicon-based lithium-ion battery SOC estimation.

## Data source

- Dataset: Multi-condition Battery In-situ Sensing Data
- Mendeley Data ID: `ft6rtwt8vm`, version `1`
- DOI: `10.17632/ft6rtwt8vm.1`
- License: CC BY 4.0
- Expected channels include time, current, voltage, discharge capacity, Wavelength 1, Wavelength 2, temperature, force, and SOC.

The repository deliberately does **not** commit the raw dataset. GitHub Actions retrieves the current public Mendeley download URLs at runtime, verifies SHA-256 when Mendeley exposes one, extracts archives, performs an audit, and uploads only compact analysis outputs as an Actions artifact.

## Automated audit

Workflow: `.github/workflows/fiberoptic-data-audit.yml`

It performs:

1. Download all public files from Mendeley Data.
2. Inventory CSV/XLS/XLSX tables and sheets.
3. Normalize the main electrical/FBG channel names.
4. Report row counts, missing values, duplicates, ranges, and channel coverage.
5. Compute per-table and pooled Pearson/Spearman association with SOC.
6. Generate representative multichannel traces, a correlation heatmap, SOC-vs-FBG plots, and SOC-binned channel profiles.
7. Run a research-integrity/leakage audit and write `results/analysis_report.md`.
8. Upload `results/` plus the Mendeley manifest as the `fiberoptic-soc-initial-audit` artifact.

You can re-run it manually from the GitHub Actions tab via **Run workflow**.

## Important benchmark rule

The source description states that SOC is calculated from discharge capacity by integration. Therefore `dis_cap` is a target-construction variable and must **not** be used as an input in the primary SOC benchmark. `time` should also be treated cautiously because fixed/repeated discharge protocols can make time a proxy for SOC.

The first clean ablation family should therefore compare:

- E0: `V + I`
- E1: `V + I + temperature`
- E2: `V + I + force`
- E3: `V + I + wavelength_1 + wavelength_2`
- E4: `V + I + temperature + force`
- E5: raw-FBG versus decoupled-FBG under identical train/test splits

## Local reproduction

```bash
python -m pip install -r requirements.txt
python scripts/download_mendeley.py
python scripts/analyze_dataset.py
```
