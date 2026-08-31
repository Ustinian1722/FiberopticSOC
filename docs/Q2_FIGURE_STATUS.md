# Q2 publication figure production status

Status: **DATA FIGURES FROZEN; SCHEMATICS DEFERRED**

Canonical QA-clean rendering:
- GitHub Actions run: `33367693061`
- head SHA: `2116e487f9c7ec5a9c1eca5fdd6e02ca54694a26`
- artifact ID: `9748901023`
- artifact label: `q2-publication-figures-v2`

## Production policy

The public `nature-figure` skill (`Yuan1z0825/nature-skills`) is used as the figure-production and QA reference. We retain its main contracts:

- one Results-level scientific question per main figure;
- no silent data sampling/exclusion for rendering convenience;
- explicit source-data mapping;
- editable SVG as the primary figure output;
- PDF and 600-dpi TIFF secondary publication exports;
- fixed physical width and explicit font-size control;
- rendered text/collision QA after export.

### Intentional typography override

The upstream skill defaults to Arial/Helvetica-like sans-serif fonts. This manuscript has standardized plot typography to a Times New Roman-style serif. The renderer therefore uses `Times New Roman` with `Times`, `Liberation Serif`, and `DejaVu Serif` fallbacks while keeping `svg.fonttype='none'` so SVG text remains editable.

This is an intentional project-level style choice; the remaining nature-figure contracts are retained.

## Final data-figure QA

All six rendered quantitative figures passed the final automated checks:

| Figure | Minimum PDF text | Collision QA |
|---|---:|---|
| Fig. 1 | 7.0 pt | PASS, 0 fail / 0 warn |
| Fig. 2 | 6.0 pt | PASS, 0 fail / 0 warn |
| Fig. 4 | 7.0 pt | PASS, 0 fail / 0 warn |
| Fig. 5 | 6.3 pt | PASS, 0 fail / 0 warn |
| Fig. 6 | 6.0 pt | PASS, 0 fail / 0 warn |
| Fig. 7 | 7.0 pt | PASS, 0 fail / 0 warn |

The earlier Fig. 2 log-colorbar mathtext issue (<5 pt superscripts) and Fig. 5/6/7 layout-collision warnings were resolved without changing any numerical data or scientific claims.

## Main-figure status

| Figure | Status | Production source | Main claim |
|---|---|---|---|
| Fig. 1 | **FROZEN DATA PANELS; panel (a) PLACEHOLDER** | `SiC-18.zip` | Electrical and dual-FBG optical signals provide synchronized but distinct views during dynamic discharge. |
| Fig. 2 | **FROZEN** | all 12 SiC-18 trajectories + frozen representation summary | Raw wavelength and decoupled T/F are alternative coordinates; interpretability and predictive conditioning need not coincide. |
| Fig. 3 | **PLACEHOLDER** | methodology spec only | Compact causal TCN + residual conformal UQ. Framework artwork is intentionally deferred. |
| Fig. 4 | **FROZEN** | fixed blocked-T1 RA-FBG-TCN rerun | The retained estimator provides sub-percent conventional SOC accuracy. |
| Fig. 5 | **FROZEN / HERO EXPLANATORY FIGURE** | frozen parameter-matched VI vs VI+W OOD analysis | Optical benefit increases as electrical measurements leave source support. |
| Fig. 6 | **FROZEN** | frozen 5-seed T4 summary | Cross-rate unseen-profile generalization is strong but asymmetric. |
| Fig. 7 | **FROZEN** | frozen wavelength-noise results + 95% conformal T1 rerun | Wavelength perturbations cause smooth degradation and 95% conformal intervals are well calibrated. |

## Figure-level source-data contracts

Committed compact source data are under `paper/source_data/`:

- `fig2_representation_transfer.csv`: frozen representation transfer comparison;
- `fig5_direction_summary.csv`: parameter-matched cross-rate VI versus VI+W summary;
- `fig5_ood_bins.csv`: electrical-OOD severity bins and optical gain;
- `fig6_t4_profile_summary.csv`: profile-wise 5-seed mean/std results;
- `fig6_t4_bootstrap.csv`: direction/overall seed-cluster bootstrap CI;
- `fig7_noise_summary.csv`: 0/0.5/1/2 pm wavelength-noise robustness;
- `fig7_uq_summary.csv`: 90/95% conformal summary, with 95% used in the main paper;
- `table3_model_comparison.csv`: frozen blocked-T1 model comparison.

Fig. 1 and the raw geometry panels of Fig. 2 are regenerated directly from all observations in the 12 released workbooks. Fig. 4 and the interval panel of Fig. 7 are regenerated from the fixed RA-FBG-TCN blocked-T1 configuration; their derived prediction table is emitted into the figure artifact as generated source data.

## Output contract

- target full-width figure: **183 mm**;
- primary editable vector: **SVG**;
- secondary vector: **PDF**;
- submission raster: **TIFF, 600 dpi, LZW**;
- review preview: **PNG, 300 dpi**;
- panel labels: `a`, `b`, ... at fixed physical offsets;
- consistent mapping across figures: electrical/retained estimate = blue; difficult/OOD direction or adverse effect = red; controls = neutral gray;
- SOC errors reported as `% SOC`.

## What is deliberately not being drawn yet

- Fig. 1(a) experimental setup / implanted dual-FBG sensing schematic;
- Fig. 3 RA-FBG-TCN methodology framework;
- graphical abstract.

These remain layout placeholders. They can be replaced later without changing the frozen quantitative panel numbering or Results narrative.

## Freeze rule

Do not revise the quantitative data figures for aesthetic experimentation alone. Reopen a frozen figure only for a demonstrated factual error, journal-specific formatting requirement, or concrete reviewer request.