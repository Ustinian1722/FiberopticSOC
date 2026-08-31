# Q2 publication figure production status

Status: **DATA FIGURES IN PRODUCTION; SCHEMATICS DEFERRED**

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

The upstream skill defaults to Arial/Helvetica-like sans-serif fonts. This manuscript has already standardized plot typography to a Times New Roman-style serif. Therefore the renderer uses `Times New Roman` with `Times`, `Liberation Serif`, and `DejaVu Serif` fallbacks while keeping `svg.fonttype='none'` so all SVG text remains editable.

This is an intentional project-level style choice, not a failure to follow the remaining nature-figure contracts.

## Main-figure status

| Figure | Status | Production source | Main claim |
|---|---|---|---|
| Fig. 1 | **DATA PANELS READY; panel (a) PLACEHOLDER** | `SiC-18.zip` | Electrical and dual-FBG optical signals provide synchronized but distinct views during dynamic discharge. |
| Fig. 2 | **READY PIPELINE** | all 12 SiC-18 trajectories + frozen representation summary | Raw wavelength and decoupled T/F are alternative coordinates; interpretability and predictive conditioning need not coincide. |
| Fig. 3 | **PLACEHOLDER** | methodology spec only | Compact causal TCN + residual conformal UQ. Framework artwork is intentionally deferred. |
| Fig. 4 | **READY PIPELINE** | fixed blocked-T1 RA-FBG-TCN rerun | The retained estimator provides sub-percent conventional SOC accuracy. |
| Fig. 5 | **READY PIPELINE / HERO EXPLANATORY FIGURE** | frozen parameter-matched VI vs VI+W OOD analysis | Optical benefit increases as electrical measurements leave source support. |
| Fig. 6 | **READY PIPELINE** | frozen 5-seed T4 summary | Cross-rate unseen-profile generalization is strong but asymmetric. |
| Fig. 7 | **READY PIPELINE** | frozen wavelength-noise results + 95% conformal T1 rerun | Wavelength perturbations cause smooth degradation and 95% conformal intervals are well calibrated. |

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

These are kept as layout placeholders until the data figures and manuscript dimensions are stable. This prevents framework artwork from driving the paper layout prematurely.
