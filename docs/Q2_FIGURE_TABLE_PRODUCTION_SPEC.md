# Q2 main-paper figure/table production specification

Status: **QUANTITATIVE FIGURES FROZEN; SCHEMATICS PLACEHOLDER**

The visual package follows a conventional Energy/JES-style structure with seven numbered main figures and five default main tables. Development-history figures are excluded from the main manuscript.

## Fig. 1 — Dataset, sensing arrangement and representative electrical–optical signals

Composition:
- (a) sensing/test schematic or platform image — **PLACEHOLDER** until the experimental visual is supplied/assembled;
- (b) representative NEDC 1C current + SOC trajectory;
- (c) synchronized W1/W2 trajectory;
- (d) 1C/2C W2–SOC comparison for the same profile.

Message: electrical and optical channels are synchronized but exhibit distinct dynamic responses.

## Fig. 2 — Dual-FBG representation analysis

Final 2×2 composition:
- (a) W1–W2 density;
- (b) T–F density;
- (c) absolute Pearson correlation by representation and rate;
- (d) matched cross-rate development MAE comparison.

Final reproducible descriptive correlation values:
- raw W1/W2: |r| = 0.738 at 1C and 0.655 at 2C;
- decoupled T/F: |r| = 0.982 at 1C and 0.985 at 2C.

Do **not** annotate the old development-stage covariance-condition-number approximations in the main figure. The main claim does not require them.

Message: physical decoupling improves semantic interpretability but creates a more strongly linearly coupled coordinate in this dataset; native W yields stronger matched predictive transfer for the retained TCN.

## Fig. 3 — RA-FBG-TCN framework

**PLACEHOLDER** until framework artwork is drawn.

Fixed content contract:
V/I/W1/W2
→ train-only normalization
→ 64-sample causal window
→ 1×1 projection (4→24)
→ residual TCN block d=1
→ residual TCN block d=2
→ residual TCN block d=4
→ final causal state
→ regression head
→ SOC point estimate
→ 95% residual conformal interval.

TCN block inset: causal conv k=3 → GroupNorm → GELU → causal conv k=3 → GroupNorm → residual + GELU.

## Fig. 4 — Conventional SOC prediction examples

Use RA-FBG-TCN only; model comparison stays in Table 3.

Panels:
- (a) representative NEDC test trajectory;
- (b) representative NYCC test trajectory;
- (c) absolute error over the dynamic segment;
- (d) overall absolute-error distribution.

Message: the retained compact causal estimator provides sub-percent conventional point-estimation accuracy.

## Fig. 5 — Electrical-OOD severity versus optical benefit

This is the main explanatory figure.

Final panels:
- (a) 1C training-current support envelope with a representative 2C test current;
- (b) OOD bin versus VI and VI+W MAE;
- (c) OOD bin versus relative optical gain.

Frozen gains:
- ID: −18.75%;
- OOD 0–25%: +5.17%;
- OOD 25–50%: +15.00%;
- OOD 50–75%: +20.05%;
- OOD 75–100%: +48.52%.

Message: optical assistance is condition dependent and becomes increasingly useful as electrical measurements leave source support.

## Fig. 6 — Strict T4 cross-rate unseen-profile generalization

Final panels:
- (a) profile-wise 1C→2C MAE, five-seed mean±SD;
- (b) profile-wise 2C→1C MAE, five-seed mean±SD;
- (c) direction-level and overall seed-cluster bootstrap 95% CI.

Headline values:
- 1C→2C MAE 1.795%;
- 2C→1C MAE 0.806%;
- overall MAE 1.301%, bootstrap 95% CI 0.961–1.677%.

Do not single out the worst seed in the main figure.

## Fig. 7 — Wavelength-noise robustness and calibrated uncertainty

Final 1×3 composition:
- (a) noise sigma 0/0.5/1/2 pm versus MAE for both directions;
- (b) Q95-AE versus wavelength-noise sigma;
- (c) representative SOC truth/prediction with 95% conformal interval.

Headline UQ annotation:
- nominal coverage: 95%;
- PICP: 95.04%;
- MPIW: 2.075% SOC.

Message: small direct wavelength perturbations cause smooth degradation and post-hoc conformal calibration provides approximately nominal uncertainty coverage.

# Main tables

## Table 1 — Dataset and operating profiles

Use the compact version in `docs/Q2_MAIN_TABLES_READY_CN.md`.

## Table 2 — Model/training configuration

Include V/I/W1/W2, window 64, hidden 24, dilations 1/2/4, kernel 3, AdamW, train-only normalization and 11,545 parameters.

## Table 3 — Conventional model comparison

Frozen candidates:
CNN, GRU, LSTM, Transformer, VI-TCN, VI+TF-TCN and RA-FBG-TCN.

Bold only the actual best metric entries; do not visually force the proposed model to appear best.

## Table 4 — Optical representation / OOD complementarity

Compact two-part table:
- Part A: raw W TCN, decoupled T/F TCN, ETMF-TF;
- Part B: parameter-matched VI versus VI+W aggregate for 1C→2C and 2C→1C.

Do not duplicate all OOD-bin values already displayed in Fig. 5.

## Table 5 — Strict T4 summary

Rows: 1C→2C and 2C→1C; overall seed-cluster MAE/CI can be a table footnote.

## Table 6 — omitted by default

Noise/UQ reliability numbers remain in Fig. 7 and Section 4.5 unless journal formatting later requires a dedicated table.

# Supplementary/internal-only candidates

- full representation/model development screens;
- whitening diagnostics;
- delta-t/CQR negative selection;
- Mamba/CrossFormer/ModernTCN/multi-delay exploration;
- external E1/E2 detail;
- WLTP structure audit;
- full seed-by-seed T4 matrix.

# Visual style and frozen QA

- full-width target: 183 mm;
- Times New Roman-style serif with publication-safe fallback;
- editable SVG primary;
- PDF vector secondary;
- TIFF 600 dpi LZW submission raster;
- PNG 300 dpi preview;
- no 3D plots or radar charts;
- consistent panel labels and VI/VI+W visual mapping;
- SOC errors reported as `% SOC`.

Canonical final figure package is generated by `.github/workflows/q2-publication-figures.yml` and must pass:
- no PDF text below 5 pt;
- collision audit: 0 FAIL and 0 WARN for every quantitative figure;
- source-data/figure contract checks.

Fig. 1(a) and Fig. 3 are the only intentionally unfinished main-display artwork.