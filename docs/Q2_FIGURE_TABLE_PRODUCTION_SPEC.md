# Q2 main-paper figure/table production specification

Status: **FROZEN PRODUCTION PLAN**

The goal is a conventional Energy/JES-style visual package: approximately 7 main figures and 5–6 main tables. Avoid development-history figures.

## Fig. 1 — Dataset, sensing arrangement and representative electrical–optical signals

Recommended composition:

- (a) sensing/test schematic or a clean redraw of the implanted dual-FBG battery setup (use original source citation; do not copy copyrighted figure directly without permission);
- (b) representative 1C dynamic trajectory: SOC + current/voltage;
- (c) synchronized W1/W2 trajectory;
- (d) optional 2C comparison for the same profile.

Message: electrical and optical channels are synchronized but respond differently to dynamic load.

## Fig. 2 — Dual-FBG representation analysis

Recommended 2×2 panel:

- (a) W1–W2 density/scatter;
- (b) T–F density/scatter;
- (c) correlation comparison for raw W and decoupled T/F;
- (d) covariance condition number / matched transfer MAE comparison.

Annotate approximate condition numbers:
- W: ~6–7;
- T/F: ~107–119.

Message: physical decoupling improves semantic interpretability but produces a more correlated/ill-conditioned predictive coordinate; raw W yields stronger matched transfer for the retained TCN.

## Fig. 3 — RA-FBG-TCN framework

One clean methodology schematic:

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

Use a small inset for each TCN block: causal conv k=3 → GroupNorm → GELU → causal conv k=3 → GroupNorm → residual + GELU.

Message: compact causal estimation plus post-hoc calibrated UQ.

## Fig. 4 — Conventional SOC prediction examples

Do not make a model-zoo figure. Use RA-FBG-TCN only.

Recommended panels:
- (a) representative SOC truth vs prediction for a smoother profile;
- (b) representative truth vs prediction for a highly dynamic profile;
- (c) absolute error over time or SOC;
- (d) overall absolute-error distribution.

Table 3 already carries the model comparison.

Message: RA-FBG-TCN has sub-percent conventional point-estimation accuracy.

## Fig. 5 — Electrical-OOD severity versus optical benefit

This is the most important explanatory figure.

Recommended panels:
- (a) training-current support envelope and example 1C→2C test current showing OOD regions;
- (b) bar/line: OOD bin versus VI MAE and VI+W MAE;
- (c) OOD bin versus relative optical gain: -18.75%, +5.17%, +15.00%, +20.05%, +48.52%;
- optional (d) scatter of window OOD fraction versus optical gain with a trend line.

Message: optical assistance is condition dependent and becomes valuable as electrical measurements move out of training support.

## Fig. 6 — Strict T4 cross-rate unseen-profile generalization

Recommended panels:
- (a) profile-wise MAE for 1C→2C, mean across 5 seeds with error bars;
- (b) profile-wise MAE for 2C→1C;
- (c) direction-level MAE/RMSE summary or seed-cluster bootstrap CI;
- optional (d) representative difficult/easy prediction trajectory only if space permits.

Headline annotations:
- 1C→2C MAE 1.795%;
- 2C→1C MAE 0.806%;
- overall MAE 1.301%, bootstrap 95% CI 0.961–1.677%.

Do not single out the worst seed in the main figure.

## Fig. 7 — Wavelength-noise robustness and calibrated uncertainty

Recommended 1×2 or 2×2 composition:

- (a) noise sigma 0/0.5/1/2 pm versus MAE for both directions;
- (b) optional Q95-AE versus noise sigma;
- (c) representative SOC truth/prediction with 95% conformal interval;
- (d) optional coverage/interval-width annotation.

Headline UQ annotation:
- 95% nominal;
- PICP 95.04%;
- MPIW 2.075% SOC.

Message: small direct wavelength perturbations cause smooth degradation; conformal calibration provides well-calibrated uncertainty bounds.

# Main tables

## Table 1 — Dataset and operating profiles

Columns:
- cell / capacity / sensing setup;
- profile;
- rate;
- sample count or trajectory duration if useful.

Keep simple.

## Table 2 — Model/training configuration

Include:
- inputs V/I/W1/W2;
- window 64;
- hidden 24;
- TCN dilations 1/2/4;
- kernel 3;
- optimizer AdamW;
- train-only normalization;
- parameter count 11,545.

## Table 3 — Conventional model comparison

Use frozen T1 table:
- CNN, GRU, LSTM, Transformer, VI-TCN, VI+TF-TCN, RA-FBG-TCN;
- Params, MAE, RMSE, R2, Q95-AE.

Do not visually mark RA-FBG-TCN as statistically best when it is not; bold best metric entries normally.

## Table 4 — Optical representation / OOD complementarity

Preferred compact format rather than a large architecture zoo:

Part A: representation screen
- VI+W TCN;
- VI+TF TCN;
- optional ETMF-TF.

Part B or adjacent text: cross-rate VI vs VI+W aggregate.

If Fig. 5 fully covers the OOD bins, do not duplicate all bin values in Table 4.

## Table 5 — Strict T4 summary

Rows:
- 1C→2C;
- 2C→1C;
- overall if meaningful.

Columns:
- MAE, RMSE, R2, Q95-AE, optional CI.

## Table 6 — Reliability summary (optional)

Keep only if journal space allows:
- 2 pm noise relative MAE increase;
- 95% UQ: PICP, MPIW, MIS.

Otherwise move these numbers into Fig. 7 annotations and omit Table 6.

# Supplementary-only visual candidates

- full representation/model development screen;
- whitening/conditioning details;
- delta-t/CQR negative ablations;
- external E1/E2 detail;
- WLTP structure audit;
- seed-by-seed full T4 matrix.

# Visual style

- A4 two-column-compatible sizing;
- Times New Roman or journal-equivalent serif for plots;
- consistent line widths and marker sizes;
- no 3D plots;
- no decorative radar charts unless absolutely necessary;
- figure labels `(a)`, `(b)`, ... in consistent upper-left positions;
- use the same color/marker mapping for VI versus VI+W across all figures;
- report SOC errors in `% SOC` consistently;
- keep figure legends short and avoid model acronyms not used in the manuscript.
