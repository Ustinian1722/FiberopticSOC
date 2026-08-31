# Q2 manuscript figure/table callout map

Status: **FROZEN FOR MANUSCRIPT EDITING**

This file fixes the main-paper display order so English translation and later schematic replacement do not renumber the Results section.

## Section 1 — Introduction

No quantitative main figure is introduced here. References to the dual-FBG sensing concept remain textual. Do not preview development-history figures.

## Section 2 — Dataset and electrical–optical signal analysis

### Section 2.1 Dataset and test conditions

Callout: **Fig. 1** after the dataset/test description.

Fig. 1 roles:
- (a) experimental setup / dual-FBG sensing arrangement — placeholder until the user supplies or assembles the platform visual;
- (b) representative current + SOC trajectory;
- (c) synchronized W1/W2 response;
- (d) representative rate-dependent optical response.

Main table callout: **Table 1**, dataset and operating profiles.

### Sections 2.2–2.3 Dual-FBG sensing, decoupling and representation analysis

Callout: **Fig. 2** after introducing raw W1/W2 and decoupled T/F coordinates.

Fig. 2 roles:
- raw wavelength geometry;
- T/F geometry;
- correlation/conditioning comparison;
- matched transfer MAE comparison.

Primary message: physical interpretability and predictive representation quality are not identical objectives.

Do not introduce Mamba/CrossFormer/ModernTCN development screens in the main text.

## Section 3 — Methodology

### Sections 3.1–3.4

Callout: **Fig. 3** immediately after the overall method overview.

Fig. 3 remains a placeholder until the framework artwork is drawn. Its final number is already reserved.

Fig. 3 fixed content contract:
V/I/W1/W2 → train-only normalization → 64-sample causal window → 4→24 projection → residual TCN blocks d=1/2/4 → final causal state → SOC head → 95% residual conformal interval.

Main table callout: **Table 2**, model/training configuration.

## Section 4 — Experiments and results

### Section 4.1 Experimental settings and metrics

Use **Table 2** if not already placed in Section 3; do not duplicate the same hyperparameter table.

### Section 4.2 Conventional SOC performance

Callout order:
1. **Table 3** — CNN, GRU, LSTM, Transformer, VI-TCN, VI+TF-TCN, RA-FBG-TCN.
2. **Fig. 4** — representative RA-FBG-TCN trajectories and error distribution.

Narrative constraint: Table 3 is a conventional-accuracy benchmark. Do not claim that FBG universally improves an already in-distribution electrical estimator.

### Section 4.3 Electrical–optical complementarity under distribution shift

Callout: **Fig. 5**.

Fig. 5 is the main explanatory display:
- electrical training-support envelope and OOD test regions;
- VI versus VI+W error by OOD severity;
- relative optical gain versus electrical OOD fraction.

Use **Table 4** only as a compact representation/cross-rate summary; do not duplicate every OOD-bin value already plotted in Fig. 5.

### Section 4.4 Cross-rate unseen-profile generalization

Callout: **Fig. 6**, then **Table 5** if needed.

Fig. 6 roles:
- profile-wise 1C→2C mean±std across five seeds;
- profile-wise 2C→1C mean±std;
- seed-cluster bootstrap 95% CI for the two directions and overall result.

Do not highlight the single worst seed in the main display.

### Section 4.5 Wavelength-noise robustness and calibrated uncertainty

Callout: **Fig. 7**.

Fig. 7 roles:
- MAE versus 0/0.5/1/2 pm direct wavelength noise;
- Q95 absolute error versus wavelength noise;
- representative 95% conformal interval with PICP/MPIW annotation.

Optional **Table 6** is omitted by default. Reliability summary numbers should remain in Fig. 7 and the accompanying paragraph unless the target journal needs a separate table.

## Section 5 — Discussion

No new main quantitative display should be introduced. Discussion interprets Fig. 2, Fig. 5 and Fig. 6 jointly:
- raw optical coordinates versus physical decoupling;
- condition-dependent value of FBG sensing;
- asymmetric cross-rate transfer;
- calibration/sensor-transfer limitations as a short boundary statement.

External E1/E2 and WLTP audit remain supplementary/internal unless specifically requested during revision.

## Conclusion

No figure/table callouts.

## Fixed main-display count

Main figures: **7** (Fig. 1–7; Fig. 1a and Fig. 3 artwork pending).

Main tables: **5 by default** (Tables 1–5). Table 6 reliability summary is optional and should be omitted unless journal layout requires it.

This numbering should not change during routine manuscript editing.