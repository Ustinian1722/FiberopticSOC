# Q2 submission convergence checklist

Status: **NO NEW NUMERICAL EXPERIMENTS — MANUSCRIPT PRODUCTION ONLY**

The research phase is closed. Remaining work is limited to manuscript/figure production and editorial consistency.

## A. Main figures

- [x] Fig. 1 quantitative electrical/optical panels generated from SiC-18.
- [ ] Fig. 1(a) experimental setup / dual-FBG sensing visual: replace placeholder after the user provides/assembles the platform image.
- [x] Fig. 2 representation analysis generated from raw data + frozen transfer summary.
- [ ] Fig. 3 RA-FBG-TCN + residual conformal framework artwork: placeholder only; draw after all data figures are frozen.
- [x] Fig. 4 conventional SOC prediction/error display generated from fixed T1.
- [x] Fig. 5 electrical-OOD optical-complementarity display generated from frozen analysis.
- [x] Fig. 6 five-seed strict cross-rate/unseen-profile display generated from frozen T4.
- [x] Fig. 7 wavelength-noise + 95% UQ display generated from frozen results.
- [ ] Final data-figure QA: zero <5-pt PDF text and no unresolved reliable collision FAILs.

## B. Tables

- [x] Table 3 conventional model comparison frozen in `paper/source_data/table3_model_comparison.csv`.
- [ ] Table 1 dataset/test-condition table: format from Section 2 and SiC-18 metadata.
- [ ] Table 2 final model/training-configuration table: compact publication formatting only.
- [ ] Table 4 representation/OOD summary: compact, no duplication of Fig. 5 bins.
- [ ] Table 5 strict T4 summary: 1C→2C, 2C→1C and overall.
- [x] Table 6 is optional and omitted by default; noise/UQ numbers stay in Fig. 7.

## C. Manuscript content

- [x] Front matter Chinese V1.
- [x] Section 1 Introduction Chinese V1.
- [x] Section 2 dataset/signal analysis Chinese V1.
- [x] Section 3 methodology Chinese V1.
- [x] Section 4 experiments/results Chinese V1.
- [x] Section 5 discussion Chinese V1.
- [x] Conclusion Chinese V1.
- [x] Figure/table numbering and callout map frozen.
- [ ] Cross-check every numeric value in text against `paper/source_data/` or frozen result docs.
- [ ] Replace temporary citation placeholders with final bibliography entries/DOIs.
- [ ] English translation and journal-style polishing.
- [ ] Final terminology pass: SOC, FBG, raw wavelength, T/F decoupling, operating-condition shift, conformal prediction.

## D. Claims to keep in the main paper

1. Raw dual-FBG wavelength coordinates are a strong transfer-oriented predictive representation for the retained compact causal estimator.
2. Optical information is condition dependent: it is not universally necessary in well-supported electrical regions, but its relative benefit increases as electrical measurements move outside the training support.
3. The frozen estimator retains useful accuracy under simultaneous C-rate and unseen-profile shift, with clear directional asymmetry.
4. Direct wavelength noise up to 2 pm causes smooth rather than catastrophic degradation.
5. A 95% residual conformal interval achieves approximately nominal coverage (PICP 95.04%) with MPIW about 2.075% SOC.

## E. Material not promoted in the main narrative

Keep internal/supplementary unless reviewers request it:
- Mamba/CrossFormer/ModernTCN/multi-delay architecture-development history;
- whitening and delta-t negative screens;
- CQR negative selection;
- external E1 cross-cell negative result;
- external E2 near-positive but tail-robustness-limited result;
- external WLTP segmentation audit;
- full seed-by-seed T4 matrix.

These records remain in the repository for provenance but do not need to occupy the main manuscript.

## F. Stop rule

Do not start a new backbone, feature transformation, UQ method, external-data rescue calibration, or target-guided tuning during normal manuscript preparation. Only reopen experiments for a concrete reviewer request or a demonstrated factual inconsistency in the frozen evidence.