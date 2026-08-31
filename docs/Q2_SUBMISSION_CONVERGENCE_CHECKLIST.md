# Q2 submission convergence checklist

Status: **NO NEW NUMERICAL EXPERIMENTS — MANUSCRIPT PRODUCTION ONLY**

The research phase is closed. Remaining work is limited to two schematic visuals, bibliography integration, English manuscript production and journal formatting.

## A. Main figures

- [x] Fig. 1 quantitative electrical/optical panels generated from SiC-18.
- [ ] Fig. 1(a) experimental setup / dual-FBG sensing visual: replace placeholder after the user provides/assembles the platform image.
- [x] Fig. 2 representation analysis finalized with reproducible Pearson correlation + frozen transfer summary.
- [ ] Fig. 3 RA-FBG-TCN + residual conformal framework artwork: placeholder only.
- [x] Fig. 4 conventional SOC prediction/error display generated from fixed T1.
- [x] Fig. 5 electrical-OOD optical-complementarity display generated from frozen analysis.
- [x] Fig. 6 five-seed strict cross-rate/unseen-profile display generated from frozen T4.
- [x] Fig. 7 wavelength-noise + 95% UQ display generated from frozen results.
- [x] Final quantitative-figure QA: zero <5-pt PDF text and 0 collision FAIL / 0 WARN on all six rendered data figures.

Canonical quantitative package: `docs/Q2_FINAL_FIGURE_FREEZE.md`.

## B. Tables

- [x] Table 1 dataset/test-condition table drafted and ready for manuscript formatting.
- [x] Table 2 model/training-configuration table drafted and ready for manuscript formatting.
- [x] Table 3 conventional model comparison frozen in `paper/source_data/table3_model_comparison.csv`.
- [x] Table 4 representation + parameter-matched cross-rate complementarity drafted.
- [x] Table 5 strict T4 summary drafted.
- [x] Table 6 omitted by default; noise/UQ numbers remain in Fig. 7.

Source: `docs/Q2_MAIN_TABLES_READY_CN.md`.

## C. Manuscript content

- [x] Front matter Chinese V1.
- [x] Section 1 Introduction revised to evidence-aligned Chinese V2.
- [x] Section 2 dataset/signal analysis revised to frozen Fig. 1/2 evidence.
- [x] Section 3 methodology code-aligned Chinese V1.
- [x] Section 4 experiments/results revised to frozen Fig. 4–7 / Table 3–5 evidence.
- [x] Section 5 discussion revised to final representation/OOD claims.
- [x] Conclusion Chinese V1.
- [x] Figure/table numbering and callout map frozen.
- [x] Numeric claim source-of-truth audit completed: `docs/Q2_NUMERIC_CLAIM_AUDIT.md`.
- [x] Recent seven-paper Introduction shortlist metadata/DOIs verified: `docs/Q2_VERIFIED_RECENT_REFERENCES.md`.
- [ ] Import/format the full bibliography through the final reference manager and add classic mechanism/filtering/TCN references.
- [ ] English translation and journal-style polishing.
- [ ] Final terminology pass: SOC, FBG, native/raw wavelength, T/F decoupling, operating-condition shift, conformal prediction.

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

## F. Remaining production sequence

1. Draw/insert Fig. 1(a) experimental setup panel.
2. Draw Fig. 3 framework using the already frozen architecture contract.
3. Build the English manuscript from the Chinese evidence-aligned sections.
4. Import final bibliography and resolve all citation numbers.
5. Journal-specific template/layout pass and final proofread.

## G. Stop rule

Do not start a new backbone, feature transformation, UQ method, external-data rescue calibration, or target-guided tuning during normal manuscript preparation. Only reopen experiments for a concrete reviewer request or a demonstrated factual inconsistency in the frozen evidence.