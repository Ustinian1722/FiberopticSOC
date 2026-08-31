# Q2 submission convergence checklist

Status: **NO NEW NUMERICAL EXPERIMENTS — FINAL MANUSCRIPT PRODUCTION ONLY**

The research phase is closed. The quantitative figures, tables, numerical claims, English manuscript V1, and core bibliography shortlist are now assembled. Remaining work is limited to two schematic visuals, citation-manager integration, editorial polishing, and journal formatting.

## A. Main figures

- [x] Fig. 1 quantitative electrical/optical panels generated from SiC-18.
- [ ] Fig. 1(a) experimental setup / dual-FBG sensing visual: replace placeholder after the platform image is assembled.
- [x] Fig. 2 representation analysis finalized with reproducible Pearson correlation + frozen transfer summary.
- [ ] Fig. 3 RA-FBG-TCN + residual conformal framework artwork: placeholder only.
- [x] Fig. 4 conventional SOC prediction/error display generated from fixed T1.
- [x] Fig. 5 electrical-OOD optical-complementarity display generated from frozen analysis.
- [x] Fig. 6 five-seed strict cross-rate/unseen-profile display generated from frozen T4.
- [x] Fig. 7 wavelength-noise + 95% UQ display generated from frozen results.
- [x] Final quantitative-figure QA: zero <5-pt PDF text and 0 collision FAIL / 0 WARN on all six rendered data figures.

Canonical quantitative package: `docs/Q2_FINAL_FIGURE_FREEZE.md` and artifact `q2-publication-figures-final` from workflow run `33368316998`.

## B. Tables

- [x] Table 1 dataset/test-condition table drafted in English.
- [x] Table 2 model/training-configuration table drafted in English.
- [x] Table 3 conventional model comparison frozen in `paper/source_data/table3_model_comparison.csv`.
- [x] Table 4 representation + parameter-matched electrical-OOD complementarity drafted in English.
- [x] Table 5 strict T4 summary drafted in English.
- [x] Table 6 omitted by default; noise/UQ numbers remain in Fig. 7.

## C. Manuscript content

- [x] Title, Abstract, Keywords, Highlights — English V1.
- [x] Section 1 Introduction — English V1.
- [x] Section 2 Dataset and electrical–optical signal analysis — English V1.
- [x] Section 3 Methodology — English V1, code aligned.
- [x] Section 4 Experiments and results — English V1.
- [x] Section 5 Discussion — English V1.
- [x] Section 6 Conclusion — English V1.
- [x] English Fig. 1–7 captions drafted.
- [x] Full English manuscript assembly completed and linted by workflow run `33372238989`.
- [x] Approximate assembled length: **8,018 words including tables and captions**.
- [x] 16/16 canonical numerical claims present in the assembled manuscript.
- [x] Banned/obsolete claim check PASS.
- [x] Section-structure and intended-placeholder checks PASS.
- [x] Numeric claim source-of-truth audit: `docs/Q2_NUMERIC_CLAIM_AUDIT.md`.

Canonical English sources:
- `docs/Q2_ENGLISH_FRONT_INTRO_V1.md`
- `docs/Q2_ENGLISH_SECTIONS2_3_V1.md`
- `docs/Q2_ENGLISH_SECTIONS4_6_V1.md`
- `docs/Q2_MAIN_FIGURE_CAPTIONS_EN.md`

Assembly artifact: `q2-english-manuscript-v1`.

## D. References

- [x] Seven recent battery/FBG/SOC/UQ references metadata checked.
- [x] Canonical SOC, FBG, TCN, LSTM, Transformer, AdamW and conformal references shortlisted in `docs/Q2_REFERENCE_SHORTLIST_VERIFIED.md`.
- [ ] Import the shortlist plus the original SiC-18 data paper into the final reference manager.
- [ ] Regenerate citation numbers in journal style.
- [ ] Add dataset/Zenodo citation if required by the target journal.

## E. Claims retained in the main paper

1. Native dual-FBG wavelength coordinates are a strong transfer-oriented predictive representation for the retained compact causal estimator.
2. Optical information is condition dependent: it is not universally necessary in well-supported electrical regions, but its relative benefit increases as electrical measurements move outside the training support.
3. The frozen estimator retains useful accuracy under simultaneous C-rate and unseen-profile shift, with clear directional asymmetry.
4. Direct wavelength noise up to 2 pm causes smooth rather than catastrophic degradation.
5. A 95% residual conformal interval achieves approximately nominal coverage (PICP 95.04%) with MPIW about 2.075% SOC.

## F. Material not promoted in the main narrative

Keep internal/supplementary unless reviewers request it:
- Mamba/CrossFormer/ModernTCN/multi-delay architecture-development history;
- whitening and delta-t negative screens;
- CQR negative selection;
- external E1 cross-cell negative result;
- external E2 near-positive but tail-robustness-limited result;
- external WLTP segmentation audit;
- full seed-by-seed T4 matrix.

These records remain in the repository for provenance but do not need to occupy the main manuscript.

## G. Remaining production sequence

1. Draw/insert Fig. 1(a) experimental setup / dual-FBG sensing panel.
2. Draw Fig. 3 RA-FBG-TCN + residual-conformal framework using the frozen architecture contract.
3. Perform one English editorial pass: shorten long sentences, remove repeated claims, standardize terms and abbreviations.
4. Import final bibliography and resolve citation numbering.
5. Apply target-journal template/layout and run final proofread.

## H. Stop rule

Do not start a new backbone, feature transformation, UQ method, external-data rescue calibration, or target-guided tuning during manuscript preparation. Only reopen experiments for a concrete reviewer request or a demonstrated factual inconsistency in the frozen evidence.