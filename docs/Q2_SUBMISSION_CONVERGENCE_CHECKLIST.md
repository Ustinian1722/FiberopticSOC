# Q2 submission convergence checklist

Status: **NO NEW NUMERICAL EXPERIMENTS — SUBMISSION MANUSCRIPT PRODUCTION ONLY**

The research phase is closed. Quantitative figures, tables, numerical claims, reviewer-scope safeguards, the English manuscript V2, and the core bibliography are assembled. Remaining work is limited to two schematic visuals, author/journal metadata, final reference-manager integration, and copy-editing/layout.

## A. Main figures

- [x] Fig. 1 quantitative electrical/optical panels generated from SiC-18.
- [ ] Fig. 1(a) experimental setup / dual-FBG sensing visual: replace placeholder after the platform image is assembled.
- [x] Fig. 2 representation analysis finalized with reproducible Pearson correlation + frozen transfer summary.
- [ ] Fig. 3 RA-FBG-TCN + residual conformal framework artwork: placeholder only.
- [x] Fig. 4 conventional SOC prediction/error display generated from fixed T1.
- [x] Fig. 5 electrical-OOD optical-complementarity display generated from frozen analysis.
- [x] Fig. 6 five-seed strict cross-rate/unseen-profile display generated from frozen T4.
- [x] Fig. 7 wavelength-noise + 95% UQ display generated from frozen results.
- [x] Final quantitative-figure QA: all PDF text ≥5 pt and 0 collision FAIL / 0 WARN across the six rendered data figures.

Canonical quantitative package: `docs/Q2_FINAL_FIGURE_FREEZE.md`; workflow run `33368316998`; artifact `q2-publication-figures-final`.

## B. Tables

- [x] Table 1 dataset/test-condition table drafted in English.
- [x] Table 2 code-aligned model/training-configuration table drafted in English.
- [x] Table 3 conventional model comparison frozen in `paper/source_data/table3_model_comparison.csv`.
- [x] Table 4 parameter-matched electrical-OOD complementarity drafted in English; representation comparison is retained primarily in Fig. 2(d) and text to avoid duplication.
- [x] Table 5 strict five-seed T4 summary drafted in English.
- [x] Additional noise/UQ table omitted by default; these results remain in Fig. 7 and text.

## C. Manuscript content

- [x] Final 250-word English abstract.
- [x] Title, Keywords, Highlights.
- [x] Section 1 Introduction.
- [x] Section 2 Dataset and electrical–optical signal analysis.
- [x] Section 3 Methodology, aligned with actual architecture and training code.
- [x] Section 4 Experiments and results.
- [x] Section 5 Discussion.
- [x] Section 6 Conclusion.
- [x] English Fig. 1–7 captions.
- [x] Data availability section with Mendeley Data DOI `10.17632/ft6rtwt8vm.1`.
- [x] Code availability section pointing to the reproducible repository/workflows.
- [x] Full English manuscript V2 assembled and linted by the canonical workflow.
- [x] Latest assembled length: approximately **8,235 words including tables, captions, and core references; ~7,086 words before figure captions/references**.
- [x] 16/16 canonical numerical claims present.
- [x] Banned/obsolete claim check PASS.
- [x] Reviewer-risk safeguards PASS.
- [x] Single-cell scope explicitly stated.
- [x] Blocked-T1 conformal coverage explicitly not generalized as a formal arbitrary-OOD guarantee.
- [x] Same-platform prior result (0.635% RMSE) acknowledged; novelty positioned on representation, compound shift, OOD complementarity, and UQ rather than conventional leaderboard accuracy.
- [x] Internal drafting metadata removed from assembled manuscript.

Canonical manuscript inputs:
- `docs/Q2_ENGLISH_FRONT_INTRO_V1.md`
- `docs/Q2_ABSTRACT_EN_FINAL.md`
- `docs/Q2_ENGLISH_SECTIONS2_3_V1.md`
- `docs/Q2_ENGLISH_SECTIONS4_6_V2.md`
- `docs/Q2_MAIN_FIGURE_CAPTIONS_EN.md`
- `docs/Q2_REFERENCES_EN_CORE.md`

Canonical assembly:
- script: `analysis/assemble_q2_english_manuscript.py`
- workflow: `.github/workflows/q2-manuscript-assembly.yml`
- output: `paper/manuscript/Q2_ENGLISH_MANUSCRIPT_V2.md`
- latest successful run with data/code availability additions: `33383890398`

## D. References

- [x] Seven recent 2025–2026 battery/FBG/SOC/UQ references metadata checked.
- [x] Classical Plett SOC-EKF reference added.
- [x] Bai–Kolter–Koltun TCN reference added.
- [x] Lei et al. and Shafer–Vovk conformal references added.
- [x] Early battery-FBG sensing references added.
- [x] Original/current SiC-18 source paper cited in Introduction, Dataset, sensing equation context, and Discussion.
- [x] Core manuscript bibliography maintained in `docs/Q2_REFERENCES_EN_CORE.md`.
- [ ] Import the final complete set into the authors' reference manager and regenerate numbering/style for the selected journal.

## E. Main claims retained

1. Native dual-FBG wavelength coordinates are a strong transfer-oriented predictive representation for the retained compact causal estimator.
2. Optical information is condition dependent: it is not universally necessary in well-supported electrical regions, but its relative benefit increases as electrical measurements move outside source support.
3. The frozen estimator retains useful accuracy under simultaneous C-rate and unseen-profile shift, with clear directional asymmetry.
4. Direct wavelength noise up to 2 pm causes smooth rather than catastrophic degradation.
5. In the blocked mixed-condition calibration/test regime, the 95% residual conformal interval attains PICP 95.04% with MPIW 2.075% SOC.

## F. Material not promoted in the main narrative

Keep internal/supplementary unless reviewers request it:
- Mamba/CrossFormer/ModernTCN/multi-delay architecture-development history;
- whitening and delta-t negative screens;
- CQR negative selection;
- external E1 cross-cell negative result;
- external E2 near-positive but tail-robustness-limited result;
- external WLTP segmentation audit;
- full seed-by-seed T4 matrix.

These records remain in the repository for provenance but do not occupy the main manuscript.

## G. Remaining production sequence

1. Draw/insert Fig. 1(a) experimental setup / dual-FBG sensing panel.
2. Draw Fig. 3 RA-FBG-TCN + residual-conformal framework using the frozen architecture contract.
3. Final English copy-edit: reduce local repetition, standardize abbreviations and journal punctuation, and check equation/figure/table references after layout.
4. Insert author names, affiliations, corresponding-author information, acknowledgements/funding, CRediT contributions, and competing-interest declaration from the author team.
5. Import the final bibliography through the reference manager and regenerate journal-style citations.
6. Apply the selected target-journal template and perform final page-proof review.

## H. Stop rule

Do not start a new backbone, feature transformation, UQ method, external-data rescue calibration, or target-guided tuning during manuscript preparation. Reopen experiments only for a concrete reviewer request or a demonstrated factual inconsistency in the frozen evidence.