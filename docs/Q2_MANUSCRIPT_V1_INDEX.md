# Q2 manuscript V1 — assembly index

Status: **FULL CHINESE V1 CONTENT COMPLETE**

The manuscript has entered editing/figure-production stage. This index is the canonical assembly order.

## Front matter

`docs/Q2_FRONT_MATTER_DRAFT_CN.md`

Contains:
- recommended English/Chinese titles;
- Chinese abstract;
- English abstract skeleton;
- keywords;
- highlights;
- claim boundaries.

## Section 1 — Introduction

`docs/Q2_SECTION1_INTRODUCTION_DRAFT_CN.md`

Core positioning:
- SOC under operating-condition shift;
- FBG/mechanical sensing as complementary internal observation;
- recent FBG SOC work already establishes sensing feasibility, so novelty is not “FBG + neural network”;
- research gaps: compound C-rate/profile shift and raw wavelength versus physics-decoupled representation;
- RA-FBG-TCN + electrical-OOD complementarity + 95% conformal UQ.

## Section 2 — Dataset and electrical–optical signal analysis

`docs/Q2_SECTION2_DATA_SIGNAL_DRAFT_CN.md`

Subsections:
- 2.1 Dataset and test conditions
- 2.2 Dual-FBG sensing and thermo-mechanical decoupling
- 2.3 Signal characteristics and representation analysis

## Section 3 — Methodology

`docs/Q2_SECTION3_METHODOLOGY_DRAFT_CN.md`

Subsections:
- 3.1 Overall RA-FBG-TCN framework
- 3.2 Representation-aware raw optical input
- 3.3 Causal TCN SOC estimator
- 3.4 95% residual split conformal UQ

Architecture is code-consistent: 4→24 projection; three residual TCN blocks, dilation 1/2/4; two k=3 causal convolutions per block; GroupNorm+GELU; last causal state; 24→24→1 regression head; ~11.5k params.

## Section 4 — Experiments and results

`docs/Q2_SECTION4_RESULTS_DRAFT_CN.md`

Subsections:
- 4.1 Settings and metrics
- 4.2 Conventional SOC performance
- 4.3 Electrical–optical complementarity under distribution shift
- 4.3.1 Raw W versus T/F representation
- 4.4 Cross-rate unseen-profile generalization
- 4.5 Wavelength-noise robustness and 95% conformal UQ

Main evidence tables are frozen in `docs/Q2_PAPER_EVIDENCE_TABLES.md`.

## Section 5 — Discussion

`docs/Q2_SECTION5_DISCUSSION_DRAFT_CN.md`

Themes:
- physical decoupling versus predictive representation;
- why optical information becomes useful under electrical OOD;
- asymmetric cross-rate transfer and robustness;
- short scope/future-work boundary.

## Conclusion

`docs/Q2_CONCLUSION_DRAFT_CN.md`

Short result-centered conclusion.

## Canonical numerical evidence

- Final T1: `docs/Q2_FINAL_T1_PAPER_RESULTS.md`
- Representation: `config/Q2_REPRESENTATION_FREEZE.md`
- Electrical OOD: `docs/ELECTRICAL_OOD_FINDINGS.md`
- Final T4: `docs/Q2_FINAL_T4_RESULTS.md` on main branch / frozen formal artifact
- Compact paper tables: `docs/Q2_PAPER_EVIDENCE_TABLES.md`
- Experiment closure: `docs/Q2_EXPERIMENTS_CLOSED.md`

## Main-paper narrative in one sentence

> Dual-FBG optical information is not universally necessary when electrical measurements are well supported by training data, but becomes increasingly complementary under operating-condition shift; for this transfer-oriented SOC task, native wavelength coordinates coupled with a lightweight causal estimator provide a more robust learning representation than forced thermo-mechanical decoupling, while conformal calibration supplies a practical 95% uncertainty interval.

## Editing rule

From this point, changes should improve:
- prose compactness;
- literature citations;
- figure/table readability;
- terminology consistency;
- English translation quality.

Do not reopen numerical model development during routine manuscript editing.