# Q2 reviewer-style manuscript self-audit

Status: editorial/reviewer-risk audit only. No new numerical experiments are required.

## 1. “Why use FBG if VI-TCN is better in the conventional interpolation table?”

**Risk:** high if the manuscript presents FBG as a universal accuracy booster.

**Current evidence:** VI-TCN is best under blocked interpolation (MAE 0.231% versus 0.482% for RA-FBG-TCN), but parameter-matched VI+W reduces 1C→2C MAE from 2.151% to 1.632%. Optical benefit rises from negative in the fully supported ID region to +48.52% in the most severe electrical-OOD bin.

**Manuscript treatment:** explicitly define the contribution as condition-dependent sensing complementarity. Do not claim that adding FBG universally improves SOC accuracy. Keep Fig. 5 as the explanatory centerpiece.

## 2. “The main dataset contains only one physical instrumented cell.”

**Risk:** moderate/high if the manuscript implies cross-cell generalization.

**Current evidence:** the primary SiC-18 analysis is a fixed-sensing-system study across six dynamic profiles, two C-rates, compound rate/profile shift, five seeds, direct wavelength perturbation, and UQ.

**Manuscript treatment:** state in Section 5.4 that the quantitative conclusions concern operating-condition transfer within one fixed dual-FBG sensing configuration. Do not write “universal cross-cell generalization.” Frame multi-cell calibration as the next deployment problem rather than as a missing claim.

## 3. “Is the five-seed cross-rate experiment an untouched independent holdout after representation selection?”

**Risk:** moderate if described as a pristine confirmatory holdout.

**Current evidence:** W/T-F representation selection used matched cross-rate development splits; the final five-seed protocol repeats the frozen retained configuration across both transfer directions and six held-out profiles.

**Manuscript treatment:** describe T4 as a strict split-wise cross-rate unseen-profile repeated evaluation after the representation/model configuration was fixed. Avoid wording such as “untouched independent holdout” or “preregistered confirmatory test.”

## 4. “Does conformal prediction guarantee 95% coverage under the cross-rate OOD task?”

**Risk:** high if the coverage result is generalized beyond its evaluated setting.

**Current evidence:** PICP 95.04% and MPIW 2.075% SOC are measured on the blocked mixed-condition T1 test set using an independent calibration split.

**Manuscript treatment:** explicitly state that the reported empirical coverage pertains to the blocked T1 calibration/test regime. Do not imply formal marginal coverage under arbitrary rate/profile shift. Future work can consider covariate-shift-aware or adaptive conformal calibration.

## 5. “Why call the method representation-aware if the final network is a compact TCN?”

**Risk:** moderate.

**Response:** representation-aware refers to the explicit experimental selection between native W1/W2 and physically decoupled T/F before fixing the estimator. The paper contribution is not a novel TCN block; it is the sensing/representation/generalization framework. Keep architecture claims modest and emphasize causality, compactness, and reproducibility.

## 6. “Is the electrical-OOD score target-label leakage?”

**Risk:** low if clearly defined.

**Current definition:** source training-current 0.5th–99.5th percentile envelope; window OOD fraction is the proportion of observed current samples outside the envelope. No SOC label is used.

**Manuscript treatment:** retain the term “label-free electrical-OOD severity” and define the support statistics as source-training-only.

## 7. “Are raw W and T/F truly different information sources?”

**Risk:** low because the manuscript explicitly says no.

**Response:** they are alternative coordinates of the same two optical degrees of freedom. Do not stack W+T+F as independent features in the narrative. Physical interpretation and predictive representation are deliberately separated.

## 8. “Is the 2 pm robustness statement an absolute error increase or a relative increase?”

**Risk:** low but easy to miswrite.

**Correct statement:** at 2 pm/channel, MAE increases by approximately 1.67% relative to the clean 1C→2C MAE and 4.57% relative to the clean 2C→1C MAE. These are relative changes, not +1.67 or +4.57 percentage points of SOC.

## 9. “Why does the paper compare with CNN/GRU/LSTM/Transformer but not claim the proposed model is best?”

**Response:** the conventional table establishes that the compact estimator has strong baseline accuracy. The paper’s central contribution is transfer-oriented sensing complementarity, not leaderboard dominance under a well-supported mixed-condition split. Avoid ranking rhetoric.

## 10. “How should the previous Energy 2026 SiC-18 paper be handled?”

**Risk:** high if novelty is framed as ‘FBG + deep learning for SOC.’

**Manuscript treatment:** cite the previous SiC-18/FBG SOC work prominently in the Introduction. State that FBG feasibility has already been established. Define the present gap as (i) native versus physically decoupled representation, (ii) compound C-rate + unseen-profile shift, (iii) electrical-support-dependent optical complementarity, and (iv) compact post-hoc UQ.

## Required editorial patches before submission

- Add one sentence in Section 5.4 explicitly stating that the main quantitative dataset contains one instrumented physical cell.
- Add one sentence in the UQ results/discussion stating that the 95.04% coverage is evaluated in blocked T1 and is not claimed as an OOD coverage guarantee.
- Cite Bai et al. for TCN and Lei et al./Shafer–Vovk for conformal prediction in Section 3.
- Cite a classical SOC filtering reference (Plett) in Introduction paragraph 2.
- Add early embedded-FBG battery references in the sensing-motivation paragraph.
- Keep the external multi-cell dataset out of the main accuracy claims; it may be cited only to motivate sensor-specific calibration in future work.

## No-experiment conclusion

None of the above reviewer risks requires reopening architecture search, hyperparameter search, T4, or external-data modeling. They are addressed by claim calibration, citation quality, and clearer scope.