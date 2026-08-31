# Q2 experiments closed

Date: 2026-08-31
Status: **COMPUTATIONAL EXPERIMENTS CLOSED — MANUSCRIPT/FIGURE STAGE**

The manuscript evidence package is sufficient for a conventional SCI Q2 battery paper. No additional architecture search, representation search, hyperparameter search, external-data rescue experiment, or UQ-method search is planned.

## Frozen main-paper evidence

1. Dataset/signal analysis: SiC-18, six dynamic profiles, 1C/2C, V/I + dual FBG.
2. Representation analysis: native W1/W2 versus physics-decoupled T/F; raw W retained for cross-condition estimator.
3. Conventional T1 benchmark: 7 matched baseline models; RA-FBG-TCN MAE 0.482%, RMSE 0.593%, R2 0.999614.
4. Electrical-OOD complementarity diagnostic: 1C->2C VI+W reduces overall MAE from 2.151% to 1.632%, and optical gain rises to 48.52% in the highest OOD bin.
5. Strict T4: 5 seeds, 12 rate+unseen-profile splits per seed, overall seed-cluster MAE 1.301% (95% CI 0.961–1.677%).
6. Wavelength-noise robustness: small monotonic degradation through 2 pm perturbation.
7. Residual conformal UQ: primary 95% interval PICP 95.04%, MPIW 2.075% SOC.
8. Optional second-dataset paragraph: fixed-sensor cross-rate mean MAE 14.92% -> 13.42%, 3/4 cells improved; not a cross-cell claim.

## Final narrative

The paper is not an architecture-novelty paper. It is a battery sensing/generalization paper with a lightweight temporal estimator:

> Dual-FBG optical observations are not universally required when electrical measurements remain within the training support, but they provide increasingly valuable complementary information under operating-condition shift. For this transfer-oriented task, direct wavelength coordinates are more reliable than forced thermo-mechanical decoupling in the retained compact causal estimator. The frozen model maintains strong cross-rate unseen-profile accuracy, wavelength-noise robustness, and calibrated 95% uncertainty intervals.

## Main manuscript structure

1. Introduction
2. Dataset and electrical–optical signal analysis
3. Methodology: RA-FBG-TCN + 95% residual conformal
4. Experiments and results
   - 4.1 Settings and metrics
   - 4.2 Conventional SOC performance
   - 4.3 Electrical–optical complementarity under distribution shift
   - 4.4 Cross-rate unseen-profile generalization
   - 4.5 Noise robustness and uncertainty
   - optional short external paragraph
5. Discussion
6. Short conclusion if required by journal format

## No-more-compute rule

From this commit forward, work is limited to:

- manuscript prose;
- reference/literature alignment;
- publication figures;
- formatting tables;
- consistency checks;
- supplementary/reviewer-response preparation.

Any new numerical experiment must be treated as a separate future revision request, not as routine continuation of the current paper.