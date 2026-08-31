# Q2 manuscript submission readiness

Status: **SCIENTIFIC CONTENT FROZEN / JOURNAL-FORMATTING DECISION PENDING**

Canonical manuscript: `paper/manuscript/Q2_ENGLISH_MANUSCRIPT_V2.md`

Latest validated scientific state:
- strict paper-completion experiments: workflow `33404899282`, all jobs successful;
- final source-level copyedit commit: `d0bb3fa52946a904c6500d798a19bb123a417355`;
- final canonical assembly: workflow `33414855050`, successful;
- manuscript length: approximately 8,719 words including tables, captions, and references;
- abstract: exactly 250 words;
- canonical numerical claims: 22/22 present;
- training-protocol consistency: PASS;
- Table 1–7 / Fig. 1–7 consistency: PASS;
- single-cell scope safeguard: PASS;
- conformal-coverage scope safeguard: PASS;
- banned/overclaim check: PASS.

## 1. Scientific evidence chain — complete

### Dataset / signal analysis
- SiC-18 source and sensing configuration documented.
- Six dynamic profiles at 1C and 2C documented.
- Leakage-prone variables excluded from predictors.
- W1/W2 and T/F correctly treated as alternative coordinates of the same two optical degrees of freedom.
- Development representation screen documented.

### Methodology
- Final RA-FBG-TCN architecture frozen at 11,545 parameters.
- Causal window = 64 samples.
- TCN dilation = 1/2/4.
- AdamW / learning rate / weight decay / clipping / batch size documented.
- Conventional blocked training and strict-transfer training explicitly separated.
- Strict transfer uses source-only frozen 29–65 epoch budgets; no target-domain early stopping.
- Residual split conformal UQ documented with finite-sample rank correction.

### Main experiments
- Conventional blocked interpolation: complete.
- Strict seven-backbone comparison: complete.
- Five-seed parameter-matched VI / VI+TF / VI+W ablation: complete.
- Electrical-OOD contribution diagnostic: complete and explicitly secondary/explanatory.
- Five-seed final cross-rate unseen-profile evaluation: complete.
- pm-scale raw-wavelength noise robustness: complete.
- Residual conformal calibration: complete.

## 2. Claims permitted in the submission

1. RA-FBG-TCN is a compact electrical–optical SOC estimator with approximately 11.5k parameters.
2. In the seed-42 strict backbone benchmark, RA-FBG-TCN has the lowest pooled MAE across 12 cross-rate/unseen-profile splits (0.864% SOC), ranks first for 2C→1C, and third for 1C→2C.
3. In the formal five-seed matched ablation, adding W1/W2 reduces 1C→2C MAE from 2.162% to 1.795% SOC, approximately a 17% relative reduction; the seed-cluster bootstrap interval for the absolute gain is positive at the reported precision.
4. W1/W2 and T/F have closely comparable formal difficult-transfer accuracy; no universal representation superiority is claimed.
5. Optical value is direction/support dependent: electrical-only VI is best in 2C→1C, whereas optical information is beneficial in the harder 1C→2C transfer.
6. The fixed development OOD diagnostic shows increasing optical gain with increasing electrical-support violation, reaching 48.52% in the most shifted bin.
7. Final five-seed RA-FBG-TCN MAE is 1.795% for 1C→2C, 0.806% for 2C→1C, with overall seed-cluster MAE 1.301% and 95% bootstrap CI 0.961–1.677% SOC.
8. Direct W1/W2 perturbations up to 2 pm produce limited degradation in the tested protocol.
9. The blocked-regime 95% residual conformal interval obtains 95.04% empirical PICP and 2.075% SOC MPIW.

## 3. Claims explicitly prohibited

- first FBG SOC estimator;
- first TCN / Transformer / Mamba battery SOC estimator;
- universal superiority of raw W over T/F;
- universal benefit of optical sensing under all conditions;
- universal cross-cell generalization;
- formal 95% conformal coverage under arbitrary cross-rate/OOD shift;
- direct leaderboard superiority over the 0.635% RMSE reported by the source SiC-18 Energy paper, because protocols differ;
- exact reproduction of the source paper's CGA model; the comparator is CGA-Matched/CGA-style.

## 4. Artwork status

Ready quantitative figures:
- Fig. 1(b–d): synchronized electrical/optical dataset signals.
- Fig. 2: W/T-F representation characteristics and development screen.
- Fig. 4: conventional SOC estimation.
- Fig. 5: electrical-OOD optical contribution.
- Fig. 6: five-seed strict generalization.
- Fig. 7: wavelength noise + conformal UQ.

Intentionally pending artwork:
- Fig. 1(a): experimental platform / implanted FBG configuration. The author will provide or assemble the source experimental photographs/schematic.
- Fig. 3: RA-FBG-TCN + conformal framework architecture artwork. The manuscript already contains the full technical specification required to draw it.

These two placeholders were deliberately permitted during manuscript convergence and are the only remaining figure placeholders.

## 5. Bibliography status

`docs/Q2_REFERENCE_AUDIT.md` verifies the current 13 core references against publisher/canonical metadata. No scientific DOI/title mismatch was identified. Final reference-manager export remains a journal-formatting task.

## 6. Author metadata still required before submission

- author list and order;
- affiliations;
- corresponding author information;
- funding acknowledgements;
- CRediT author contribution statement;
- conflict-of-interest declaration;
- data/code statements adjusted to the selected journal template if needed.

## 7. Decision gate

No additional SiC-18 experiment is required for the current paper story. The next material decision is the **target journal**. Journal choice determines:
- final title-page layout;
- reference style;
- abstract/highlight constraints;
- figure dimensions/resolution requirements;
- whether a graphical abstract is required or encouraged;
- cover-letter wording;
- supplementary-material structure;
- final word-count and section-format adjustments.

Until the journal is selected, further formatting would create duplicate work. The scientific manuscript itself is frozen.