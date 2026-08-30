# Q1 competitor literature matrix (2024–2026)

Date: 2026-08-31

This matrix is used to prevent novelty drift while designing the FiberopticSOC Q1 mainline.

| Work | Journal / year | Sensing / inputs | Main method | Validation emphasis | What is already claimed | Gap relevant to our paper |
|---|---|---|---|---|---|---|
| Lithium-ion battery expansion mechanism and Gaussian process regression based SOC estimation with expansion characteristics | Energy 292, 2024, 130541 | expansion/thickness + electrical context | electrochemical-thermal-mechanical analysis + GPR | expansion-SOC relation | expansion is a useful SOC characteristic | does not address dual-FBG representation or adaptive electrical-mechanical residual fusion |
| Mechanical stress-based SOC estimation for lithium-ion batteries via deep learning techniques | Energy 326, 2025, 136216 | voltage + mechanical stress features | PLO-TCNUltra-SE, feature selection | lithium + sodium datasets | mechanical stress improves SOC, optimized TCN/SE | generic stress feature fusion; not bounded correction / hierarchical cross-rate unseen-profile protocol |
| Estimation of SOC based on simultaneous internal strain and temperature monitoring by fiber optic sensors | Journal of Energy Storage 133, 2025, 117969 | V/I + internally decoupled temperature/strain | CNN-Transformer | static/dynamic conditions, 0/25/40 C | implanted dual FBG and internal strain/temperature improve SOC by >40% | direct multi-source fusion; no explicit negative-transfer control |
| Physics-enhanced hybrid KAN with dynamic coupling for interpretable SOC estimation | Applied Energy 400, 2025, 126533 | electrochemical/thermal/mechanical information | physics-enhanced KAN + dynamic gating + physics loss | 174 batteries, diverse conditions | dynamic coupling and physics consistency improve robustness/generalization | much stronger scale than SiC-18; motivates our need for second-dataset validation and restrained claims |
| In-situ data-driven high-precision SOC estimation for silicon-based lithium-ion batteries | Energy 349, 2026, 140609 | SiC-18 electrical + implanted FBG thermo-mechanical data | feature engineering + noise augmentation + CNN-GRU-Attention | six dynamic profiles x two rates | high-precision SOC on the source public data family | no need for us to replicate pooled accuracy only; our evaluation must emphasize condition-shift/generalization |
| SOC estimation using implanted dual-FBG sensing and Local-to-Global Temporal Former | Journal of Energy Storage 180, 2026, 124293 | I/U + decoupled T/F | Local-to-Global Temporal Former | multi-temperature and unseen driving cycle; multiple cells | dual-FBG T/F + strong temporal model already established | rules out novelty claims based on decoupling or local/global temporal modeling alone |
| Adaptive estimation of battery pack SOC with optical fibre strain measurements | Applied Energy 407, 2026, 127330 | distributed optical strain + pack voltage | GPR-assisted adaptive UKF | pack level, multiple chemistries/conditions | reliability/adaptation of optical-assisted filtering | rules out generic 'adaptive optical weighting' as novelty; our mechanism must be tied to bounded residual fusion and domain-shift validation |
| Enhancing reliability in electrified transportation: conformalized quantile regression for SOC UQ | Journal of Power Sources 666, 2026, 239123 | battery features | FeatureFormer + CQR | calibrated SOC intervals | distribution-free uncertainty bounds | CQR can only be a supporting reliability layer for our work |
| Estimate SOC in lithium-ion batteries with unknown data | Applied Energy 389, 2025, 125736 | electrical/physics-guided features | physics-informed meta learning | unknown data distributions | cross-distribution SOC generalization | motivates strong unseen-condition protocol; our differentiator is internal mechanical sensing rather than meta-learning alone |
| Adaptive multi-scale diagonal state-space model with position-aware Performer attention | Journal of Energy Storage 163, 2026, 122094 | conventional battery signals | S4D + Performer | unseen cycles, cross-chemistry/temperature | sophisticated temporal architecture and zero-shot generalization | warns against competing only through a more complicated sequence backbone |

## Design implications

1. **Do not sell dual-FBG decoupling as our novelty.** It is an input representation inherited from the sensing literature.
2. **Do not sell TCN + GRU + attention as the core novelty.** Recent papers already contain stronger temporal architectures.
3. **Do not sell adaptive weighting alone.** Physics-enhanced dynamic gating and adaptive optical UKF already exist.
4. **Do not sell conformal UQ alone.** SOC-specific CQR work exists.
5. Our Q1 differentiation should be the combination of:
   - electrical estimator retained as a strong base path;
   - internal mechanical/thermo-mechanical information used as a bounded residual correction;
   - explicit study of when naive fusion helps or hurts under condition shift;
   - hierarchical conventional -> unseen-profile -> cross-rate -> cross-rate+unseen evaluation;
   - second mechanical-sensing dataset to move beyond the single SiC-18 cell;
   - optional raw-W/T-F consistency and UQ only after the main mechanism is validated.

## Primary DOI list

- 10.1016/j.energy.2024.130541
- 10.1016/j.energy.2025.136216
- 10.1016/j.est.2025.117969
- 10.1016/j.apenergy.2025.126533
- 10.1016/j.energy.2026.140609
- 10.1016/j.est.2026.124293
- 10.1016/j.apenergy.2025.127330
- 10.1016/j.jpowsour.2025.239123
- 10.1016/j.apenergy.2025.125736
- 10.1016/j.est.2026.122094
