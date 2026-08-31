# Q2 verified recent reference shortlist

Status: **METADATA CHECKED FOR ENGLISH MANUSCRIPT**

This list is intentionally short. It contains the recent papers that directly support the Introduction positioning; classic SOC/filtering/TCN/FBG-mechanism references can be added separately from the reference manager.

1. **Yao, J.; Kowal, J.** Towards a smarter battery management system: A critical review on deep learning-based state of charge estimation of lithium-ion batteries. *Energy and AI*, **21** (2025), 100585. DOI: **10.1016/j.egyai.2025.100585**.
   - Use for: standardized evaluation, limitations of pure accuracy benchmarking, emerging transfer/few-shot/continual-learning directions.

2. **Wu, X.; Yan, C.; Wang, L.; Dou, W.; Li, Y.; Gao, G.; Wang, J.; Fan, Y.; Tan, X.** Data-driven SOC estimation method for power batteries under driving cycle conditions and a wide temperature range. *Energy*, **340** (2025), 139147. DOI: **10.1016/j.energy.2025.139147**.
   - Use for: recent multi-condition SOC modelling and transferability under wide operating conditions.

3. **Fan, Y.; Yan, C.; Wu, X.; Li, Y.; Dou, W.; Gao, G.; Zhang, P.; Guan, Q.; Tan, X.** Mechanical stress-based state-of-charge estimation for lithium-ion batteries via deep learning techniques. *Energy*, **326** (2025), 136216. DOI: **10.1016/j.energy.2025.136216**.
   - Use for: mechanical sensing as a complementary SOC observable and a representative Energy-style battery-paper methodology/experiment structure.

4. **Chu, Y.; Ren, F.; Zhou, X.; Li, T.; et al.** Estimation of state-of-charge for lithium-ion batteries based on simultaneous internal strain and temperature monitoring by fiber optic sensors. *Journal of Energy Storage*, **133** (2025), 117969. DOI: **10.1016/j.est.2025.117969**.
   - Use for: implanted/parallel-distributed FBG, wavelength decoupling, internal strain/temperature sensing and CNN–Transformer SOC estimation.
   - Before final bibliography export, import the full six-author record from Crossref/Scopus/reference manager rather than manually expanding `et al.`.

5. **Ling, C.; Lin, Q.; Luo, J.; Lin, Z.; Gong, Q.; Zhang, M.; Xie, H.; Yu, Z.; Yang, Y.; Yue, H.; Dong, H.; Shi, Z.; Lin, Z.; Su, J.; Yang, S.** In-situ data-driven high-precision SOC estimation for silicon-based lithium-ion batteries. *Energy*, **349** (2026), 140609. DOI: **10.1016/j.energy.2026.140609**.
   - Use for: the directly related SiC-18/FGB deep-learning study; reports 0.635% RMSE in its own evaluation setting.
   - Do not compare its headline RMSE numerically with the strict cross-rate unseen-profile T4 as if the protocols were equivalent.

6. **Liu, S.; Li, K.; Yu, J.** Adaptive estimation of battery pack state of charge with optical fibre strain measurements. *Applied Energy*, **407** (2026), 127330. DOI: **10.1016/j.apenergy.2025.127330**.
   - Use for: optical strain sensing at pack level, sensor heterogeneity and adaptive state estimation.

7. **Soon, K. L.; Soon, L. T.** Enhancing reliability in electrified transportation: A conformalized quantile regression framework for battery state-of-charge uncertainty quantification. *Journal of Power Sources*, **666** (2026), 239123. DOI: **10.1016/j.jpowsour.2025.239123**.
   - Use for: recent conformalized SOC uncertainty quantification and the need for calibrated prediction intervals.

## Citation-positioning constraints

- Do not claim the present paper is the first FBG-based SOC estimator.
- Do not use the Energy 2026 SiC-18 paper's 0.635% RMSE as a directly matched baseline for the strict T4 protocol.
- Cite the JES 2025 paper as evidence that FBG wavelength decoupling is physically meaningful, not as proof that decoupling must be the best predictive representation.
- Cite the 2026 pack-level Applied Energy paper when discussing sensor/cell heterogeneity and future calibration-aware transfer.
- The current manuscript uses residual split conformal, not CQR; the JPS 2026 paper is a related UQ precedent rather than the implemented method source.
