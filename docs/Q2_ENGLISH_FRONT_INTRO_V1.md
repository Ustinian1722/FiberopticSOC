# English manuscript V1 — Front matter and Introduction

## Title

**Representation-aware dual-FBG optical sensing for robust battery state-of-charge estimation under operating-condition shifts**

## Abstract

Accurate battery state-of-charge (SOC) estimation remains challenging when operating conditions depart from those represented in the training data. This study investigates the complementary value of implanted dual-fiber Bragg grating (FBG) observations under such condition shifts and develops a representation-aware electrical–optical SOC estimation framework. The directly measured wavelength coordinates (W1/W2) and thermo-mechanically decoupled temperature/force coordinates are first analyzed as alternative representations of the same two optical sensing degrees of freedom. Under matched cross-condition evaluation, the native wavelength representation provides more stable predictive transfer than explicit physical decoupling. A lightweight causal temporal convolutional network with only 11.5k trainable parameters is therefore constructed using voltage, current, W1, and W2, followed by residual split conformal calibration for uncertainty reporting. Under blocked mixed-condition interpolation, the proposed estimator achieves an MAE of 0.482% SOC, an RMSE of 0.593% SOC, and an R² of 0.999614. More importantly, parameter-matched electrical-only and electrical–optical comparisons show that, for 1C→2C transfer, incorporating FBG observations reduces the MAE from 2.151% to 1.632%. The relative optical benefit increases with electrical out-of-distribution severity and reaches 48.52% in the most shifted region. Under a stricter cross-rate plus unseen-profile protocol, the five-seed aggregate MAE is 1.301% SOC, with a seed-cluster bootstrap 95% confidence interval of 0.961–1.677% SOC. Direct wavelength perturbations up to 2 pm cause only limited performance degradation. In addition, the 95% residual conformal interval attains 95.04% empirical coverage with a mean interval width of 2.075% SOC. These results indicate that dual-FBG sensing is particularly valuable as a complementary internal-state observation when conventional electrical measurements move beyond their training support, while native optical coordinates provide a simple and robust representation for cross-condition data-driven SOC estimation.

## Keywords

State of charge; Fiber Bragg grating; Multimodal sensing; Temporal convolutional network; Distribution shift; Conformal prediction

## Highlights

- Dual-FBG optical sensing complements electrical SOC estimation under operating-condition shift.
- Native W1/W2 coordinates transfer more reliably than explicit thermo-mechanical decoupling in the retained causal estimator.
- Optical benefit increases with electrical-OOD severity and reaches 48.52% in the most shifted region.
- Five-seed cross-rate unseen-profile evaluation yields an aggregate MAE of 1.301% SOC.
- A 95% residual conformal interval achieves 95.04% empirical coverage with a mean width of 2.075% SOC.

# 1. Introduction

Accurate state-of-charge (SOC) estimation is a core function of battery management systems (BMSs), underpinning energy management, power allocation, charge/discharge control, and safety protection in electric vehicles and energy-storage systems. Because SOC cannot be measured directly, it must be inferred from accessible signals such as terminal voltage, current, and temperature. This inference becomes particularly difficult under dynamic operation, where polarization, transient voltage response, nonlinear electrochemical behavior, and changing load profiles can substantially alter the relationship between electrical measurements and SOC. Consequently, an estimator that performs well around the training distribution may still degrade when the discharge rate or driving profile shifts during deployment.

Existing SOC estimation approaches can broadly be divided into model-based observer/filtering methods and data-driven methods that learn nonlinear mappings from measured signals to SOC. Deep temporal architectures, including convolutional neural networks (CNNs), recurrent neural networks, temporal convolutional networks (TCNs), and Transformers, have achieved high accuracy under increasingly complex battery datasets. Recent studies have also moved beyond single-condition accuracy toward wide-temperature modeling, cross-material transfer, and multi-condition generalization [1,2]. A recent critical review explicitly emphasized the need for more standardized evaluation protocols and highlighted transfer, few-shot, and continual-learning capability as important directions for future SOC estimation [1]. These developments indicate that low in-distribution MAE or RMSE alone is insufficient to characterize the practical robustness of an SOC estimator.

Mechanical and thermo-mechanical battery responses provide an additional sensing pathway beyond conventional electrical measurements. Lithium insertion and extraction induce electrode expansion, structural deformation, and stress evolution, creating mechanical observables that are coupled to lithium content and SOC. Recent work has shown that mechanical stress can complement voltage-based SOC estimation under dynamic load conditions [3]. Fiber Bragg grating (FBG) sensors are especially attractive for such applications because of their compact size, immunity to electromagnetic interference, embeddability, and high sensitivity to strain and temperature. Through shifts in the Bragg wavelength, FBGs enable in-situ observation of internal or surface thermo-mechanical behavior and have therefore become an increasingly important tool for multiphysics battery-state sensing.

FBG-assisted SOC estimation has advanced rapidly in the past two years. Chu et al. developed a parallel-distributed FBG implantation scheme and decoupled reflected wavelengths to obtain internal strain and temperature, which were then combined with electrical variables in a CNN–Transformer SOC framework [4]. Ling et al. used implanted FBG sensors to collect in-situ thermo-mechanical information from a silicon-based lithium-ion battery and combined these observations with electrical measurements in a data-driven SOC estimator [5]. At the pack level, Liu et al. integrated distributed optical strain measurements with an adaptive state-estimation framework to address cell heterogeneity and monitoring complexity [6]. These studies demonstrate that FBG sensing can provide useful internal-state information for SOC estimation. The remaining question is therefore not whether FBGs can be used for SOC estimation, but how their information should be represented and evaluated when operating conditions shift beyond the source domain.

Two issues are particularly relevant. First, many existing studies establish accuracy by combining multiple operating conditions during model development or by evaluating within conventional train/test partitions. When discharge rate and driving profile change simultaneously, however, the target data may occupy regions of the electrical-input space that are poorly represented in the training set. In this compound-shift setting, an estimator relying predominantly on terminal electrical signals must extrapolate to previously unseen excitation amplitudes and temporal patterns. A more stringent protocol that excludes both the target C-rate and the target driving profile is therefore needed to assess whether electrical–optical sensing provides genuine robustness rather than merely improving interpolation within a well-covered distribution.

Second, dual-FBG systems are commonly transformed from the two measured wavelength channels into physically interpretable temperature and strain/force coordinates using a sensitivity matrix [4]. Although this decoupling is valuable for sensing interpretation, the raw wavelength variables and the decoupled variables originate from the same two optical sensing degrees of freedom. The transformation therefore changes the representation rather than adding new measurements. In the present dataset, the transformation substantially increases the linear coupling between the two coordinates: the absolute Pearson correlation of raw W1/W2 is 0.738 at 1C and 0.655 at 2C, whereas the corresponding values for the decoupled temperature/force coordinates are 0.982 and 0.985. Linear inversion also changes feature scaling and the propagation of measurement perturbations. It is thus not self-evident that a physically more interpretable coordinate system is also the most transferable representation for data-driven SOC estimation.

Motivated by these issues, this study develops a representation-aware dual-FBG SOC estimation framework, termed RA-FBG-TCN. Instead of introducing a large architecture as the primary contribution, the study focuses on the predictive role of optical sensing under distribution shift. Raw wavelength and thermo-mechanically decoupled coordinates are first compared under matched modeling and training conditions. Based on this comparison, voltage, current, W1, and W2 are directly modeled by a lightweight causal TCN that uses only current and historical observations. To quantify when optical sensing is most useful, an electrical out-of-distribution (OOD) severity score is defined solely from the training-current support, allowing FBG benefit to be analyzed without using target SOC labels. Finally, after freezing the point estimator, residual split conformal prediction is applied to obtain a 95% SOC prediction interval. Recent battery research has begun to adopt conformalized uncertainty methods to provide distribution-free uncertainty bounds for SOC estimation [7], motivating the inclusion of a lightweight post-hoc calibration layer rather than a second complex probabilistic network.

The main contributions of this work are summarized as follows.

1. **Representation-aware electrical–optical SOC estimation.** An implanted dual-FBG sensing framework is combined with voltage and current measurements, while raw W1/W2 and thermo-mechanically decoupled T/F coordinates are analyzed as alternative representations of the same optical sensing information.

2. **Matched evaluation of native and physically decoupled optical representations.** Under identical model capacity, training budget, and cross-condition partitions, native W1/W2 yields more stable predictive transfer than explicit T/F decoupling for the retained lightweight causal estimator, demonstrating that physical interpretability and predictive representation quality need not coincide.

3. **Strict compound-shift evaluation and label-free OOD analysis.** The model is trained on five complete driving profiles at one C-rate and evaluated on an entirely unseen sixth profile at the opposite C-rate, with both transfer directions repeated across five random seeds. A training-current-based electrical-OOD stratification further shows that the relative value of FBG information increases systematically as electrical measurements leave source support.

4. **Reliability assessment through sensing perturbation and calibrated uncertainty.** Direct pm-scale wavelength perturbations are used to quantify measurement robustness, while an independent calibration split is used to construct a 95% residual conformal interval that complements the SOC point estimate with empirically calibrated uncertainty coverage.

The remainder of this paper is organized as follows. Section 2 introduces the SiC-18 dual-FBG battery dataset, sensing principle, and electrical–optical representation characteristics. Section 3 presents the RA-FBG-TCN estimator and residual conformal uncertainty quantification. Section 4 reports conventional estimation performance, electrical-OOD complementarity, cross-rate unseen-profile generalization, wavelength-noise robustness, and uncertainty results. Section 5 discusses representation choice, the condition-dependent value of optical sensing, transfer asymmetry, and scope limitations. The final section summarizes the main conclusions and outlines future work.

## Recent-reference key used in this draft

[1] Yao, J.; Kowal, J. *Energy and AI* 21 (2025) 100585. DOI: 10.1016/j.egyai.2025.100585.

[2] Wu, X. et al. *Energy* 340 (2025) 139147. DOI: 10.1016/j.energy.2025.139147.

[3] Fan, Y. et al. *Energy* 326 (2025) 136216. DOI: 10.1016/j.energy.2025.136216.

[4] Chu, Y. et al. *Journal of Energy Storage* 133 (2025) 117969. DOI: 10.1016/j.est.2025.117969.

[5] Ling, C. et al. *Energy* 349 (2026) 140609. DOI: 10.1016/j.energy.2026.140609.

[6] Liu, S.; Li, K.; Yu, J. *Applied Energy* 407 (2026) 127330. DOI: 10.1016/j.apenergy.2025.127330.

[7] Soon, K. L.; Soon, L. T. *Journal of Power Sources* 666 (2026) 239123. DOI: 10.1016/j.jpowsour.2025.239123.

The final reference numbering will be regenerated after importing classic SOC, TCN, conformal-prediction, and FBG-mechanism references into the manuscript reference manager.