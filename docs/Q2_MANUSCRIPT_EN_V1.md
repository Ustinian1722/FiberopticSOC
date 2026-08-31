# Representation-aware dual-FBG optical sensing for robust battery state-of-charge estimation under operating-condition shifts

## Abstract

Accurate battery state-of-charge (SOC) estimation remains challenging when operating conditions depart from those represented in the training data. This study investigates the complementary value of implanted dual-fiber Bragg grating (FBG) observations under such condition shifts and develops a representation-aware electrical–optical SOC estimation framework. The directly measured wavelength coordinates (W1/W2) and thermo-mechanically decoupled temperature/force coordinates are first analyzed as alternative representations of the same two optical sensing degrees of freedom. Under matched cross-condition evaluation, the native wavelength representation provides more stable predictive transfer than explicit physical decoupling. A lightweight causal temporal convolutional network with only 11.5k trainable parameters is therefore constructed using voltage, current, W1, and W2, followed by residual split conformal calibration for uncertainty reporting. Under blocked mixed-condition interpolation, the proposed estimator achieves an MAE of 0.482% SOC, an RMSE of 0.593% SOC, and an R² of 0.999614. More importantly, parameter-matched electrical-only and electrical–optical comparisons show that, for 1C→2C transfer, incorporating FBG observations reduces the MAE from 2.151% to 1.632%. The relative optical benefit increases with electrical out-of-distribution severity and reaches 48.52% in the most shifted region. Under a stricter cross-rate plus unseen-profile protocol, the five-seed aggregate MAE is 1.301% SOC, with a seed-cluster bootstrap 95% confidence interval of 0.961–1.677% SOC. Direct wavelength perturbations up to 2 pm cause only limited performance degradation. In addition, the 95% residual conformal interval attains 95.04% empirical coverage with a mean interval width of 2.075% SOC. These results indicate that dual-FBG sensing is particularly valuable as a complementary internal-state observation when conventional electrical measurements move beyond their training support, while native optical coordinates provide a simple and robust representation for cross-condition data-driven SOC estimation.

**Keywords:** State of charge; Fiber Bragg grating; Multimodal sensing; Temporal convolutional network; Distribution shift; Conformal prediction

## Highlights

- Dual-FBG optical sensing complements electrical SOC estimation under operating-condition shift.
- Native W1/W2 coordinates transfer more reliably than explicit thermo-mechanical decoupling in the retained causal estimator.
- Optical benefit increases with electrical-OOD severity and reaches 48.52% in the most shifted region.
- Five-seed cross-rate unseen-profile evaluation yields an aggregate MAE of 1.301% SOC.
- A 95% residual conformal interval achieves 95.04% empirical coverage with a mean width of 2.075% SOC.

# 1. Introduction

Accurate state-of-charge (SOC) estimation is a core function of battery management systems (BMSs), underpinning energy management, power allocation, charge/discharge control, and safety protection in electric vehicles and energy-storage systems. Because SOC cannot be measured directly, it must be inferred from accessible signals such as terminal voltage, current, and temperature. This inference becomes particularly difficult under dynamic operation, where polarization, transient voltage response, nonlinear electrochemical behavior, and changing load profiles can substantially alter the relationship between electrical measurements and SOC. Consequently, an estimator that performs well around the training distribution may still degrade when the discharge rate or driving profile shifts during deployment.

Existing SOC estimation approaches can broadly be divided into model-based observer/filtering methods and data-driven methods that learn nonlinear mappings from measured signals to SOC. Deep temporal architectures, including convolutional neural networks (CNNs), recurrent neural networks, temporal convolutional networks (TCNs), and Transformers, have achieved high accuracy under increasingly complex battery datasets. Recent studies have also moved beyond single-condition accuracy toward wide-temperature modeling and multi-condition generalization [1,2]. A recent critical review emphasized standardized evaluation and highlighted transfer, few-shot, and continual-learning capability as important directions for future SOC estimation [1]. These developments indicate that low in-distribution MAE or RMSE alone is insufficient to characterize practical estimator robustness.

Mechanical and thermo-mechanical battery responses provide an additional sensing pathway beyond conventional electrical measurements. Lithium insertion and extraction induce electrode expansion, structural deformation, and stress evolution, creating mechanical observables coupled to lithium content and SOC. Recent work has shown that mechanical stress can complement voltage-based SOC estimation under dynamic load conditions [3]. Fiber Bragg grating (FBG) sensors are especially attractive because of their compact size, immunity to electromagnetic interference, embeddability, and high sensitivity to strain and temperature. Through shifts in the Bragg wavelength, FBGs enable in-situ observation of internal or surface thermo-mechanical behavior and have therefore become an increasingly important tool for multiphysics battery-state sensing.

FBG-assisted SOC estimation has advanced rapidly in recent years. Chu et al. developed a parallel-distributed FBG implantation scheme and decoupled reflected wavelengths to obtain internal strain and temperature, which were then combined with electrical variables in a CNN–Transformer SOC framework [4]. Ling et al. used implanted FBG sensors to collect in-situ thermo-mechanical information from a silicon-based lithium-ion battery and combined these observations with electrical measurements in a data-driven SOC estimator [5]. At the pack level, Liu et al. integrated distributed optical strain measurements with an adaptive state-estimation framework to address cell heterogeneity and monitoring complexity [6]. These studies demonstrate that FBG sensing can provide useful internal-state information for SOC estimation. The remaining question is therefore not whether FBGs can be used for SOC estimation, but how their information should be represented and evaluated when operating conditions shift beyond the source domain.

Two issues are particularly relevant. First, many existing studies establish accuracy by combining multiple operating conditions during model development or by evaluating within conventional train/test partitions. When discharge rate and driving profile change simultaneously, however, target data may occupy regions of electrical-input space that are poorly represented in the training set. In this compound-shift setting, an estimator relying predominantly on terminal electrical signals must extrapolate to previously unseen excitation amplitudes and temporal patterns. A more stringent protocol that excludes both the target C-rate and the target driving profile is therefore needed to assess whether electrical–optical sensing provides genuine robustness rather than merely improving interpolation within a well-covered distribution.

Second, dual-FBG systems are commonly transformed from two measured wavelength channels into physically interpretable temperature and strain/force coordinates using a sensitivity matrix [4]. Although this decoupling is valuable for sensing interpretation, the raw wavelength variables and the decoupled variables originate from the same two optical sensing degrees of freedom. The transformation therefore changes the representation rather than adding new measurements. In the present dataset, the transformation substantially increases linear coupling between the two coordinates: the absolute Pearson correlation of raw W1/W2 is 0.738 at 1C and 0.655 at 2C, whereas the corresponding values for the decoupled temperature/force coordinates are 0.982 and 0.985. Linear inversion also changes feature scaling and the propagation of measurement perturbations. It is thus not self-evident that a physically more interpretable coordinate system is also the most transferable representation for data-driven SOC estimation.

Motivated by these issues, this study develops a representation-aware dual-FBG SOC estimation framework, termed RA-FBG-TCN. Instead of introducing a large architecture as the primary contribution, the study focuses on the predictive role of optical sensing under distribution shift. Raw wavelength and thermo-mechanically decoupled coordinates are first compared under matched modeling and training conditions. Based on this comparison, voltage, current, W1, and W2 are directly modeled by a lightweight causal TCN that uses only current and historical observations. To quantify when optical sensing is most useful, an electrical out-of-distribution (OOD) severity score is defined solely from the training-current support, allowing FBG benefit to be analyzed without using target SOC labels. Finally, after freezing the point estimator, residual split conformal prediction is applied to obtain a 95% SOC prediction interval. Recent battery research has begun to adopt conformalized uncertainty methods to provide distribution-free uncertainty bounds for SOC estimation [7], motivating a lightweight post-hoc calibration layer rather than a second complex probabilistic network.

The main contributions of this work are summarized as follows.

1. **Representation-aware electrical–optical SOC estimation.** Implanted dual-FBG sensing is combined with voltage and current measurements, while native W1/W2 and thermo-mechanically decoupled T/F coordinates are analyzed as alternative representations of the same optical sensing information.
2. **Matched evaluation of native and physically decoupled optical representations.** Under identical model capacity, training budget, and cross-condition partitions, native W1/W2 provides more stable predictive transfer than explicit T/F decoupling for the retained lightweight causal estimator, demonstrating that physical interpretability and predictive representation quality need not coincide.
3. **Strict compound-shift evaluation and label-free OOD analysis.** The model is trained on five complete driving profiles at one C-rate and evaluated on an entirely unseen sixth profile at the opposite C-rate, with both directions repeated across five random seeds. Training-current-based OOD stratification further shows that the relative value of FBG information increases as electrical measurements leave source support.
4. **Reliability assessment through sensing perturbation and calibrated uncertainty.** Direct pm-scale wavelength perturbations quantify measurement robustness, while an independent calibration split constructs a 95% residual conformal interval that complements the SOC point estimate with empirically calibrated uncertainty coverage.

The remainder of this paper is organized as follows. Section 2 introduces the SiC-18 dual-FBG battery dataset, sensing principle, and electrical–optical representation characteristics. Section 3 presents the RA-FBG-TCN estimator and residual conformal uncertainty quantification. Section 4 reports conventional estimation performance, electrical-OOD complementarity, cross-rate unseen-profile generalization, wavelength-noise robustness, and uncertainty results. Section 5 discusses representation choice, condition-dependent optical value, transfer asymmetry, and scope limitations. Section 6 summarizes the main conclusions.

# 2. Dataset and electrical–optical signal analysis

## 2.1. Dataset and operating conditions

This study uses the publicly available SiC-18 dataset collected from a SiOx/C pouch-type lithium-ion cell instrumented with two embedded fiber Bragg grating sensors. The nominal cell capacity is approximately 2.5 Ah. An armored FBG and a bare FBG were embedded inside the cell to provide synchronized Bragg-wavelength observations associated with internal thermo-mechanical evolution during operation. Terminal voltage, current, and reference SOC were recorded simultaneously, enabling time-aligned electrical–optical state estimation.

The released data contain six representative dynamic driving profiles—HWFET, LA92, NEDC, NYCC, US06, and WLTC—tested at both 1C and 2C discharge rates. This produces 12 principal dynamic-discharge trajectories and approximately 68,086 valid samples. The profiles differ substantially in current amplitude, pulse density, and load-transition frequency, providing a useful test bed for SOC estimation under dynamic loading, C-rate changes, and unseen driving profiles. The dataset and primary variables are summarized in Table 1.

**Table 1. Dataset and sensing configuration used in this study.**

| Item | Description |
|---|---|
| Cell | SiOx/C pouch cell, approximately 2.5 Ah |
| Optical sensing | Embedded dual FBG: armored FBG + bare FBG |
| Dynamic profiles | HWFET, LA92, NEDC, NYCC, US06, WLTC |
| Discharge rates | 1C and 2C |
| Principal trajectories | 12 |
| Main predictor variables | Voltage, current, W1, W2 |
| Prediction target | SOC |
| Valid released samples | Approximately 68,086 |

The predictor vector contains terminal voltage V, current I, and the two FBG wavelength channels W1 and W2. SOC is used only as the supervised target and is never provided as an input. Variables that can directly encode discharge progress, including cumulative discharged capacity and absolute trajectory time, are excluded. For every training/validation/calibration/test partition, normalization statistics are estimated exclusively from the corresponding training data.

Figure 1 illustrates representative synchronized electrical and optical responses. The current exhibits rapid and frequent transients, whereas the FBG wavelength channels show smoother but distinct dynamic trajectories. For the NEDC example, both W1 and W2 shift substantially during discharge, but their amplitudes and local temporal patterns are not identical. The optical trajectories also change between 1C and 2C for the same driving profile. These observations indicate that electrical and optical measurements are synchronized but not dynamically redundant.

## 2.2. Dual-FBG sensing principle and thermo-mechanical decoupling

The center wavelength of an FBG is governed by

\[
\lambda_B=2n_{\mathrm{eff}}\Lambda,
\]

where \(n_{\mathrm{eff}}\) denotes the effective refractive index and \(\Lambda\) is the grating period. Changes in temperature and strain modify both quantities and therefore shift the reflected Bragg wavelength.

For the dual-FBG system considered here, the two sensing channels have different thermal and mechanical sensitivities. In the released dataset, the two wavelength responses and the decoupled temperature T and deformation-force-related quantity F approximately satisfy

\[
W_1=0.0208T+0.00054F,
\]

\[
W_2=0.0254T+0.00085F.
\]

The relation can be written as

\[
\begin{bmatrix}W_1\\W_2\end{bmatrix}
=
\mathbf{K}
\begin{bmatrix}T\\F\end{bmatrix},
\qquad
\mathbf{K}=
\begin{bmatrix}
0.0208 & 0.00054\\
0.0254 & 0.00085
\end{bmatrix}.
\]

When \(\mathbf{K}\) is invertible, the approximate inverse transformation is

\[
T=214.43W_1-136.23W_2,
\]

\[
F=-6407.67W_1+5247.23W_2.
\]

Thus, W1/W2 and T/F are not four independent sensing channels; they are two coordinate systems describing the same two optical sensing degrees of freedom. T/F offers clearer physical interpretation, whereas W1/W2 preserves the native coordinates directly produced by optical interrogation. Whether the physically decoupled representation is also the most transferable representation for data-driven SOC estimation is therefore an empirical question.

## 2.3. Electrical–optical characteristics and representation analysis

### 2.3.1. Dynamic-response differences under varying load

Under a dynamic driving cycle, terminal voltage is jointly affected by SOC, ohmic drop, polarization, and instantaneous current. The electrical response can therefore fluctuate rapidly even when SOC evolves smoothly. The implanted FBGs respond to internal thermo-mechanical evolution, whose characteristic time scales need not coincide with those of terminal electrical variables. Figure 1 illustrates this difference: current contains high-frequency pulses, whereas the wavelength channels evolve more smoothly while retaining local responses to changing load.

Moving from 1C to 2C directly expands the current range and alters the terminal electrical response, whereas the FBG wavelengths continue to reflect the accumulated and coupled internal thermo-mechanical state. The optical measurements are therefore treated as potentially complementary internal observations rather than duplicate measurements of V/I. Section 4 evaluates this hypothesis using parameter-matched models and electrical-OOD analysis.

### 2.3.2. Statistical structure of native and decoupled optical coordinates

Although W1/W2 and T/F contain the same number of optical degrees of freedom, their statistical structures differ substantially. Figure 2(a,b) shows the joint distributions of the two representations over all 12 dynamic trajectories. The absolute Pearson correlation between W1 and W2 is 0.738 at 1C and 0.655 at 2C. After thermo-mechanical decoupling, the corresponding absolute T/F correlations increase to 0.982 and 0.985, respectively, as summarized in Fig. 2(c).

This stronger coupling does not diminish the physical value of thermo-mechanical decoupling. The transformation provides clearer temperature and mechanical semantics, but it also changes feature scale, correlation structure, and perturbation propagation. For a transfer-oriented end-to-end SOC estimator, representation choice should therefore be determined empirically rather than assumed from interpretability alone.

### 2.3.3. Representation-aware predictive comparison

Native W1/W2 and decoupled T/F were compared under the same causal TCN architecture, training budget, and 1C→2C unseen-profile development protocol. Across six held-out-profile splits, the compact TCN using W1/W2 achieved an average MAE of 1.385% SOC, RMSE of 2.084% SOC, and Q95-AE of 4.873% SOC. Replacing W1/W2 with T/F increased these errors to 2.033%, 3.060%, and 6.835% SOC. A more complex electrical–thermo-mechanical fusion model reached an MAE of 1.569% SOC but still did not outperform the compact native-wavelength model, as shown in Fig. 2(d).

These results show that adding a physical interpretation layer does not necessarily improve predictive transfer. The final estimator therefore uses V, I, W1, and W2, while T/F is retained for sensing interpretation and representation ablation.

# 3. Methodology

## 3.1. Overview of the RA-FBG-TCN framework

A representation-aware dual-FBG temporal convolutional network, denoted RA-FBG-TCN, is developed for SOC estimation under dynamic loads and changing operating conditions. The framework first determines which optical coordinate system is more suitable for cross-condition prediction and then combines voltage, current, and the retained native FBG wavelengths in a compact causal temporal model. Residual split conformal prediction is applied after point-model training to provide calibrated uncertainty intervals.

At time t,

\[
\mathbf{x}_t=[V_t,I_t,W_{1,t},W_{2,t}],
\]

and a causal history window of length \(L=64\) is formed as

\[
\mathbf{X}_t=[\mathbf{x}_{t-L+1},\ldots,\mathbf{x}_t].
\]

The model outputs the SOC estimate \(\hat{y}_t\) at the final time step. Because the window contains only present and historical observations, no future information is used during inference. Figure 3 summarizes the complete pipeline: training-only normalization, causal windowing, dilated TCN encoding, SOC regression, and post-hoc 95% conformal calibration.

## 3.2. Representation-aware optical input selection

Let

\[
\Delta\boldsymbol{\lambda}=[\Delta\lambda_1,\Delta\lambda_2]^T,
\qquad
\mathbf{z}=[T,F]^T.
\]

Their relationship is

\[
\Delta\boldsymbol{\lambda}=\mathbf{K}\mathbf{z},
\qquad
\mathbf{z}=\mathbf{K}^{-1}\Delta\boldsymbol{\lambda}.
\]

The two representations therefore encode the same two measured optical degrees of freedom. Explicit decoupling improves physical semantics but does not introduce new sensing information. It also modifies feature correlation and scale. Representation choice is consequently treated as part of model design. Native W1/W2 gives lower average MAE, RMSE, and Q95 error in the matched cross-rate development comparison and is fixed as the final optical representation.

## 3.3. Causal TCN SOC estimator

### 3.3.1. Training-only normalization

Each channel is standardized using statistics computed only from training data:

\[
x'_{t,j}=\frac{x_{t,j}-\mu_j}{\sigma_j}.
\]

The same frozen \(\mu_j\) and \(\sigma_j\) are applied to validation, calibration, and test data. No target-domain statistics are re-estimated.

### 3.3.2. Dilated causal convolution

All temporal convolutions use left-only padding. For layer l with kernel size k and dilation \(d_l\),

\[
\mathbf{h}^{(l)}_t=
\sum_{i=0}^{k-1}\mathbf{W}^{(l)}_i
\mathbf{h}^{(l-1)}_{t-id_l}+\mathbf{b}^{(l)}.
\]

The four-channel input is projected to a 24-dimensional hidden space by a 1×1 convolution. Three residual temporal blocks are stacked with dilation factors 1, 2, and 4. Each block contains two causal Conv1D layers with kernel size 3, GroupNorm, GELU activation, and a residual connection:

\[
\mathbf{Z}_1=\mathrm{GELU}(\mathrm{GN}(\mathrm{Conv}_{c}(\mathbf{H};k=3,d))),
\]

\[
\mathbf{Z}_2=\mathrm{GN}(\mathrm{Conv}_{c}(\mathbf{Z}_1;k=3,d)),
\]

\[
\mathbf{H}_{out}=\mathrm{GELU}(\mathbf{H}+\mathbf{Z}_2).
\]

The three residual blocks yield an effective receptive field of 29 samples, while the 64-sample input window supplies a longer causal context.

### 3.3.3. Temporal representation and SOC regression

The final causal hidden state \(\mathbf{h}_t\) is passed to a two-layer regression head:

\[
\mathbf{u}_t=\mathrm{GELU}(\mathbf{W}_1\mathbf{h}_t+\mathbf{b}_1),
\]

\[
\hat{y}_t=\mathbf{W}_2\mathbf{u}_t+\mathbf{b}_2.
\]

The final model contains approximately 11.5k trainable parameters and is optimized using

\[
\mathcal{L}_{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2.
\]

AdamW is used with a learning rate of \(10^{-3}\), weight decay of \(10^{-4}\), gradient clipping at 2.0, and batch size 256. Training is limited to 50 epochs, with a minimum of 10 epochs and patience of 7. The checkpoint with the lowest validation MAE is retained.

**Table 2. Main RA-FBG-TCN architecture and training settings.**

| Item | Setting |
|---|---|
| Input variables | V, I, W1, W2 |
| Causal window | 64 samples |
| Input projection | 4→24, 1×1 Conv1D |
| Residual blocks | 3 |
| Dilations | 1, 2, 4 |
| Conv layers per block | 2 |
| Kernel size | 3 |
| Normalization / activation | GroupNorm / GELU |
| Regression head | 24→24→1 |
| Trainable parameters | ~11.5k |
| Optimizer / loss | AdamW / MSE |
| Learning rate / weight decay | 1×10⁻³ / 1×10⁻⁴ |
| Batch size | 256 |
| Gradient clipping | 2.0 |
| Maximum epochs | 50 |
| Early stopping | Validation MAE; minimum 10 epochs; patience 7 |

## 3.4. Residual split-conformal uncertainty quantification

Let the independent calibration set be

\[
\mathcal{D}_{cal}=\{(\mathbf{X}_i,y_i)\}_{i=1}^{n}.
\]

The frozen estimator produces calibration predictions \(\hat{y}_i\) and nonconformity scores

\[
r_i=|y_i-\hat{y}_i|.
\]

For \(\alpha=0.05\), the finite-sample corrected conformal rank is

\[
k=\lceil(n+1)(1-\alpha)\rceil,
\]

and \(q_{1-\alpha}\) is the k-th order statistic of the calibration residuals. The 95% interval for a new input is

\[
\mathcal{C}_{0.95}(\mathbf{X}_t)=[\hat{y}_t-q_{0.95},\hat{y}_t+q_{0.95}],
\]

with both endpoints clipped to [0,1]. The procedure uses only calibration residuals and does not adapt interval width using test labels.

PICP, MPIW, and MIS are used for uncertainty evaluation. PICP and MPIW are

\[
\mathrm{PICP}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\{y_i\in[L_i,U_i]\},
\]

\[
\mathrm{MPIW}=\frac{1}{N}\sum_{i=1}^{N}(U_i-L_i).
\]

For miscoverage \(\alpha\), the interval score is

\[
IS_i=(U_i-L_i)+\frac{2}{\alpha}(L_i-y_i)\mathbf{1}(y_i<L_i)+\frac{2}{\alpha}(y_i-U_i)\mathbf{1}(y_i>U_i),
\]

and MIS is its average over the test set. This post-hoc calibration provides an uncertainty interval with negligible additional online model complexity.

# 4. Experiments and results

## 4.1. Experimental settings and evaluation metrics

The framework is evaluated from five complementary perspectives: conventional mixed-condition accuracy, electrical–optical complementarity under distribution shift, optical-representation choice, strict cross-rate unseen-profile generalization, and reliability under wavelength perturbation and conformal uncertainty calibration. Unless otherwise stated, all experiments use the 64-sample causal window and training-only normalization described in Section 3. Test data are never used to estimate normalization statistics, select epochs, or tune the uncertainty interval.

Point performance is evaluated using MAE, RMSE, R², Q95-AE, and MaxAE. Prediction intervals are evaluated using PICP, MPIW, and MIS. Two evaluation regimes are emphasized: blocked mixed-condition interpolation and a stricter cross-rate unseen-profile protocol in which five profiles at one C-rate are used for training and the sixth profile at the opposite C-rate is held out completely.

## 4.2. Conventional SOC estimation performance

Table 3 compares representative sequence models under blocked mixed-condition interpolation.

**Table 3. Conventional blocked-interpolation SOC estimation performance.**

| Model | Parameters | MAE (% SOC) | RMSE (% SOC) | R² | Q95-AE (% SOC) |
|---|---:|---:|---:|---:|---:|
| VI-TCN | 11,497 | **0.231** | **0.296** | **0.999904** | **0.581** |
| VI+TF-TCN | 11,545 | 0.311 | 0.412 | 0.999814 | 0.825 |
| GRU | 10,801 | 0.426 | 0.549 | 0.999670 | 1.014 |
| RA-FBG-TCN | 11,545 | 0.482 | 0.593 | 0.999614 | 1.088 |
| LSTM | 14,129 | 0.548 | 0.708 | 0.999450 | 1.303 |
| CNN | 12,081 | 0.564 | 0.738 | 0.999403 | 1.414 |
| Transformer | 20,177 | 0.761 | 0.910 | 0.999090 | 1.584 |

All models attain high accuracy in this well-supported setting. VI-TCN gives the lowest point-estimation error, showing that voltage and current already provide strong SOC cues when the test distribution is represented during training. RA-FBG-TCN nevertheless achieves 0.482% MAE, 0.593% RMSE, and R²=0.999614 with only approximately 11.5k parameters. Representative trajectories are shown in Fig. 4. The role of FBG sensing is therefore not interpreted as a universal in-distribution accuracy gain; its contribution is examined primarily under operating-condition shift.

## 4.3. Electrical–optical complementarity under operating-condition shift

Two parameter-matched causal TCNs isolate the contribution of optical sensing. VI receives voltage and current, while VI+W receives voltage, current, W1, and W2. For 1C→2C transfer, the VI MAE is 2.151% SOC and decreases to 1.632% SOC with dual-FBG input, a relative reduction of approximately 24.1%. For 2C→1C, MAE decreases from 0.866% to 0.814% SOC.

An electrical support envelope is defined using the 0.5th–99.5th percentile range of source-training current. The fraction of samples outside this envelope within each test window defines a label-free electrical-OOD fraction. Figure 5 and Table 4 show the resulting stratification.

**Table 4. Parameter-matched electrical–optical comparison under increasing electrical-OOD severity for 1C→2C transfer.**

| Electrical-OOD level | VI MAE (% SOC) | VI+W MAE (% SOC) | Relative optical gain |
|---|---:|---:|---:|
| ID | **0.735** | 0.873 | −18.75% |
| OOD 0–25% | 1.550 | **1.470** | +5.17% |
| OOD 25–50% | 2.772 | **2.356** | +15.00% |
| OOD 50–75% | 3.665 | **2.930** | +20.05% |
| OOD 75–100% | 5.529 | **2.846** | **+48.52%** |

Within the fully supported ID region, FBG input does not improve the electrical-only estimator. Once the trajectory leaves source electrical support, however, optical benefit becomes positive and increases monotonically with OOD severity, reaching 48.52% in the most shifted windows. The auxiliary optical modality is therefore most useful when terminal electrical excitation is unfamiliar to the source-trained model.

The representation comparison in Fig. 2(d) further shows that the native W1/W2 TCN obtains an average MAE of 1.385% SOC across six 1C→2C unseen-profile development splits, compared with 2.033% for the same TCN using T/F. The more complex ETMF-TF model reaches 1.569% SOC but still does not outperform the native-wavelength estimator.

## 4.4. Cross-rate unseen-profile generalization

The final RA-FBG-TCN is evaluated using six held-out profiles in each transfer direction and five independent random seeds per split. For 1C→2C, the model achieves 1.795% MAE, 3.037% RMSE, R²=0.985689, and 6.804% Q95-AE. For 2C→1C, the corresponding values are 0.806%, 0.988%, 0.998480, and 1.830% SOC. Across both directions, the seed-cluster mean MAE is 1.301% SOC, with a bootstrap 95% confidence interval of 0.961–1.677% SOC.

**Table 5. Five-seed cross-rate unseen-profile performance.**

| Direction | MAE (% SOC) | RMSE (% SOC) | R² | Q95-AE (% SOC) |
|---|---:|---:|---:|---:|
| 1C→2C | 1.795 | 3.037 | 0.985689 | 6.804 |
| 2C→1C | **0.806** | **0.988** | **0.998480** | **1.830** |
| Overall | **1.301** | — | — | — |

The transfer directions are asymmetric. The 2C source data cover a wider high-current range, so most 1C target states remain within or near source support. In contrast, 1C→2C requires extrapolation toward stronger excitation and exhibits higher error and seed variability. This interpretation is consistent with the electrical-OOD analysis.

## 4.5. Wavelength-noise robustness and conformal uncertainty

Independent zero-mean Gaussian noise with standard deviations of 0.5, 1, and 2 pm is added to W1 and W2 while keeping the trained model and source-domain normalization fixed. As shown in Fig. 7(a,b), error increases smoothly. At 2 pm per channel, MAE rises by only approximately 1.67% relative to the clean baseline for 1C→2C and 4.57% for 2C→1C.

The frozen point estimator is then calibrated using residual split conformal prediction. The 95% nominal interval attains 95.04% empirical coverage, a mean prediction interval width of 2.075% SOC, a mean interval score of 0.02441, and an absolute-residual quantile of approximately 1.089% SOC. Figure 7(c) shows a representative calibrated interval. These results demonstrate that a simple post-hoc conformal layer can add uncertainty information without requiring a second probabilistic neural network.

# 5. Discussion

## 5.1. Physical decoupling versus predictive representation

The dual-FBG system allows the measured Bragg wavelengths to be transformed into temperature- and force-related variables using sensor sensitivities. This decoupling is useful for physical interpretation but does not add sensing degrees of freedom. In the present dataset, it also produces much stronger linear coupling between the two variables and changes feature scaling and perturbation propagation.

Matched experiments show that native W1/W2 yields lower average cross-condition error than both the same TCN using T/F and a more complex electrical–thermo-mechanical fusion model. This does not invalidate physical decoupling; rather, it demonstrates that sensing interpretation and predictive representation serve different objectives. For the present transfer-oriented SOC task, retaining native optical coordinates is the more effective engineering choice.

## 5.2. Why optical value emerges under distribution shift

The blocked-interpolation result shows that an electrical-only TCN can outperform the electrical–optical estimator when the test distribution is already well supported. Voltage and current contain strong SOC information in this regime, so auxiliary optical information can be partly redundant.

During 1C→2C transfer, however, target current increasingly enters regions rare or absent during source-rate training. A V/I-only estimator must extrapolate from unfamiliar electrical excitation. FBG wavelengths also reflect internal thermo-mechanical evolution and therefore provide a distinct state constraint. The OOD-stratified results make this effect visible: optical benefit is negative in the fully supported ID region, becomes positive once windows contain out-of-support electrical states, and reaches approximately 48.5% in the most severe OOD region.

For BMS deployment, this distinction matters because real operating conditions cannot be exhaustively represented during offline training. Internal optical sensing is therefore best viewed as an additional state constraint for difficult operating conditions rather than as a replacement for conventional voltage and current sensing.

## 5.3. Transfer asymmetry, sensing robustness, and uncertainty

The strict evaluation reveals a pronounced directional asymmetry: 2C→1C is considerably easier than 1C→2C. Training at 2C exposes the estimator to a broader high-current range, whereas training at 1C does not cover the highest-current 2C states and therefore requires more difficult extrapolation.

Reliability experiments provide complementary evidence. Native W1/W2 degrades smoothly under pm-scale perturbation, and the residual conformal interval reaches empirical coverage essentially equal to its 95% nominal target. The point estimator and uncertainty layer remain decoupled, keeping the online model compact while providing an explicit reliability measure.

## 5.4. Scope and future work

The present study focuses on operating-condition transfer within a fixed dual-FBG sensing configuration. Across different physical cells, FBG initial wavelength, bonding condition, strain-transfer efficiency, and sensor-specific sensitivity may vary and can alter optical signal level and dynamics. The current results should therefore not be interpreted as uncalibrated universal cross-cell generalization.

Future work should investigate sensor-aware calibration across multiple cells, broader temperature ranges, long-term sensor aging, and online adaptation. These sensing-calibration and domain-transfer issues may be more important for practical deployment than further increasing neural-network complexity.

# 6. Conclusion

This study investigated representation-aware dual-FBG optical sensing for battery SOC estimation under operating-condition shift. Native W1/W2 and thermo-mechanically decoupled T/F were analyzed as alternative representations of the same optical sensing information. Under matched cross-condition evaluation, W1/W2 exhibited more stable predictive transfer, motivating a compact RA-FBG-TCN using voltage, current, and two native wavelength channels.

Under blocked interpolation, RA-FBG-TCN achieved 0.482% MAE, 0.593% RMSE, and R²=0.999614. Parameter-matched OOD analysis showed that W1/W2 reduced overall 1C→2C MAE from 2.151% to 1.632% SOC, while optical benefit increased with electrical-OOD severity and reached approximately 48.5% in the most shifted region. The strict cross-rate unseen-profile evaluation yielded 1.795% MAE for 1C→2C and 0.806% for 2C→1C, with an overall seed-cluster MAE of 1.301% SOC and bootstrap 95% CI of 0.961–1.677% SOC.

Direct wavelength perturbations up to 2 pm caused only limited error growth. The 95% residual conformal interval attained 95.04% empirical coverage with a mean width of 2.075% SOC. The combined evidence indicates that the main value of internal optical sensing is not simply the addition of predictor variables, but the provision of state information from a distinct physical pathway when terminal electrical observations become weakly supported by the source distribution. Future work will extend the framework toward multi-cell sensor calibration, wider temperature conditions, sensor aging, and online adaptation.

# References — verified recent core set

[1] J. Yao, J. Kowal, “Towards a smarter battery management system: A critical review on deep learning-based state of charge estimation of lithium-ion batteries,” *Energy and AI*, 21 (2025) 100585. https://doi.org/10.1016/j.egyai.2025.100585.

[2] X. Wu et al., “Data-driven SOC estimation method for power batteries under driving cycle conditions and a wide temperature range,” *Energy*, 340 (2025) 139147. https://doi.org/10.1016/j.energy.2025.139147.

[3] Y. Fan et al., “Mechanical stress-based state-of-charge estimation for lithium-ion batteries via deep learning techniques,” *Energy*, 326 (2025) 136216. https://doi.org/10.1016/j.energy.2025.136216.

[4] Y. Chu et al., “Estimation of state-of-charge for lithium-ion batteries based on simultaneous internal strain and temperature monitoring by fiber optic sensors,” *Journal of Energy Storage*, 133 (2025) 117969. https://doi.org/10.1016/j.est.2025.117969.

[5] C. Ling et al., “In-situ data-driven high-precision SOC estimation for silicon-based lithium-ion batteries,” *Energy*, 349 (2026) 140609. https://doi.org/10.1016/j.energy.2026.140609.

[6] S. Liu, K. Li, J. Yu, “Adaptive estimation of battery pack state of charge with optical fibre strain measurements,” *Applied Energy*, 407 (2026) 127330. https://doi.org/10.1016/j.apenergy.2025.127330.

[7] K. L. Soon, L. T. Soon, “Enhancing reliability in electrified transportation: A conformalized quantile regression framework for battery state-of-charge uncertainty quantification,” *Journal of Power Sources*, 666 (2026) 239123. https://doi.org/10.1016/j.jpowsour.2025.239123.

> Reference-manager note: add classic SOC observer/filtering references, the original TCN reference, foundational conformal-prediction references, FBG sensing-mechanism references, and the original SiC-18 data/paper citation before journal submission.