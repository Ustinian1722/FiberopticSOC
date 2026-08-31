# Representation-aware dual-FBG optical sensing for robust battery state-of-charge estimation under operating-condition shifts

## Abstract

Accurate battery state-of-charge (SOC) estimation becomes challenging when discharge rate and driving profile differ from those represented during training. This study develops a compact dual-fiber Bragg grating (FBG)-assisted electrical–optical SOC estimation framework and evaluates its accuracy, cross-condition generalization, sensing contribution, and uncertainty. Native wavelength coordinates (W1/W2) and thermo-mechanically decoupled temperature/force coordinates are first compared as alternative representations of the same optical sensing information. A lightweight causal temporal convolutional network (RA-FBG-TCN) with 11.5k trainable parameters is then constructed using voltage, current, W1, and W2. Under blocked mixed-condition interpolation, RA-FBG-TCN achieves 0.482% MAE, 0.593% RMSE, and R²=0.999614. Under a strict seed-42 cross-rate plus unseen-profile benchmark, it obtains the lowest pooled MAE among seven electrical–optical backbones (0.864% SOC across 12 splits), ranking first for 2C→1C and third for the harder 1C→2C transfer. A five-seed parameter-matched input ablation further shows that dual-FBG information reduces 1C→2C MAE from 2.162% with electrical-only inputs to 1.795% with W1/W2, while native wavelengths and decoupled thermo-mechanical coordinates are statistically comparable in this difficult direction. Electrical-OOD stratification shows that optical benefit increases with distribution-shift severity and reaches 48.52% in the most shifted region. Wavelength perturbations up to 2 pm cause limited degradation, and a 95% residual conformal interval achieves 95.04% empirical coverage with a mean width of 2.075% SOC. The results support dual-FBG sensing as a compact complementary pathway for robust SOC estimation under demanding operating-condition changes in practical deployment.

## Keywords

State of charge; Fiber Bragg grating; Multimodal sensing; Temporal convolutional network; Cross-condition generalization; Conformal prediction

## Highlights

- A compact dual-FBG-assisted TCN is developed for electrical–optical SOC estimation.
- The proposed model gives the lowest pooled MAE across 12 strict cross-rate/unseen-profile splits.
- Five-seed ablation shows that FBG information improves the difficult 1C→2C transfer by about 17% in MAE.
- Optical benefit increases with electrical-OOD severity and reaches 48.52% in the most shifted region.
- A 95% residual conformal interval achieves 95.04% empirical coverage with a mean width of 2.075% SOC.

# 1. Introduction

Accurate state-of-charge (SOC) estimation is a core function of battery management systems (BMSs), underpinning energy management, power allocation, charge/discharge control, and safety protection in electric vehicles and energy-storage systems. Because SOC cannot be measured directly, it must be inferred from accessible signals such as terminal voltage, current, and temperature. Under dynamic operation, however, polarization, transient voltage response, nonlinear electrochemical behavior, and changing load profiles alter the mapping between these electrical measurements and SOC. An estimator that performs well for familiar operating conditions may therefore degrade when discharge rate, load intensity, or driving profile changes during deployment.

Existing SOC estimation approaches can broadly be divided into model-based observer/filtering methods [8] and data-driven methods that learn nonlinear mappings from measured signals to SOC. Deep temporal architectures, including convolutional neural networks (CNNs), recurrent neural networks, temporal convolutional networks (TCNs), and Transformers, have achieved high accuracy on increasingly complex battery datasets. Recent studies have also moved beyond single-condition accuracy toward wide-temperature modeling, cross-material transfer, and multi-condition generalization [1,2]. A recent critical review emphasized the need for more standardized evaluation protocols and highlighted transfer, few-shot, and continual-learning capability as important future directions for SOC estimation [1]. These developments indicate that a practical estimator should be assessed not only by interpolation accuracy, but also by its response to operating-condition changes.

Mechanical and thermo-mechanical battery responses provide an additional sensing pathway beyond conventional electrical measurements. Lithium insertion and extraction induce electrode expansion, structural deformation, thermal evolution, and stress variation, producing observables that are coupled to battery state. Recent work has shown that mechanical stress can complement voltage-based SOC estimation under dynamic load conditions [3]. Fiber Bragg grating (FBG) sensors are especially attractive for such applications because of their compact size, immunity to electromagnetic interference, embeddability, and high sensitivity to strain and temperature [12,13]. Through shifts in Bragg wavelength, embedded or surface-mounted FBGs can provide synchronized in-situ information about internal or local thermo-mechanical evolution.

FBG-assisted SOC estimation has developed rapidly in recent years. Chu et al. used implanted FBG sensors to obtain internal strain and temperature and combined these signals with electrical variables in a CNN–Transformer SOC estimator [4]. Ling et al. reported in-situ dual-FBG measurements from a silicon-based lithium-ion battery and integrated thermo-mechanical observations with electrical signals for high-precision SOC estimation [5]. At pack level, distributed optical strain sensing has also been combined with adaptive state estimation to address cell heterogeneity and monitoring complexity [6]. These studies demonstrate the feasibility of optical sensing for battery-state estimation. However, two engineering questions remain important when such sensing is used under changing operating conditions: how the two FBG channels should be represented for learning, and whether the additional optical pathway remains useful when both discharge rate and driving profile change.

The first question arises because a dual-FBG system can be processed in two different ways. The two directly measured wavelength channels W1/W2 can be used as model inputs, or they can first be transformed through a sensitivity matrix into thermo-mechanically interpretable temperature/force-related coordinates T/F. The transformation improves physical interpretability but does not introduce additional sensing degrees of freedom. In the present SiC-18 data, it also changes correlation, feature scale, and noise propagation. Rather than assuming one representation is universally superior, this study treats W1/W2 and T/F as alternative optical feature spaces and evaluates them under matched modeling conditions.

The second question concerns generalization. Many SOC studies combine multiple operating conditions during model development or use train/test partitions in which the target regime remains well represented. A more demanding case occurs when a model trained on one C-rate is deployed on an unseen driving profile at another C-rate. In this situation, the electrical excitation itself can move outside the support covered by the source data. An internal optical measurement may then provide complementary information, but this benefit should be verified using parameter-matched ablation rather than inferred from a multimodal model alone.

To address these issues, this work develops a compact representation-aware dual-FBG temporal convolutional network, termed RA-FBG-TCN. Voltage, current, and the selected dual-FBG representation are modeled through a lightweight causal TCN. The method is evaluated using both conventional blocked interpolation and a strict cross-rate plus unseen-profile protocol. A mainstream backbone benchmark is conducted using identical V/I/W1/W2 inputs, while a five-seed parameter-matched input ablation isolates the effects of electrical-only, thermo-mechanically decoupled, and native-wavelength inputs. A training-current-based electrical out-of-distribution (OOD) analysis is then used to explain the observed direction-dependent optical contribution. Finally, wavelength perturbation tests and residual split conformal prediction are used to assess sensing robustness and uncertainty reporting.

The main contributions of this work are summarized as follows.

1. **Dual-FBG-assisted lightweight SOC estimation framework.** A compact causal RA-FBG-TCN is constructed to fuse voltage, current, and dual-FBG observations for dynamic SOC estimation. The model contains approximately 11.5k trainable parameters and requires only present and historical measurements during inference.

2. **Matched optical-feature analysis and formal input ablation.** Native W1/W2 and decoupled T/F are analyzed as alternative representations of the same optical sensing information. The development-stage representation screen is followed by a five-seed parameter-matched formal ablation of VI, VI+TF, and VI+W under strict cross-rate/unseen-profile transfer.

3. **Comprehensive cross-condition model evaluation.** Seven electrical–optical sequence backbones are compared using the same V/I/W1/W2 inputs and source-only frozen training budgets. RA-FBG-TCN achieves the lowest pooled MAE across the 12 strict transfer splits, while five-seed evaluation quantifies profile- and initialization-dependent variability.

4. **Mechanism-oriented robustness and reliability analysis.** A label-free electrical-OOD stratification explains when the optical pathway contributes most strongly, direct pm-scale wavelength perturbations quantify sensing robustness, and residual split conformal prediction provides a lightweight 95% SOC uncertainty interval.

The remainder of this paper is organized as follows. Section 2 introduces the SiC-18 dual-FBG battery dataset, sensing principle, and electrical–optical feature characteristics. Section 3 presents the RA-FBG-TCN estimator and residual conformal uncertainty method. Section 4 reports conventional accuracy, strict backbone comparison, input ablation, OOD-based optical-contribution analysis, multi-seed cross-condition generalization, wavelength-noise robustness, and uncertainty results. Section 5 discusses the engineering implications, transfer asymmetry, representation choice, and scope limitations. Section 6 summarizes the main conclusions and outlines future work.

# 2. Dataset and electrical–optical signal analysis

## 2.1. Dataset and operating conditions

This study uses the publicly available SiC-18 dataset previously reported for an instrumented SiOx/C pouch-type lithium-ion cell [5]. The cell contains two embedded fiber Bragg grating (FBG) sensors. The nominal cell capacity is approximately 2.5 Ah. An armored FBG and a bare FBG were embedded inside the cell to provide two synchronized Bragg-wavelength observations associated with internal thermo-mechanical evolution during charge and discharge. Terminal voltage, current, and reference SOC were recorded simultaneously, enabling time-aligned electrical–optical state estimation.

The released data contain six representative dynamic driving profiles—HWFET, LA92, NEDC, NYCC, US06, and WLTC—tested at both 1C and 2C discharge rates. This produces 12 principal dynamic-discharge trajectories and approximately 68,086 valid samples. The profiles differ substantially in current amplitude, pulse density, and load-transition frequency, providing a useful test bed for evaluating SOC estimation under dynamic loading, C-rate changes, and unseen driving profiles. The dataset and primary experimental variables are summarized in Table 1.

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

The main predictor vector contains terminal voltage V, current I, and the two FBG wavelength channels W1 and W2. SOC is used only as the supervised target and is never provided to the estimator as an input. Variables that can directly encode discharge progress, such as cumulative discharged capacity or absolute trajectory time, are excluded from the predictor set. For every training/validation/calibration/test partition, normalization statistics are estimated exclusively from the corresponding training data.

Figure 1 illustrates representative synchronized electrical and optical responses. The driving current exhibits rapid and frequent transients, while the electrical response evolves jointly with SOC and the instantaneous load. In contrast, the two FBG wavelength channels show smoother but distinct dynamic trajectories. For the NEDC example, both W1 and W2 shift substantially during discharge, but their amplitudes and local temporal patterns are not identical. The optical trajectories also change between 1C and 2C for the same driving profile. These observations indicate that the electrical and optical measurements are synchronized but not dynamically redundant, motivating a closer examination of the complementary value of FBG sensing under operating-condition shift.

## 2.2. Dual-FBG sensing principle and thermo-mechanical decoupling

The center wavelength of an FBG is governed by

\[
\lambda_B = 2n_{\mathrm{eff}}\Lambda,
\]

where \(n_{\mathrm{eff}}\) denotes the effective refractive index and \(\Lambda\) is the grating period. Changes in temperature and strain modify both quantities and therefore shift the reflected Bragg wavelength. FBGs can consequently provide in-situ observations of local thermal and mechanical evolution through wavelength variation.

For the dual-FBG system considered here, the two sensing channels have different thermal and mechanical sensitivities. The coefficients below are adopted directly from the published SiC-18 sensing relation [5] and retain the variable definitions and units used in the released data. The two wavelength responses and the decoupled temperature T and deformation-force-related quantity F approximately satisfy

\[
W_1 = 0.0208T + 0.00054F,
\]

\[
W_2 = 0.0254T + 0.00085F.
\]

The relation can be written compactly as

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

When the sensitivity matrix \(\mathbf{K}\) is invertible, the thermo-mechanical coordinates can be reconstructed from the two measured wavelength channels. The corresponding approximate inverse transformation is

\[
T = 214.43W_1 - 136.23W_2,
\]

\[
F = -6407.67W_1 + 5247.23W_2.
\]

This relationship is important for the modeling strategy adopted in this work. W1/W2 and T/F are not four independent sensing channels; rather, they are two coordinate systems describing the same two optical sensing degrees of freedom. T/F offers clearer physical interpretation for analyzing internal thermal and mechanical evolution, whereas W1/W2 preserves the native coordinates directly produced by the optical interrogation system. Whether the physically decoupled representation is also the most transferable representation for data-driven SOC estimation is therefore an empirical question rather than an assumption.

## 2.3. Electrical–optical characteristics and representation analysis

### 2.3.1. Dynamic-response differences under varying load

Under a dynamic driving cycle, terminal voltage is jointly affected by SOC, ohmic drop, polarization, and instantaneous current. As a result, the electrical response can fluctuate rapidly even when SOC evolves smoothly. The implanted FBGs respond to internal thermo-mechanical evolution, whose characteristic time scales need not coincide with those of the terminal electrical variables. The synchronized sequences in Fig. 1 illustrate this difference: current contains high-frequency pulses, whereas the wavelength channels evolve more smoothly while retaining local responses to the changing load.

The contrast becomes more pronounced when the discharge rate changes. Moving from 1C to 2C directly expands the current range and alters the terminal electrical response, whereas the FBG wavelengths continue to reflect the accumulated and coupled internal thermo-mechanical state. The optical measurements are therefore treated here as potentially complementary internal observations rather than as duplicate measurements of V/I. Section 4 evaluates this hypothesis using parameter-matched models and an electrical-OOD analysis.

### 2.3.2. Statistical structure of native and decoupled optical coordinates

Although W1/W2 and T/F contain the same number of optical degrees of freedom, their statistical structures differ substantially. Figure 2(a,b) shows the joint distributions of the two representations over all 12 dynamic trajectories. When the data are grouped by C-rate, the absolute Pearson correlation between W1 and W2 is 0.738 at 1C and 0.655 at 2C. After thermo-mechanical decoupling, the corresponding absolute T/F correlations increase to 0.982 and 0.985, respectively, as summarized in Fig. 2(c). Thus, the two decoupled variables are much more linearly coupled in this dataset.

This observation does not diminish the physical value of thermo-mechanical decoupling. Mapping the two measured wavelengths to temperature- and force-related coordinates provides clearer physical semantics and is useful for sensing-mechanism interpretation. However, the linear inversion also changes feature scale, correlation structure, and the way measurement perturbations propagate through the representation. For an end-to-end learning problem focused on cross-rate and unseen-profile transfer, a physically interpretable coordinate system should therefore not be assumed a priori to provide the most favorable predictive representation.

### 2.3.3. Representation-aware predictive comparison

To select the optical input representation, W1/W2 and T/F were compared under the same causal TCN architecture, training budget, and 1C→2C unseen-profile development protocol. Figure 2(d) and Table 4 report the main results. Across the six held-out-profile splits, the compact TCN using native W1/W2 achieved an average MAE of 1.385% SOC, RMSE of 2.084% SOC, and Q95 absolute error of 4.873% SOC. Replacing W1/W2 with the decoupled T/F coordinates increased the corresponding errors to 2.033%, 3.060%, and 6.835% SOC. A more complex electrical–thermo-mechanical fusion model (ETMF-TF) reduced the MAE to 1.569% SOC but still did not outperform the compact model operating directly in wavelength space.

These results show that adding a physical interpretation layer does not necessarily improve predictive transfer under condition shift. Native W1/W2 preserves the complete dual-FBG measurement while avoiding an additional coordinate inversion and provides more stable average generalization in the matched experiments. The final estimator therefore uses V, I, W1, and W2 as inputs, whereas T/F is retained for sensing interpretation and representation ablation.

Overall, the role of dual-FBG sensing in this work is not to stack wavelength, temperature, and force as redundant features. Instead, the optical system provides an internal observation pathway with a different physical origin from the terminal electrical measurements. Based on the preceding signal and representation analysis, the next section develops a lightweight causal estimator operating directly on the native dual-FBG wavelength coordinates.

# 3. Methodology

## 3.1. Overview of the RA-FBG-TCN framework

A representation-aware dual-FBG temporal convolutional network, denoted RA-FBG-TCN, is developed for SOC estimation under dynamic loads and changing operating conditions. The temporal backbone follows the causal/dilated convolutional sequence-modeling principle established for TCNs [9]. The framework does not treat the thermo-mechanically decoupled variables as additional independent measurements. Instead, it first determines which optical coordinate system is more suitable for cross-condition prediction and then directly combines voltage, current, and the selected native FBG wavelengths in a compact causal temporal model. Residual split conformal prediction is applied after point-model training to provide calibrated uncertainty intervals.

At time t, the input vector is

\[
\mathbf{x}_t=[V_t, I_t, W_{1,t}, W_{2,t}],
\]

where \(V_t\) and \(I_t\) are terminal voltage and current, and \(W_{1,t}\) and \(W_{2,t}\) are the two FBG wavelength observations. A causal history window of length \(L=64\) is formed as

\[
\mathbf{X}_t=[\mathbf{x}_{t-L+1},\ldots,\mathbf{x}_t],
\]

and the model outputs the SOC estimate \(\hat{y}_t\) at the final time step. Because the window contains only present and historical observations, no future information is used during inference.

As illustrated in Fig. 3, the complete pipeline consists of four stages: (1) training-only channel normalization, (2) 64-sample causal window construction, (3) dilated residual TCN encoding and SOC regression, and (4) post-hoc residual conformal calibration. In compact form,

\[
V/I/W1/W2 \rightarrow \text{train-only normalization}
\rightarrow \text{causal window}
\rightarrow \text{dilated TCN}
\rightarrow \widehat{SOC}
\rightarrow 95\%\ \text{conformal interval}.
\]

## 3.2. Representation-aware optical input selection

Let the native wavelength-shift vector and the decoupled thermo-mechanical vector be denoted by

\[
\Delta\boldsymbol{\lambda}=[\Delta\lambda_1,\Delta\lambda_2]^T,
\qquad
\mathbf{z}=[T,F]^T.
\]

Their relationship is

\[
\Delta\boldsymbol{\lambda}=\mathbf{K}\mathbf{z},
\]

and, when \(\mathbf{K}\) is invertible,

\[
\mathbf{z}=\mathbf{K}^{-1}\Delta\boldsymbol{\lambda}.
\]

The two representations therefore encode the same two measured optical degrees of freedom. Explicit decoupling improves physical semantics but does not introduce new sensing information. It also modifies the correlation and scale of the features and can alter the propagation of wavelength perturbations in feature space.

For this reason, representation choice is treated as part of model design. Native W1/W2 and decoupled T/F are evaluated using the same data partitions, model capacity, and training budget. The native wavelength representation gives lower average MAE, RMSE, and Q95 error in the cross-rate unseen-profile development comparison described in Section 2.3.3. RA-FBG-TCN therefore fixes W1/W2 as the optical inputs. This decision concerns predictive transferability and does not imply that T/F decoupling lacks value for physical interpretation.

## 3.3. Causal TCN SOC estimator

### 3.3.1. Training-only normalization

Because voltage, current, and FBG wavelength observations have different scales, each input channel is standardized using statistics computed only from the training data. For channel j,

\[
x'_{t,j}=\frac{x_{t,j}-\mu_j}{\sigma_j},
\]

where \(\mu_j\) and \(\sigma_j\) are the training-set mean and standard deviation. The same fixed statistics are then applied to the validation, calibration, and test sets. No target-domain statistics are re-estimated.

### 3.3.2. Dilated causal convolution

To ensure online causality, all temporal convolutions use left-only padding. For layer l with kernel size k and dilation \(d_l\), the causal convolution can be written as

\[
\mathbf{h}^{(l)}_t=
\sum_{i=0}^{k-1}\mathbf{W}^{(l)}_i
\mathbf{h}^{(l-1)}_{t-id_l}+\mathbf{b}^{(l)}.
\]

Thus, the representation at time t cannot access any observation after t.

The four-channel input is first projected to a 24-dimensional hidden space by a 1×1 convolution. Three residual temporal blocks are then stacked with dilation factors 1, 2, and 4. Each block contains two causal Conv1D layers with kernel size 3, GroupNorm, GELU activation, and a residual connection. For a block input \(\mathbf{H}\),

\[
\mathbf{Z}_1=\mathrm{GELU}\left(\mathrm{GN}(\mathrm{Conv}_{c}(\mathbf{H};k=3,d))\right),
\]

\[
\mathbf{Z}_2=\mathrm{GN}(\mathrm{Conv}_{c}(\mathbf{Z}_1;k=3,d)),
\]

\[
\mathbf{H}_{out}=\mathrm{GELU}(\mathbf{H}+\mathbf{Z}_2).
\]

The dilation sequence provides multi-scale local temporal context while retaining a small parameter count. The three residual blocks yield an effective receptive field of 29 samples, while the 64-sample input window supplies a longer causal context from which the final hidden state is extracted.

### 3.3.3. Temporal representation and SOC regression

After the residual blocks, the encoder produces a hidden sequence \(\mathbf{H}\in\mathbb{R}^{L\times C}\). Because the target is the SOC at the end of the current window, only the final causal hidden state \(\mathbf{h}_t\) is passed to a two-layer nonlinear regression head:

\[
\mathbf{u}_t=\mathrm{GELU}(\mathbf{W}_1\mathbf{h}_t+\mathbf{b}_1),
\]

\[
\hat{y}_t=\mathbf{W}_2\mathbf{u}_t+\mathbf{b}_2.
\]

The final RA-FBG-TCN contains approximately 11.5k trainable parameters. The model is optimized using mean-squared error,

\[
\mathcal{L}_{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2,
\]

with AdamW. The implementation uses a learning rate of \(10^{-3}\), weight decay of \(10^{-4}\), gradient clipping at 2.0, and a batch size of 256. Training is limited to 50 epochs, with a minimum of 10 epochs and patience of 7 epochs. The model checkpoint with the lowest validation MAE is retained, and the validation set is never used for final test metrics. The main architecture and training settings are summarized in Table 2.

**Table 2. Main RA-FBG-TCN architecture and training settings.**

| Item | Setting |
|---|---|
| Input variables | V, I, W1, W2 |
| Causal window length | 64 samples |
| Input projection | 4 → 24, 1×1 Conv1D |
| Residual TCN blocks | 3 |
| Dilations | 1, 2, 4 |
| Convolutions per block | 2 causal Conv1D layers |
| Kernel size | 3 |
| Normalization / activation | GroupNorm / GELU |
| Regression head | 24 → 24 → 1 |
| Trainable parameters | Approximately 11.5k |
| Loss | Mean-squared error |
| Optimizer | AdamW |
| Learning rate | 1×10⁻³ |
| Weight decay | 1×10⁻⁴ |
| Gradient clipping | 2.0 |
| Batch size | 256 |
| Maximum epochs | 50 |
| Early stopping | Validation MAE, minimum 10 epochs, patience 7 |

## 3.4. Residual split-conformal uncertainty quantification

A point estimate alone does not communicate prediction reliability. To obtain an uncertainty interval without introducing an additional probabilistic neural network, residual split conformal prediction [10,11] is applied after the point estimator has been trained and frozen.

Let the independent calibration set be

\[
\mathcal{D}_{cal}=\{(\mathbf{X}_i,y_i)\}_{i=1}^{n}.
\]

The frozen RA-FBG-TCN produces calibration predictions \(\hat{y}_i\), and the nonconformity score is defined as the absolute residual

\[
r_i=|y_i-\hat{y}_i|.
\]

For a target miscoverage level \(\alpha=0.05\), the finite-sample corrected conformal rank is

\[
k=\left\lceil(n+1)(1-\alpha)\right\rceil,
\]

and \(q_{1-\alpha}\) is the k-th order statistic of the calibration residuals. For a new input \(\mathbf{X}_t\), the 95% prediction interval is

\[
\mathcal{C}_{0.95}(\mathbf{X}_t)=
[\hat{y}_t-q_{0.95},\hat{y}_t+q_{0.95}].
\]

The lower and upper bounds are clipped to the physical SOC range [0,1]. The calibration procedure uses only calibration residuals and does not modify the frozen point estimator or adapt interval width using test labels.

Uncertainty performance is quantified by prediction interval coverage probability (PICP), mean prediction interval width (MPIW), and mean interval score (MIS). The first two are

\[
\mathrm{PICP}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\{y_i\in[L_i,U_i]\},
\]

\[
\mathrm{MPIW}=\frac{1}{N}\sum_{i=1}^{N}(U_i-L_i).
\]

For miscoverage level \(\alpha\), the interval score is

\[
IS_i=(U_i-L_i)
+\frac{2}{\alpha}(L_i-y_i)\mathbf{1}(y_i<L_i)
+\frac{2}{\alpha}(y_i-U_i)\mathbf{1}(y_i>U_i),
\]

and MIS is the average of \(IS_i\). A useful interval should achieve coverage close to the nominal level while remaining as narrow as possible. Decoupling the lightweight causal point estimator from post-hoc conformal calibration allows the framework to provide both an SOC estimate and an interpretable uncertainty interval with negligible additional online model complexity.

# 4. Experiments and results

## 4.1. Experimental settings and evaluation metrics

The proposed framework is evaluated from five complementary perspectives: conventional mixed-condition accuracy, electrical–optical complementarity under distribution shift, optical-representation choice, strict cross-rate unseen-profile generalization, and reliability under wavelength perturbation and conformal uncertainty calibration. Unless otherwise stated, all experiments use a 64-sample causal window and training-only normalization. The RA-FBG-TCN architecture and optimization settings are listed in Table 2. Test data are never used to estimate normalization statistics, select epochs, or tune the uncertainty interval.

Point-estimation performance is evaluated using mean absolute error (MAE), root mean square error (RMSE), coefficient of determination (R²), 95th-percentile absolute error (Q95-AE), and maximum absolute error (MaxAE). MAE and RMSE quantify average prediction accuracy, whereas Q95-AE and MaxAE characterize the upper error tail and extreme deviations. Uncertainty estimates are evaluated using prediction interval coverage probability (PICP), mean prediction interval width (MPIW), and mean interval score (MIS).

Two evaluation regimes are emphasized. The first is a blocked mixed-condition interpolation setting, which evaluates conventional SOC estimation when the operating distribution is well represented during training. The second is a stricter cross-rate unseen-profile protocol. For each held-out case, the model is trained on five complete driving profiles from one C-rate and tested on the sixth profile at the opposite C-rate, while the same named profile is completely excluded from training. The second setting therefore introduces both C-rate shift and driving-profile shift.

## 4.2. Conventional SOC estimation performance

Table 3 compares representative sequence models under blocked mixed-condition interpolation. CNN, GRU, LSTM, Transformer, and the different TCN input variants use the same data partitions, training-derived normalization, and early-stopping rule.

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

All models attain high accuracy in this well-supported interpolation setting. The electrical-only VI-TCN gives the lowest point-estimation error, indicating that voltage and current already contain highly informative SOC cues when the test distribution is well represented by the training data. RA-FBG-TCN nevertheless reaches an MAE of 0.482% SOC, an RMSE of 0.593% SOC, and an R² of 0.999614 with only approximately 11.5k trainable parameters. Representative test trajectories and the corresponding error distribution are shown in Fig. 4.

This result is important for interpreting the role of optical sensing. The present study does not assume that FBG inputs must improve every in-distribution sample. Instead, the central question is whether internal optical observations become more useful when the conventional electrical measurements leave the region represented during training.

## 4.3. Electrical–optical complementarity under operating-condition shift

To isolate the contribution of the optical channels from model-capacity effects, two parameter-matched causal TCNs are compared. The VI model receives voltage and current, with the remaining two input positions fixed to zero, whereas VI+W receives voltage, current, W1, and W2. Both models therefore have the same architecture and parameter count.

For 1C→2C transfer, the average MAE of the VI model is 2.151% SOC. Adding the two native FBG wavelength channels reduces the MAE to 1.632% SOC, corresponding to a relative reduction of approximately 24.1%. In the reverse 2C→1C direction, the MAE decreases from 0.866% to 0.814% SOC, a relative reduction of approximately 6.0%. The much larger gain in the low-to-high-rate direction suggests that optical observations become particularly useful when the target current extends beyond the excitation represented in the source-rate data.

To examine this hypothesis directly, an electrical support envelope is defined from the 0.5th–99.5th percentile range of the source-training current. For each test window, the proportion of samples outside this envelope is defined as the electrical-OOD fraction. The score uses only source-current statistics and observed current; target SOC labels are not involved. Figure 5(a) illustrates a representative 2C trajectory relative to the 1C training support, while Fig. 5(b,c) reports error and optical benefit across OOD-severity bins.

**Table 4. Parameter-matched electrical–optical comparison under increasing electrical-OOD severity for 1C→2C transfer.**

| Electrical-OOD level | VI MAE (% SOC) | VI+W MAE (% SOC) | Relative optical gain |
|---|---:|---:|---:|
| ID | **0.735** | 0.873 | −18.75% |
| OOD 0–25% | 1.550 | **1.470** | +5.17% |
| OOD 25–50% | 2.772 | **2.356** | +15.00% |
| OOD 50–75% | 3.665 | **2.930** | +20.05% |
| OOD 75–100% | 5.529 | **2.846** | **+48.52%** |

Within the fully supported electrical ID region, adding FBG observations does not improve the electrical-only estimator. Once the trajectory begins to leave that support, however, the optical benefit becomes positive and increases monotonically with OOD severity. In the most shifted windows, the relative MAE reduction reaches 48.52%. Thus, the value of the auxiliary optical modality is strongly condition dependent: it is not required to duplicate already sufficient electrical information, but becomes increasingly useful when terminal electrical excitation is unfamiliar to the source-trained estimator.

A related question is whether the two FBG channels should be used in their native wavelength coordinates or after thermo-mechanical decoupling. As described in Section 2.3.3 and Fig. 2(d), the native W1/W2 TCN obtains an average MAE of 1.385% SOC across six 1C→2C unseen-profile development splits, compared with 2.033% SOC for the same TCN using T/F. The more complex ETMF-TF model reaches 1.569% SOC but still does not outperform the compact native-wavelength estimator. Together with the markedly stronger linear coupling of T/F shown in Fig. 2(c), these results support the use of W1/W2 as the retained predictive representation.

## 4.4. Cross-rate unseen-profile generalization

After the representation and estimator are fixed, RA-FBG-TCN is evaluated under the strict cross-rate unseen-profile protocol. For each held-out profile, neither the target rate nor the same named driving profile appears in the training data. Six held-out profiles are evaluated in each transfer direction, and every split is repeated using five independent random seeds.

The aggregate results are shown in Table 5 and Fig. 6. For 1C→2C transfer, the model achieves an average MAE of 1.795% SOC, RMSE of 3.037% SOC, R² of 0.985689, and Q95-AE of 6.804% SOC. In the reverse 2C→1C direction, the corresponding values are 0.806%, 0.988%, 0.998480, and 1.830% SOC. Across both directions, the seed-cluster mean MAE is 1.301% SOC. A seed-cluster bootstrap gives a 95% confidence interval of 0.961–1.677% SOC for the overall MAE.

**Table 5. Five-seed cross-rate unseen-profile performance.**

| Direction | MAE (% SOC) | RMSE (% SOC) | R² | Q95-AE (% SOC) |
|---|---:|---:|---:|---:|
| 1C→2C | 1.795 | 3.037 | 0.985689 | 6.804 |
| 2C→1C | **0.806** | **0.988** | **0.998480** | **1.830** |
| Overall | **1.301** | — | — | — |

The two transfer directions are clearly asymmetric. The 2C source data cover a wider range of high-current excitation, so most 1C target states remain within or near the source support. In contrast, 1C→2C requires extrapolation toward stronger and faster current excitation that is insufficiently represented during training. This support-based interpretation is consistent with the electrical-OOD analysis and also explains the larger error and stronger seed-to-seed variation in the low-to-high-rate direction.

## 4.5. Wavelength-noise robustness and conformal uncertainty

### 4.5.1. Robustness to wavelength perturbation

Practical FBG interrogation is subject to finite measurement noise. To evaluate sensitivity to direct optical perturbation, independent zero-mean Gaussian noise with standard deviations of 0.5, 1, and 2 pm is added to W1 and W2 before applying the frozen training-derived normalization. The trained model and all source-domain preprocessing parameters remain unchanged.

As shown in Fig. 7(a,b), error increases smoothly with wavelength-noise level rather than exhibiting abrupt degradation. At 2 pm per channel, the MAE increases by only approximately 1.67% relative to the clean baseline for 1C→2C transfer and 4.57% for 2C→1C transfer. Within the tested perturbation range, directly using native wavelength coordinates therefore does not introduce pronounced sensitivity to small optical measurement errors.

### 4.5.2. 95% residual conformal prediction interval

The frozen RA-FBG-TCN point estimator is further calibrated using residual split conformal prediction. Absolute residuals from the independent calibration split define the 95% conformal quantile, after which the same fixed interval rule is applied to the test data.

The resulting 95% nominal interval attains an empirical PICP of 95.04%, closely matching the target coverage. The mean prediction interval width is 2.075% SOC, the mean interval score is 0.02441, and the corresponding absolute-residual quantile is approximately 1.089% SOC. Figure 7(c) illustrates the point estimate and calibrated interval over a representative test segment. These empirical coverage results pertain to the blocked mixed-condition calibration/test regime; no formal 95% coverage guarantee is claimed here for arbitrary cross-rate or unseen-profile distribution shift. The result nevertheless shows that a simple post-hoc conformal layer can supplement the deterministic point estimator with an uncertainty interval whose empirical coverage is aligned with the nominal level, without requiring a second probabilistic neural network.

Taken together, the experiments support a transfer-oriented interpretation of the proposed framework. RA-FBG-TCN retains sub-percent conventional accuracy, provides substantial optical benefit in strongly shifted electrical regions, maintains useful performance under simultaneous C-rate and unseen-profile changes, degrades smoothly under pm-scale wavelength perturbations, and supports calibrated uncertainty reporting.

# 5. Discussion

## 5.1. Physical decoupling versus predictive representation

The dual-FBG system permits the two measured Bragg wavelengths to be transformed into temperature- and force-related variables using sensor sensitivities. From a sensing perspective, this decoupling is valuable because it assigns clearer physical meaning to the optical response. The present results nevertheless demonstrate that stronger physical interpretability does not automatically imply a better representation for cross-condition data-driven prediction.

W1/W2 and T/F describe the same two optical sensing degrees of freedom and are related through an invertible linear transformation. The transformation changes the coordinate system rather than adding measurements. In the present data, it also produces substantially stronger linear coupling between the two variables and changes feature scaling and perturbation propagation. These properties can affect how a temporal model extracts patterns that remain stable outside the source operating condition.

The matched representation experiments illustrate this distinction empirically. Native W1/W2 yields lower average cross-condition error than both the same TCN using T/F and a more complex electrical–thermo-mechanical fusion model. This should not be interpreted as evidence against physical decoupling. Rather, it highlights that sensing interpretation and predictive representation serve different objectives: one seeks physically meaningful variables, whereas the other seeks coordinates that preserve task-relevant information in a form that transfers reliably. For the present SOC task, retaining the native optical coordinates is the more effective engineering choice.

## 5.2. Why optical value emerges under distribution shift

The blocked-interpolation experiment shows that an electrical-only TCN can outperform the electrical–optical estimator when the test distribution is already well supported. Terminal voltage and current contain strong SOC information in this regime, so additional FBG observations can be partly redundant. This does not contradict the proposed sensing strategy; it clarifies where the auxiliary modality is actually useful. High-accuracy SOC estimation has already been demonstrated on the same sensing platform, including a reported RMSE of 0.635% SOC [5]. Conventional interpolation accuracy is therefore not treated as the primary novelty here, and direct leaderboard comparison is avoided because the evaluation protocols are not identical. The present contribution instead centers on optical representation choice, compound operating-condition shift, electrical-OOD-dependent complementarity, and calibrated uncertainty.

During 1C→2C transfer, the target current increasingly enters regions that were absent or rare during source-rate training. A V/I-only estimator must then extrapolate from unfamiliar electrical excitation. The FBG wavelengths are influenced not only by instantaneous load but also by internal thermo-mechanical evolution coupled to battery state. Their response is therefore not identical to the terminal electrical variables, allowing them to provide additional state constraints when electrical support weakens.

The OOD-stratified results make this mechanism visible at the data level. Optical input is not beneficial in the fully supported ID region, but its relative contribution becomes positive once test windows contain out-of-support electrical states and rises to approximately 48.5% in the most severe OOD region. This pattern suggests a broader interpretation for multimodal battery sensing: an auxiliary modality need not outperform the primary modality everywhere to be valuable. Its practical role can instead be to supply complementary information when the primary sensing pathway becomes poorly supported by the training distribution.

For BMS deployment, this distinction is important because real operating conditions cannot be exhaustively represented during offline training. Rate changes, aggressive acceleration, and other high-power transients may move the electrical measurements outside historical support. Internal optical sensing can therefore be viewed as an additional state constraint for difficult operating conditions rather than as a replacement for voltage and current sensing.

## 5.3. Transfer asymmetry, sensing robustness, and uncertainty

The strict generalization experiment reveals a pronounced directional asymmetry: 2C→1C transfer is considerably easier than 1C→2C transfer. This difference is consistent with the support of the source current distribution. Training at 2C exposes the estimator to a broader high-current range, so lower-rate target states remain largely covered. Training at 1C does not provide equivalent coverage of the highest-current 2C states and therefore requires a more difficult extrapolation.

The wavelength-noise and conformal experiments provide complementary reliability evidence. Native W1/W2 produces smooth rather than abrupt degradation under direct pm-scale perturbation, while the residual conformal interval reaches empirical coverage essentially equal to its 95% nominal target. The point estimator and uncertainty layer are deliberately decoupled: the former remains compact and causal, while the latter uses calibration residuals to add uncertainty information without materially increasing online inference complexity.

Together, these results suggest that robust multimodal SOC estimation should be assessed through more than a single average error measured near the training distribution. Transfer direction, sensing support, perturbation sensitivity, and predictive uncertainty all influence whether a model is useful in realistic operating conditions.

## 5.4. Scope and future work

The primary quantitative dataset in this study contains one physical cell instrumented with a fixed dual-FBG sensing configuration. The conclusions apply most directly to operating-condition transfer in which sensor installation and calibration remain consistent while discharge rate and driving profile change. Across different physical cells, FBG initial wavelength, bonding condition, strain-transfer efficiency, and sensor-specific sensitivity may vary and can alter both the absolute level and dynamics of the optical signal.

Future work should therefore investigate sensor-aware calibration and adaptation across multiple cells, broader temperature ranges, long-term sensor aging, and changes in bonding condition. These sensing-calibration and transfer issues may ultimately be more important for practical deployment than further increasing neural-network complexity. Online calibration and lightweight adaptive mechanisms are particularly relevant for translating fiber-optic-assisted SOC estimation from controlled datasets to operational BMS applications.

Overall, the present results indicate that FBG sensing should not be judged solely by whether it reduces interpolation error on every sample. Under meaningful operating-condition shift, optical observations can provide a distinct internal-state constraint when conventional electrical measurements leave their training support. Combining this sensing complementarity with a compact causal estimator and calibrated uncertainty offers a practical route toward more robust multimodal battery-state estimation.

# 6. Conclusion

This study investigated representation-aware dual-FBG optical sensing for battery SOC estimation under operating-condition shift. Native wavelength coordinates and thermo-mechanically decoupled temperature/force coordinates were analyzed as alternative representations of the same dual-FBG sensing information. Under matched cross-condition evaluation, W1/W2 exhibited more stable predictive transfer, motivating a compact RA-FBG-TCN that combines voltage, current, and the two native wavelength channels in a 64-sample causal input sequence.

Under blocked interpolation, RA-FBG-TCN achieved an MAE of 0.482% SOC, an RMSE of 0.593% SOC, and an R² of 0.999614. More importantly, parameter-matched OOD analysis showed that incorporating W1/W2 reduced the overall 1C→2C MAE from 2.151% to 1.632% SOC, while the relative optical benefit increased with electrical-OOD severity and reached approximately 48.5% in the most shifted region. The strict cross-rate unseen-profile evaluation yielded average MAEs of 1.795% SOC for 1C→2C and 0.806% SOC for 2C→1C, with an overall seed-cluster MAE of 1.301% SOC and a bootstrap 95% confidence interval of 0.961–1.677% SOC.

Direct wavelength perturbations up to 2 pm caused only limited error growth. The 95% residual conformal interval attained 95.04% empirical coverage with a mean width of 2.075% SOC, providing calibrated uncertainty without modifying the point estimator. The combined evidence indicates that the main value of internal optical sensing is not simply the addition of more predictor variables, but the provision of state information from a different physical pathway when terminal electrical observations become weakly supported by the source distribution. Future work will extend this framework toward multi-cell sensor calibration, wider temperature conditions, sensor aging, and online adaptation.

# Data availability

The SiC-18 dataset analyzed in this study is publicly available through Mendeley Data (DOI: 10.17632/ft6rtwt8vm.1), as reported with the source study [5].

# Code availability

The analysis code, frozen experimental workflows, source-data tables, and reproducible figure-generation pipeline used in this study are maintained at https://github.com/Ustinian1722/FiberopticSOC. The submission branch preserves the numerical provenance of the results reported in the manuscript.

# Figure captions

## Fig. 1

**Fig. 1. Synchronized electrical and optical observations during dynamic discharge of the SiC-based lithium-ion cell.** (a) Placeholder for the battery test platform and implanted dual-FBG sensing configuration; the final artwork will be assembled separately. (b) Current and reference SOC during a representative NEDC 1C trajectory. (c) Synchronously recorded W1 and W2 Bragg-wavelength shifts. (d) W2–SOC response under NEDC at 1C and 2C. The quantitative panels are generated directly from the released SiC-18 trajectories without random subsampling. The figure illustrates that terminal electrical measurements and the two internal optical channels are synchronized but exhibit distinct dynamic evolution.

## Fig. 2

**Fig. 2. Representation characteristics and transfer performance of native and thermo-mechanically decoupled dual-FBG coordinates.** (a) Joint distribution of W1 and W2 over all 12 dynamic trajectories. (b) Corresponding joint distribution of the decoupled temperature and force coordinates. (c) Absolute Pearson correlation for the two coordinate systems at 1C and 2C. Native W1/W2 gives |r|=0.738 and 0.655, whereas the decoupled T/F coordinates give |r|=0.982 and 0.985. (d) Average SOC MAE under the matched cross-rate development protocol for raw W, decoupled T/F, and the more complex ETMF-TF representation. The results show that improved physical semantics do not automatically translate into more stable predictive transfer; the final estimator therefore retains W1/W2.

## Fig. 3

**Fig. 3. Overall RA-FBG-TCN and residual split-conformal uncertainty framework.** Placeholder for the final methodology artwork. The final figure will show V/I/W1/W2 inputs, training-only normalization, a 64-sample causal window, 4→24 input projection, residual TCN blocks with dilation factors 1/2/4, the SOC regression head, and a 95% conformal prediction interval calibrated from an independent residual set. An inset will show two k=3 causal Conv1D layers, GroupNorm, GELU activation, and the residual connection within each TCN block.

## Fig. 4

**Fig. 4. Conventional SOC estimation performance of RA-FBG-TCN under the blocked mixed-condition interpolation protocol.** Representative test segments compare reference and estimated SOC, followed by the corresponding absolute-error trajectory and the test-set absolute-error distribution. The retained estimator achieves an overall MAE of 0.482% SOC, RMSE of 0.593% SOC, and R² of 0.999614. This experiment establishes the baseline point-estimation capability of the compact causal estimator and is not interpreted as evidence that optical inputs universally outperform electrical-only inputs under in-distribution conditions.

## Fig. 5

**Fig. 5. Relationship between electrical distribution-shift severity and the benefit of dual-FBG optical observations.** (a) Electrical support envelope defined from the 0.5th–99.5th percentiles of source-rate training current, with a representative 1C→2C test trajectory showing out-of-support regions. (b) MAE of parameter-matched VI and VI+W models across bins of window-level electrical-OOD fraction. (c) Relative optical gain produced by adding W1/W2. The relative gain is −18.75% in the fully supported ID region and increases to +5.17%, +15.00%, +20.05%, and +48.52% as OOD severity increases. The result indicates that the value of FBG sensing emerges primarily when conventional electrical observations leave their source support.

## Fig. 6

**Fig. 6. Five-seed generalization of RA-FBG-TCN under the cross-rate unseen-profile protocol.** (a) MAE for the six held-out driving profiles in the 1C→2C direction; bars and error bars denote the mean and standard deviation across five random seeds. (b) Corresponding results for 2C→1C. (c) Seed-cluster bootstrap 95% confidence intervals for the two transfer directions and their overall MAE. Aggregate MAEs are 1.795% SOC for 1C→2C and 0.806% SOC for 2C→1C; the overall MAE is 1.301% SOC with a bootstrap 95% confidence interval of 0.961–1.677% SOC. The low-to-high-rate direction is more difficult and more variable than the reverse transfer.

## Fig. 7

**Fig. 7. Robustness to dual-FBG wavelength perturbation and calibrated SOC uncertainty.** (a) MAE after independently adding Gaussian wavelength noise with standard deviations of 0, 0.5, 1, and 2 pm to W1 and W2. (b) Corresponding Q95 absolute error. (c) Representative blocked-test trajectory showing reference SOC, RA-FBG-TCN point prediction, and the 95% residual split-conformal prediction interval. At 2 pm noise, degradation remains smooth and limited. The 95% nominal interval achieves a PICP of 95.04% with an MPIW of 2.075% SOC.

# References

[1] J. Yao, J. Kowal, Towards a smarter battery management system: A critical review on deep learning-based state of charge estimation of lithium-ion batteries, *Energy and AI* 21 (2025) 100585. https://doi.org/10.1016/j.egyai.2025.100585.

[2] X. Wu et al., Data-driven SOC estimation method for power batteries under driving cycle conditions and a wide temperature range, *Energy* 340 (2025) 139147. https://doi.org/10.1016/j.energy.2025.139147.

[3] Y. Fan et al., Mechanical stress-based state-of-charge estimation for lithium-ion batteries via deep learning techniques, *Energy* 326 (2025) 136216. https://doi.org/10.1016/j.energy.2025.136216.

[4] Y. Chu et al., Estimation of state-of-charge for lithium-ion batteries based on simultaneous internal strain and temperature monitoring by fiber optic sensors, *Journal of Energy Storage* 133 (2025) 117969. https://doi.org/10.1016/j.est.2025.117969.

[5] C. Ling et al., In-situ data-driven high-precision SOC estimation for silicon-based lithium-ion batteries, *Energy* 349 (2026) 140609. https://doi.org/10.1016/j.energy.2026.140609.

[6] S. Liu, K. Li, J. Yu, Adaptive estimation of battery pack state of charge with optical fibre strain measurements, *Applied Energy* 407 (2026) 127330. https://doi.org/10.1016/j.apenergy.2025.127330.

[7] K. L. Soon, L. T. Soon, Enhancing reliability in electrified transportation: A conformalized quantile regression framework for battery state-of-charge uncertainty quantification, *Journal of Power Sources* 666 (2026) 239123. https://doi.org/10.1016/j.jpowsour.2025.239123.

[8] G. L. Plett, Extended Kalman filtering for battery management systems of LiPB-based HEV battery packs: Part 3. State and parameter estimation, *Journal of Power Sources* 134(2) (2004) 277–292. https://doi.org/10.1016/j.jpowsour.2004.02.033.

[9] S. Bai, J. Z. Kolter, V. Koltun, An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, arXiv:1803.01271 (2018). https://doi.org/10.48550/arXiv.1803.01271.

[10] J. Lei, M. G’Sell, A. Rinaldo, R. J. Tibshirani, L. Wasserman, Distribution-Free Predictive Inference for Regression, *Journal of the American Statistical Association* 113(523) (2018) 1094–1111. https://doi.org/10.1080/01621459.2017.1307116.

[11] G. Shafer, V. Vovk, A Tutorial on Conformal Prediction, *Journal of Machine Learning Research* 9 (2008) 371–421.

[12] C.-J. Bae, A. Manandhar, P. Kiesel, A. Raghavan, Monitoring the Strain Evolution of Lithium-Ion Battery Electrodes using an Optical Fiber Bragg Grating Sensor, *Energy Technology* 4(7) (2016) 851–855. https://doi.org/10.1002/ente.201500514.

[13] A. Fortier, M. Tsao, N. D. Williard, Y. Xing, M. G. Pecht, Preliminary Study on Integration of Fiber Optic Bragg Grating Sensors in Li-Ion Batteries and In Situ Strain and Temperature Monitoring of Battery Cells, *Energies* 10(7) (2017) 838. https://doi.org/10.3390/en10070838.
