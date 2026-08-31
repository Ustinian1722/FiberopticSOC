# English manuscript V2 — Sections 2–3

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

where \(n_{\mathrm{eff}}\) denotes the effective refractive index and \(\Lambda\) is the grating period. Changes in temperature and strain modify both quantities, shifting the reflected Bragg wavelength. FBGs can consequently provide in-situ observations of local thermal and mechanical evolution through wavelength variation.

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

This relationship is important for the modeling strategy adopted in this work. W1/W2 and T/F are not four independent sensing channels; rather, they are two coordinate systems describing the same two optical sensing degrees of freedom. T/F offers clearer physical interpretation for analyzing internal thermal and mechanical evolution, whereas W1/W2 preserves the native coordinates directly produced by the optical interrogation system. Whether the physically decoupled representation is also the most transferable representation for data-driven SOC estimation is an empirical question rather than an assumption.

## 2.3. Electrical–optical characteristics and representation analysis

### 2.3.1. Dynamic-response differences under varying load

Under a dynamic driving cycle, terminal voltage is jointly affected by SOC, ohmic drop, polarization, and instantaneous current. As a result, the electrical response can fluctuate rapidly even when SOC evolves smoothly. The implanted FBGs respond to internal thermo-mechanical evolution, whose characteristic time scales need not coincide with those of the terminal electrical variables. The synchronized sequences in Fig. 1 illustrate this difference: current contains high-frequency pulses, whereas the wavelength channels evolve more smoothly while retaining local responses to the changing load.

The contrast becomes more pronounced when the discharge rate changes. Moving from 1C to 2C directly expands the current range and alters the terminal electrical response, whereas the FBG wavelengths continue to reflect the accumulated and coupled internal thermo-mechanical state. The optical measurements are treated here as potentially complementary internal observations rather than as duplicate measurements of V/I. Section 4 evaluates this hypothesis using parameter-matched models and an electrical-OOD analysis.

### 2.3.2. Statistical structure of native and decoupled optical coordinates

Although W1/W2 and T/F contain the same number of optical degrees of freedom, their statistical structures differ substantially. Figure 2(a,b) shows the joint distributions of the two representations over all 12 dynamic trajectories. When the data are grouped by C-rate, the absolute Pearson correlation between W1 and W2 is 0.738 at 1C and 0.655 at 2C. After thermo-mechanical decoupling, the corresponding absolute T/F correlations increase to 0.982 and 0.985, respectively, as summarized in Fig. 2(c). Thus, the two decoupled variables are much more linearly coupled in this dataset.

This observation does not diminish the physical value of thermo-mechanical decoupling. Mapping the two measured wavelengths to temperature- and force-related coordinates provides clearer physical semantics and is useful for sensing-mechanism interpretation. However, the linear inversion also changes feature scale, correlation structure, and the way measurement perturbations propagate through the representation. For an end-to-end learning problem focused on cross-rate and unseen-profile transfer, a physically interpretable coordinate system should not be assumed a priori to provide the most favorable predictive representation.

### 2.3.3. Representation-aware predictive comparison

To select the optical input representation, W1/W2 and T/F were compared under the same causal TCN architecture, training budget, and 1C→2C unseen-profile development protocol. Figure 2(d) reports the development-stage representation results. Across the six held-out-profile splits, the compact TCN using native W1/W2 achieved an average MAE of 1.385% SOC, RMSE of 2.084% SOC, and Q95 absolute error of 4.873% SOC. Replacing W1/W2 with the decoupled T/F coordinates increased the corresponding errors to 2.033%, 3.060%, and 6.835% SOC. A more complex electrical–thermo-mechanical fusion model (ETMF-TF) reduced the MAE to 1.569% SOC but still did not outperform the compact model operating directly in wavelength space.

These results show that adding a physical interpretation layer does not necessarily improve predictive transfer under condition shift. Native W1/W2 preserves the complete dual-FBG measurement while avoiding an additional coordinate inversion and provides more stable average generalization in the matched experiments. Accordingly, the final estimator uses V, I, W1, and W2 as inputs, whereas T/F is retained for sensing interpretation and representation ablation.

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

The two representations encode the same two measured optical degrees of freedom. Explicit decoupling improves physical semantics but does not introduce new sensing information. It also modifies the correlation and scale of the features and can alter the propagation of wavelength perturbations in feature space.

For this reason, representation choice is treated as part of model design. Native W1/W2 and decoupled T/F are evaluated using the same data partitions, model capacity, and training budget. The native wavelength representation gives lower average MAE, RMSE, and Q95 error in the cross-rate unseen-profile development comparison described in Section 2.3.3. RA-FBG-TCN consequently fixes W1/W2 as the optical inputs. This decision concerns predictive transferability and does not imply that T/F decoupling lacks value for physical interpretation.

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

with AdamW. Across experiments, the optimizer uses a learning rate of \(10^{-3}\), weight decay of \(10^{-4}\), gradient clipping at 2.0, and a batch size of 256. Training termination depends on the evaluation protocol. In the conventional blocked-interpolation experiment, training is capped at 50 epochs and uses validation-MAE early stopping with a minimum of 10 epochs and patience of 7. For the strict cross-rate plus unseen-profile experiments, target-domain observations are not used for early stopping. Instead, a source-only nested profile-validation procedure is run before target evaluation, and the resulting split-specific epoch counts are frozen. The final strict experiments use these fixed budgets, which range from 29 to 65 epochs across the 12 direction/profile splits. This distinction prevents target-condition performance from influencing training duration. The main architecture and protocol-specific training settings are summarized in Table 2.

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
| Conventional blocked training | Maximum 50 epochs; validation-MAE early stopping, minimum 10 epochs, patience 7 |
| Strict transfer training | Source-only selected fixed epochs; 29–65 epochs across the 12 frozen splits |
| Target-domain early stopping | Not used in strict cross-rate/unseen-profile evaluation |

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
