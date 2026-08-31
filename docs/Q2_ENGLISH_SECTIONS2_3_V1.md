# English manuscript V1 — Sections 2–3

# 2. Dataset and electrical–optical signal analysis

## 2.1 Dataset and operating conditions

The study uses the public SiC-18 dataset acquired from a silicon-oxide/carbon (SiOx/C) pouch cell instrumented with an implanted dual-fiber Bragg grating (FBG) sensing configuration. The tested cell has a nominal capacity of approximately 2.5 Ah. An armored FBG and a bare FBG were embedded in the cell to provide synchronized Bragg-wavelength observations associated with the internal thermo-mechanical response during operation. Terminal voltage, current, and reference SOC are recorded simultaneously, enabling direct comparison between conventional electrical measurements and internal optical observations.

The released dataset covers six representative dynamic driving profiles—HWFET, LA92, NEDC, NYCC, US06, and WLTC—at two discharge rates, 1C and 2C. The resulting 12 principal dynamic-discharge trajectories contain approximately 68,086 valid samples. Because the six profiles exhibit markedly different current amplitudes, pulse frequencies, and load-transition patterns, the dataset is suitable for evaluating both conventional SOC estimation and transfer under simultaneous rate and profile changes. The main dataset characteristics are summarized in Table 1.

**Table 1. Main characteristics of the SiC-18 dataset used in this study.**

| Item | Description |
|---|---|
| Cell | SiOx/C pouch cell, approximately 2.5 Ah |
| Optical sensing | Implanted dual FBG: armored FBG + bare FBG |
| Dynamic profiles | HWFET, LA92, NEDC, NYCC, US06, WLTC |
| Discharge rates | 1C and 2C |
| Main trajectories | 12 |
| Primary predictors | Voltage, current, W1, W2 |
| Prediction target | SOC |
| Valid released samples | Approximately 68,086 |

The primary predictors are terminal voltage V, current I, and the two measured Bragg-wavelength channels W1 and W2. SOC is used only as the supervised target and never as a model input. Variables that directly encode discharge progress, including accumulated discharge capacity and absolute trajectory time, are excluded from the estimator to avoid trivial reconstruction of the target. For every train/validation/calibration/test partition, normalization statistics are calculated exclusively from the corresponding training data.

Figure 1 illustrates representative synchronized electrical and optical observations during dynamic discharge. The current exhibits rapid and frequent load variations, whereas the optical channels evolve more smoothly while retaining local responses to operating changes. In the NEDC example, both W1 and W2 undergo substantial wavelength drift over the discharge trajectory, but their amplitudes and local temporal patterns differ. The same profile also produces different optical trajectories at 1C and 2C. These observations indicate that the electrical and dual-FBG channels are synchronized but do not simply duplicate the same dynamic information, motivating a more detailed analysis of their complementary value under operating-condition shift.

## 2.2 Dual-FBG sensing principle and thermo-mechanical decoupling

The center Bragg wavelength of an FBG is expressed as

\[
\lambda_B = 2 n_{\mathrm{eff}}\Lambda,
\]

where \(n_{\mathrm{eff}}\) is the effective refractive index and \(\Lambda\) is the grating period. Changes in temperature and strain alter both quantities, producing a measurable Bragg-wavelength shift. FBGs can therefore provide in-situ information on local thermo-mechanical changes inside or on the surface of a battery cell.

For the dual-FBG configuration used in the SiC-18 dataset, the two sensing channels have different thermal and mechanical sensitivities. The released wavelength variables are approximately related to the decoupled temperature \(T\) and deformation-force response \(F\) by

\[
W1 = 0.0208T + 0.00054F,
\]

\[
W2 = 0.0254T + 0.00085F.
\]

Equivalently,

\[
\begin{bmatrix} W1 \\ W2 \end{bmatrix}
=
\mathbf{K}
\begin{bmatrix} T \\ F \end{bmatrix},
\qquad
\mathbf{K}=
\begin{bmatrix}
0.0208 & 0.00054 \\
0.0254 & 0.00085
\end{bmatrix}.
\]

When the sensitivity matrix \(\mathbf{K}\) is invertible, the thermo-mechanical coordinates can be reconstructed from the wavelength observations. For the released coefficients, the inverse transformation is approximately

\[
T = 214.43W1 - 136.23W2,
\]

\[
F = -6407.67W1 + 5247.23W2.
\]

Thus, W1/W2 and T/F should not be regarded as four independent sensing channels. They are two coordinate systems describing the same two optical sensing degrees of freedom. The T/F representation has clearer physical semantics and is useful for interpreting internal thermo-mechanical evolution, whereas W1/W2 preserve the native coordinate system directly produced by the optical interrogation chain. For data-driven SOC estimation, however, a physically interpretable coordinate system is not necessarily the representation that transfers most effectively across operating conditions. This distinction motivates the representation analysis below.

## 2.3 Electrical–optical characteristics and representation analysis

### 2.3.1 Dynamic response differences

Under dynamic driving cycles, terminal voltage is jointly affected by SOC, ohmic voltage drop, polarization, and instantaneous current. Consequently, the electrical response can vary rapidly even when the underlying SOC evolves smoothly. The implanted FBGs, in contrast, respond to internal thermal and structural changes whose temporal evolution is not identical to the terminal electrical dynamics. As illustrated in Fig. 1, the current contains high-frequency pulses, whereas the two wavelength channels exhibit a smoother overall evolution superimposed with local dynamic responses.

Rate changes further accentuate this distinction. Moving from 1C to 2C directly changes the current amplitude and electrical-input distribution, while the FBG wavelengths continue to reflect the accumulated internal thermo-mechanical response. The optical channels are therefore treated in this study as potential complementary internal observations under operating-condition shift rather than as redundant replicas of V/I. This hypothesis is evaluated quantitatively in Section 4 through parameter-matched electrical-only and electrical–optical comparisons.

### 2.3.2 Statistical structure of native and decoupled optical coordinates

Although W1/W2 and T/F contain the same number of sensing degrees of freedom, their statistical geometry differs substantially. Figures 2(a) and 2(b) show the joint distributions of the two coordinate systems across all 12 dynamic trajectories. When the data are grouped by rate, the absolute Pearson correlation between W1 and W2 is 0.738 at 1C and 0.655 at 2C. After thermo-mechanical decoupling, the corresponding absolute correlation between T and F increases to 0.982 and 0.985, respectively, as summarized in Fig. 2(c).

This stronger linear coupling does not imply that thermo-mechanical decoupling is physically inappropriate. On the contrary, the transformation maps the two measured wavelengths to coordinates with clearer temperature and mechanical meaning. However, the inversion simultaneously changes feature scaling, correlation structure, and the way measurement perturbations are propagated through the representation. For a transfer-oriented end-to-end SOC estimator, it is therefore preferable to determine the representation empirically rather than assume that the more interpretable physical coordinates must also provide the most stable predictive geometry.

### 2.3.3 Representation-aware predictive comparison

To select the optical representation used by the final estimator, native wavelength and decoupled thermo-mechanical inputs are compared under the same causal TCN architecture, training budget, and cross-rate unseen-profile development protocol. The principal results are shown in Fig. 2(d) and Table 4. Across the six 1C→2C unseen-profile development splits, the compact TCN using W1/W2 achieves an average MAE of 1.385% SOC, an RMSE of 2.084% SOC, and a Q95 absolute error of 4.873% SOC. Using T/F in the same network increases these errors to 2.033%, 3.060%, and 6.835%, respectively. A more complex electrical–thermo-mechanical fusion model (ETMF-TF) achieves an average MAE of 1.569% SOC and also fails to outperform the compact native-wavelength estimator.

These results indicate that introducing a physically interpretable intermediate coordinate system does not necessarily improve predictive transfer. In the present cross-condition task, retaining W1/W2 preserves the complete dual-FBG measurement while avoiding an additional coordinate inversion and provides the most stable average transfer performance among the tested representations. The final estimator therefore uses V, I, W1, and W2 as its input variables, while T/F is retained for sensing interpretation and representation ablation.

Overall, dual-FBG sensing is not used in this study to stack wavelength, temperature, and force as redundant predictors. Instead, the two raw wavelength channels provide internal optical observations with a physical origin distinct from the terminal electrical response. Based on the above signal and representation analysis, Section 3 develops a lightweight causal estimator that operates directly on the native dual-FBG coordinates.

# 3. Methodology

## 3.1 Overall RA-FBG-TCN framework

A Representation-Aware Dual-FBG Temporal Convolutional Network (RA-FBG-TCN) is developed for SOC estimation under dynamic operation and changing discharge rates. The framework does not treat the thermo-mechanically decoupled variables as additional independent measurements. Instead, it first evaluates native and decoupled optical coordinates as alternative predictive representations and then uses the retained native wavelength representation together with voltage and current. A compact causal temporal convolutional network extracts the electrical–optical temporal pattern, and residual split conformal prediction is subsequently applied to calibrate the uncertainty of the frozen point estimator.

At time \(t\), the model input is

\[
\mathbf{x}_t=[V_t, I_t, W1_t, W2_t],
\]

where \(V_t\) and \(I_t\) are the terminal voltage and current, and \(W1_t\) and \(W2_t\) are the two measured Bragg-wavelength channels. A causal history window of length \(L=64\),

\[
\mathbf{X}_t=[\mathbf{x}_{t-L+1},\ldots,\mathbf{x}_t],
\]

is used to estimate the SOC at the final time step, \(\hat{y}_t\). Because each window contains only current and historical observations, the estimator is compatible with online causal inference.

As summarized in Fig. 3, the framework contains four main stages: (1) training-only normalization, (2) construction of a 64-sample causal sequence window, (3) dilated causal temporal representation learning, and (4) SOC regression followed by post-hoc 95% conformal calibration. The overall path is therefore

\[
V/I/W1/W2 \rightarrow \text{train-only normalization} \rightarrow \text{causal window} \rightarrow \text{dilated TCN} \rightarrow \text{SOC regression} \rightarrow \text{95\% conformal interval}.
\]

## 3.2 Representation-aware native dual-FBG input

Let the dual-FBG wavelength shift be denoted by

\[
\Delta\boldsymbol{\lambda}=[\Delta\lambda_1,\Delta\lambda_2]^\top
\]

and the corresponding thermo-mechanical coordinate by

\[
\mathbf{z}=[T,F]^\top.
\]

Their linear relation can be written as

\[
\Delta\boldsymbol{\lambda}=\mathbf{K}\mathbf{z},
\]

where \(\mathbf{K}\) is the 2×2 sensitivity matrix. If \(\mathbf{K}\) is invertible,

\[
\mathbf{z}=\mathbf{K}^{-1}\Delta\boldsymbol{\lambda}.
\]

Consequently, W1/W2 and T/F are alternative coordinate descriptions of the same dual-channel optical measurement. Explicit decoupling improves physical semantics but does not introduce new sensing degrees of freedom. At the same time, matrix inversion changes feature scaling, cross-variable correlation, and perturbation propagation, all of which can influence the behavior of a learned time-series model under domain shift.

For this reason, optical representation selection is treated as part of model design rather than fixed a priori. Native W1/W2 and decoupled T/F are compared under identical partitions, model capacity, and optimization budgets. As demonstrated in Section 4, native wavelength coordinates provide lower average MAE, RMSE, and Q95 error in the cross-rate unseen-profile task. The final RA-FBG-TCN therefore uses W1/W2 directly. This choice concerns predictive transferability and does not negate the value of T/F for physical interpretation.

## 3.3 Causal temporal convolutional SOC estimator

### 3.3.1 Training-only normalization

The four input variables have different physical units and numerical ranges. For each input channel \(j\), the mean \(\mu_j\) and standard deviation \(\sigma_j\) are therefore calculated from the training data only, and the normalized variable is

\[
x'_{t,j}=\frac{x_{t,j}-\mu_j}{\sigma_j}.
\]

The same frozen \(\mu_j\) and \(\sigma_j\) are then applied to validation, calibration, and test data. No target-domain statistics are re-estimated, preventing test-domain information from entering preprocessing.

### 3.3.2 Dilated causal convolution

To ensure that the prediction at time \(t\) never depends on future observations, all temporal convolutions use left-only causal padding. For a one-dimensional convolution at layer \(l\) with kernel size \(k\) and dilation \(d_l\),

\[
\mathbf{h}^{(l)}_t = \sum_{i=0}^{k-1}\mathbf{W}^{(l)}_i\mathbf{h}^{(l-1)}_{t-id_l}+\mathbf{b}^{(l)}.
\]

The four-dimensional input is first projected to a hidden width of \(C=24\) through a 1×1 convolution. Three temporal residual blocks are then stacked with dilation factors of 1, 2, and 4. Each residual block contains two causal Conv1D layers with kernel size 3, GroupNorm, and GELU activation. For a block input \(\mathbf{H}\), the computation is

\[
\mathbf{Z}_1=\mathrm{GELU}\left(\mathrm{GN}\left(\mathrm{Conv}_{\mathrm{causal}}(\mathbf{H};k=3,d)\right)\right),
\]

\[
\mathbf{Z}_2=\mathrm{GN}\left(\mathrm{Conv}_{\mathrm{causal}}(\mathbf{Z}_1;k=3,d)\right),
\]

\[
\mathbf{H}_{\mathrm{out}}=\mathrm{GELU}(\mathbf{H}+\mathbf{Z}_2).
\]

With the three dilation levels, the stacked residual encoder has an effective temporal receptive field of 29 samples, while the 64-sample input window provides additional causal historical context. This configuration keeps the model compact while allowing the encoder to represent both short transients and broader local dynamics.

### 3.3.3 Temporal representation and regression head

After the three residual TCN blocks, the encoder produces a hidden sequence \(\mathbf{H}\in\mathbb{R}^{L\times C}\). Because the objective is to estimate the SOC corresponding to the final sample of the input window, the last causal hidden state \(\mathbf{h}_t\) is used as the temporal representation. A two-layer regression head then gives

\[
\mathbf{u}_t=\mathrm{GELU}(\mathbf{W}_1\mathbf{h}_t+\mathbf{b}_1),
\]

\[
\hat{y}_t=\mathbf{W}_2\mathbf{u}_t+\mathbf{b}_2.
\]

The complete RA-FBG-TCN contains approximately 11.5k trainable parameters, making it substantially smaller than typical large Transformer-style sequence models and suitable for computationally constrained BMS or edge-controller deployment.

The point estimator is optimized using mean squared error,

\[
\mathcal{L}_{\mathrm{MSE}}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2.
\]

AdamW is used for optimization, and the checkpoint with the lowest MAE on an independent validation split is retained. The validation split is not included in the final test metrics.

**Table 2. Main RA-FBG-TCN and training configuration.**

| Item | Setting |
|---|---|
| Input variables | V, I, W1, W2 |
| Sequence length | 64 samples |
| Input projection | 4 → 24, 1×1 Conv1D |
| Residual blocks | 3 |
| Dilations | 1, 2, 4 |
| Kernel size | 3 |
| Layers per block | 2 causal Conv1D layers |
| Normalization | GroupNorm |
| Activation | GELU |
| Regression head | 24 → 24 → 1 |
| Trainable parameters | Approximately 11.5k |
| Point-estimation loss | MSE |
| Optimizer | AdamW |
| Model selection | Validation MAE, early stopping |

## 3.4 Residual split-conformal uncertainty quantification

A deterministic neural estimator provides a single SOC value but does not directly indicate prediction reliability. To obtain an uncertainty interval with an explicit coverage interpretation without introducing an additional probabilistic network, residual split conformal prediction is applied after the point estimator has been trained and frozen.

Let the independent calibration set be

\[
\mathcal{D}_{\mathrm{cal}}=\{(\mathbf{X}_i,y_i)\}_{i=1}^{n}.
\]

The frozen RA-FBG-TCN produces calibration predictions \(\hat{y}_i\), from which the nonconformity scores are defined as

\[
r_i=|y_i-\hat{y}_i|.
\]

For a target miscoverage level \(\alpha=0.05\), the finite-sample corrected residual quantile is

\[
q_{1-\alpha}=\mathrm{Quantile}_{\lceil(n+1)(1-\alpha)\rceil/n}\left(\{r_i\}_{i=1}^{n}\right).
\]

For a new input \(\mathbf{X}_t\), the 95% prediction interval is then

\[
\mathcal{C}_{0.95}(\mathbf{X}_t)=
[\hat{y}_t-q_{0.95},\hat{y}_t+q_{0.95}].
\]

Because SOC is physically bounded between 0 and 1, the interval endpoints are finally clipped to this range. The calibration procedure uses only residuals from the dedicated calibration split, does not modify the frozen point estimator, and does not use test labels to adjust interval width.

Uncertainty quality is evaluated using prediction interval coverage probability (PICP), mean prediction interval width (MPIW), and mean interval score (MIS). PICP is

\[
\mathrm{PICP}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}(y_i\in[L_i,U_i]),
\]

and MPIW is

\[
\mathrm{MPIW}=\frac{1}{N}\sum_{i=1}^{N}(U_i-L_i).
\]

For miscoverage level \(\alpha\), the interval score for sample \(i\) is

\[
\mathrm{IS}_i=(U_i-L_i)+\frac{2}{\alpha}(L_i-y_i)\mathbb{I}(y_i<L_i)
+\frac{2}{\alpha}(y_i-U_i)\mathbb{I}(y_i>U_i),
\]

and MIS is the average of \(\mathrm{IS}_i\) over the test set. A useful interval should achieve coverage close to its nominal level while remaining as narrow as possible.

By decoupling the compact causal point estimator from post-hoc conformal calibration, the proposed framework outputs both an SOC point estimate and an uncertainty interval with negligible additional online inference complexity.