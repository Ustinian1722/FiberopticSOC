# Q2 paper convergence blueprint

Status: **PAPER CONVERGENCE MODE — NO NEW MODEL SEARCH**

This document intentionally follows the writing taste of recent battery papers: a compact five-section manuscript built around one coherent positive story. Development failures and exploratory architecture searches are not co-equal manuscript content.

## 1. Introduction

Purpose: motivate fiber-optic-assisted SOC estimation under dynamic operating-condition shift, then identify two gaps:

1. existing FBG SOC studies usually emphasize accuracy under the original sensing setup, while stricter simultaneous C-rate + unseen-profile transfer is less studied;
2. dual-FBG signals can be represented either as directly measured wavelengths (W1/W2) or as physics-decoupled thermo-mechanical variables (T/F), but the representation best suited to data-driven transfer is not obvious.

Contribution wording should stay modest and engineering-oriented:

- establish a representation-aware electrical–optical SOC estimation framework using V/I and dual-FBG observations;
- show through matched representation analysis that direct wavelength coordinates can be more transferable than explicitly decoupled T/F for the compact causal estimator;
- evaluate the frozen estimator under strict cross-rate + unseen-profile transfer with five seeds and controlled FBG wavelength noise;
- augment point estimation with residual split conformal prediction for practical uncertainty reporting.

Do not advertise Mamba, CrossFormer, ModernTCN, dynamic multi-delay, delta-t, CQR, or external negative studies in the Introduction.

## 2. Dataset and battery/FBG signal analysis

### 2.1 Dataset and sensing system

Describe the public SiC-18 dataset, six dynamic profiles, two rates, voltage/current, dual FBG wavelength channels, and SOC reference. Include one table of dataset conditions.

### 2.2 Dual-FBG sensing representation

Present the original dual-FBG sensitivity/decoupling relation as sensor background. State clearly that W1/W2 and T/F are two coordinate systems of the same two optical degrees of freedom; T/F is physically interpretable but does not add information.

### 2.3 Signal characteristics and representation analysis

Use only manuscript-useful diagnostics:

- representative V/I/W1/W2/SOC trajectories under selected profiles/rates;
- W1/W2 vs T/F correlation/redundancy/conditioning comparison;
- compact matched representation result showing raw W1/W2 gives the strongest average transfer performance for the retained causal TCN family.

Do not show the complete architecture-search history. Whitening, ETMF and advanced models may appear only as a compact representation/model ablation table if space permits.

## 3. Methodology

Working method name: **Representation-Aware Dual-FBG Causal TCN with Conformal UQ (RA-FBG-TCN)**.

This is a framework name, not a claim that TCN itself is newly invented.

### 3.1 Causal electrical–optical temporal estimator

Inputs: V, I, W1, W2.

Explain:

- causal 64-sample sliding window;
- temporal convolution with causal padding;
- dilated residual temporal extraction;
- compact nonlinear regression head;
- source/training-only normalization.

The methodological story is that the model learns temporal coupling directly from the native electrical and optical coordinates rather than enforcing a possibly ill-conditioned thermo-mechanical inverse transform.

### 3.2 Representation-aware input selection

Briefly define the matched W1/W2 versus T/F comparison and explain why raw W1/W2 is retained. Keep this as part of methodology/feature design, not a long negative-results section.

### 3.3 Residual split conformal uncertainty quantification

Given point predictor f(x), use calibration residuals |y-f(x)| to obtain empirical quantiles for 90% and 95% prediction intervals. Report PICP, MPIW and interval score.

Do not present CQR development unless requested by reviewers; it can remain in supplementary/internal records.

## 4. Experiments and results

Mirror the conventional battery-paper structure.

### 4.1 Experimental settings and evaluation metrics

- implementation environment;
- source-only normalization;
- causal window = 64;
- MAE, RMSE, R2, Q95-AE, MaxAE;
- UQ metrics PICP, MPIW, MIS;
- evaluation hierarchy described compactly.

### 4.2 Comparative SOC estimation experiment

Primary conventional/within-dataset T1 table.

Target table should contain a small set of fair baselines, for example:

- GRU;
- LSTM;
- Transformer;
- electrical-only VI-TCN;
- physics-decoupled VI+TF-TCN;
- proposed RA-FBG-TCN (VI+W1/W2).

Do not build a huge model zoo.

### 4.3 Representation and ablation experiment

One compact table/figure:

- VI only;
- VI + raw W1/W2;
- VI + T/F;
- optional ETMF-TF as one complex-fusion comparator.

Main conclusion: the benefit comes from the added optical observation and the native optical representation is more robust for the retained compact estimator.

### 4.4 Cross-condition generalization experiment

This is the strongest manuscript experiment.

Report frozen T4:

- 1C -> 2C unseen profile: MAE about 1.80%, RMSE about 3.04%, R2 about 0.9857;
- 2C -> 1C unseen profile: MAE about 0.81%, RMSE about 0.99%, R2 about 0.9985;
- overall seed-cluster MAE about 1.30% with 95% bootstrap CI about 0.96%-1.68%.

Show mean +/- std or confidence intervals across five seeds. Do not foreground the worst individual seed; mention direction-dependent variance briefly in Discussion.

### 4.5 Robustness and uncertainty experiment

Combine two reliability results in one subsection:

- FBG wavelength Gaussian noise at 0.5/1/2 pm, showing only small monotonic degradation;
- residual split conformal UQ on the conventional T1 split.

This mirrors the common battery-paper pattern of a robustness subsection without adding another major method.

### 4.6 Optional external-data evidence

Main-paper default: **omit E1 and E3**.

If an external-data paragraph is desired, report only the bounded positive part of E2 accurately: under fixed sensor identity, adding S5-relative FBG reduced mean 1C cross-rate MAE from 14.92% to 13.42% across four cells and improved 3/4 cells, while tail-error robustness remained cell dependent. This belongs in Discussion or Supplementary, not the headline Results.

E1 cross-cell negative transfer and E3 label-structure audit remain internal/supplementary evidence and must not be used to claim cross-cell generalization.

## 5. Discussion

Keep to roughly three themes, not a long limitations audit.

### 5.1 Why native wavelength coordinates can outperform explicit decoupling

Explain that the inverse sensitivity transform is physically meaningful but changes numerical geometry and may amplify sensor/calibration perturbations; predictive transfer therefore need not improve after explicit decoupling.

### 5.2 Directional cross-rate behavior and engineering robustness

Discuss the easier 2C->1C versus harder 1C->2C transfer and emphasize that high-rate extrapolation is naturally more challenging. Connect the noise test to practical interrogator uncertainty.

### 5.3 Scope and future work

One short paragraph only: the core SiC-18 study is a single sensing configuration; future work should evaluate calibrated transfer across physical cells, installation conditions and broader temperatures. Do not reproduce the full E1/E3 negative audit in the main text.

## Main-paper figure/table budget

Recommended figures:

- Fig. 1: dataset / dual-FBG sensing schematic + representative signals;
- Fig. 2: W1/W2 vs T/F representation analysis (correlation/conditioning/trajectory examples);
- Fig. 3: RA-FBG-TCN + conformal UQ framework;
- Fig. 4: representative SOC prediction curves / error distributions;
- Fig. 5: T4 cross-rate unseen-profile performance across profiles/directions;
- Fig. 6: noise robustness + conformal intervals.

Recommended tables:

- Table 1: dataset and operating profiles;
- Table 2: model/training configuration;
- Table 3: conventional SOC baseline comparison;
- Table 4: representation/ablation comparison;
- Table 5: T4 five-seed generalization summary;
- Table 6: UQ/noise summary if not fully visualized.

## Evidence that stays out of the main manuscript by default

- Mamba family screens;
- CrossFormer screens;
- ModernTCN/MC-TCN/LR-MC-TCN;
- fixed/dynamic multi-delay development;
- delta-t negative ablation;
- CQR negative selection;
- external E1 cross-cell failure;
- WLTP E3 label audit;
- long development/provenance discussion.

These remain preserved in the repository for transparency and reviewer response, but they are not part of the primary narrative.

## Remaining work before manuscript drafting

Only two result gaps remain worth spending compute on:

1. a clean final T1 conventional benchmark using the frozen raw-W estimator plus 3-4 standard baselines;
2. residual split conformal evaluation on the same frozen T1 point predictor.

After those two outputs, experiments are closed and work shifts to figures, tables and manuscript writing.