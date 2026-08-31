# English manuscript V1 — Sections 4–6

# 4. Experiments and results

## 4.1 Experimental settings and evaluation metrics

The proposed framework is evaluated from five complementary perspectives: conventional mixed-condition SOC accuracy, electrical–optical complementarity under distribution shift, optical-representation choice, strict cross-rate unseen-profile generalization, and reliability under wavelength perturbation and uncertainty calibration. All input normalization statistics are calculated exclusively from the corresponding training data. A causal window of 64 samples is used throughout, and the estimator predicts the SOC associated with the final sample in each window without accessing future information.

AdamW is used to optimize the neural models, while an independent validation split is used for early stopping and checkpoint selection. Point-estimation performance is evaluated using mean absolute error (MAE), root mean square error (RMSE), the coefficient of determination (R²), the 95th-percentile absolute error (Q95-AE), and maximum absolute error (MaxAE). MAE and RMSE characterize overall accuracy, whereas Q95-AE and MaxAE describe the upper tail and extreme prediction error. Prediction intervals are evaluated using prediction interval coverage probability (PICP), mean prediction interval width (MPIW), and mean interval score (MIS).

Two principal evaluation regimes are considered. The first is a blocked mixed-condition interpolation setting, which evaluates conventional estimation accuracy when the operating distribution is well represented during training. The second is a stricter cross-rate unseen-profile protocol: for each test case, the model is trained using five complete driving profiles from one C-rate and is evaluated on the sixth profile at the opposite C-rate, with the held-out profile entirely excluded from training. This second protocol simultaneously introduces a discharge-rate shift and a driving-profile shift.

## 4.2 Conventional SOC estimation performance

Table 3 compares representative sequence models under the blocked mixed-condition interpolation setting. CNN, GRU, LSTM, Transformer, and the different TCN input variants use the same train/validation/test partitions and the same normalization and early-stopping rules.

**Table 3. Conventional blocked-interpolation SOC estimation performance.**

| Model | Params | MAE (% SOC) | RMSE (% SOC) | R² | Q95-AE (% SOC) |
|---|---:|---:|---:|---:|---:|
| VI-TCN | 11,497 | **0.231** | **0.296** | **0.999904** | **0.581** |
| VI+TF-TCN | 11,545 | 0.311 | 0.412 | 0.999814 | 0.825 |
| GRU | 10,801 | 0.426 | 0.549 | 0.999670 | 1.014 |
| RA-FBG-TCN | 11,545 | 0.482 | 0.593 | 0.999614 | 1.088 |
| LSTM | 14,129 | 0.548 | 0.708 | 0.999450 | 1.303 |
| CNN | 12,081 | 0.564 | 0.738 | 0.999403 | 1.414 |
| Transformer | 20,177 | 0.761 | 0.910 | 0.999090 | 1.584 |

All retained models achieve high SOC accuracy in this relatively well-supported setting. The electrical-only VI-TCN gives the lowest point-estimation error, indicating that voltage and current already provide highly informative SOC cues when the test distribution is well represented by the training data. The proposed RA-FBG-TCN nevertheless achieves an MAE of 0.482% SOC, an RMSE of 0.593% SOC, and an R² of 0.999614 with only approximately 11.5k trainable parameters. Representative trajectories and the corresponding error distribution are shown in Fig. 4.

The role of FBG sensing is therefore not interpreted as a universal in-distribution accuracy gain. Instead, the following experiments focus on whether optical measurements provide additional information when conventional electrical observations move outside their training support.

## 4.3 Electrical–optical complementarity under operating-condition shift

To isolate the contribution of the optical channels, two parameter-matched causal TCNs are compared. The electrical-only model, denoted VI, receives voltage and current while the remaining two input channels are set to zero. The electrical–optical model, VI+W, uses voltage, current, W1, and W2. Both models have the same architecture and parameter count, so their difference reflects input information rather than model capacity.

For 1C→2C transfer, the average MAE of the VI model is 2.151% SOC, whereas adding the dual-FBG wavelengths reduces the MAE to 1.632% SOC, corresponding to a relative reduction of approximately 24.1%. For 2C→1C transfer, the MAE decreases from 0.866% to 0.814% SOC. The stronger improvement in the 1C→2C direction suggests that optical observations become particularly useful when the target current amplitude extends beyond the range represented in the source-rate data.

To examine this effect directly, an electrical support envelope is constructed from the 0.5th–99.5th percentile range of the training-set current only. For each test window, the fraction of samples outside this envelope is defined as the electrical-OOD fraction. This score is label-free with respect to target SOC because it depends only on the source-current distribution and the observed test current.

Figure 5 and Table 4 show the resulting error stratification for 1C→2C transfer.

**Table 4. Parameter-matched electrical–optical comparison under increasing electrical-OOD severity.**

| Electrical-OOD level | VI MAE (% SOC) | VI+W MAE (% SOC) | Relative optical gain |
|---|---:|---:|---:|
| ID | **0.735** | 0.873 | −18.75% |
| OOD 0–25% | 1.550 | **1.470** | +5.17% |
| OOD 25–50% | 2.772 | **2.356** | +15.00% |
| OOD 50–75% | 3.665 | **2.930** | +20.05% |
| OOD 75–100% | 5.529 | **2.846** | **+48.52%** |

Inside the electrical training support, adding FBG observations does not improve the electrical-only estimator. Once the test trajectory begins to leave that support, however, the optical benefit becomes positive and increases monotonically with OOD severity. In the most shifted windows, the relative MAE reduction reaches 48.52%. The result demonstrates that the value of optical sensing is strongly condition dependent: the FBG channels are not required to duplicate already sufficient electrical information, but they provide increasingly useful complementary state information as the terminal electrical excitation becomes unfamiliar to the source-trained model.

This behavior is physically plausible because voltage and current primarily describe the external electrical response, whereas the implanted FBGs also respond to internal thermo-mechanical evolution. These modalities are therefore affected differently by operating-condition changes. When the target current trajectory extends beyond the electrical support represented during training, the optical channels can still retain state-related information that constrains SOC estimation.

### 4.3.1 Native wavelength versus thermo-mechanically decoupled representation

The two native FBG wavelengths can be linearly transformed into temperature/force coordinates using the sensitivity matrix. Although this decoupling is physically meaningful, the two representations contain the same number of optical sensing degrees of freedom. Their suitability for predictive transfer is therefore assessed under matched modeling conditions.

Across six 1C→2C unseen-profile development splits, the compact TCN using native W1/W2 achieves an average MAE of 1.385% SOC, an RMSE of 2.084% SOC, and a Q95-AE of 4.873% SOC. Replacing W1/W2 with the decoupled T/F variables in the same network increases these errors to 2.033%, 3.060%, and 6.835% SOC. A more complex electrical–thermo-mechanical fusion model achieves an average MAE of 1.569% SOC and also does not outperform the compact native-wavelength model.

The descriptive statistics in Fig. 2 further show that the native W1/W2 coordinates have absolute Pearson correlations of 0.738 and 0.655 at 1C and 2C, respectively, while the T/F coordinates exhibit much stronger linear coupling, with corresponding values of 0.982 and 0.985. The decoupled coordinates therefore provide clearer physical semantics but a markedly different predictive geometry. For the transfer-oriented task considered here, directly retaining the native sensor coordinates yields the most stable average performance and is therefore adopted in the final RA-FBG-TCN.

## 4.4 Cross-rate unseen-profile generalization

After fixing the optical representation and estimator, the final model is evaluated using the strict cross-rate unseen-profile protocol. For each held-out case, neither the target rate nor the same named driving profile appears in the training data. Six held-out profiles are evaluated in each transfer direction, and every split is repeated with five independent random seeds.

The aggregate results are summarized in Table 5 and Fig. 6. For 1C→2C transfer, RA-FBG-TCN achieves an average MAE of 1.795% SOC, an RMSE of 3.037% SOC, and an R² of 0.985689. In the reverse 2C→1C direction, the average MAE decreases to 0.806% SOC, the RMSE to 0.988% SOC, and R² increases to 0.998480. Across both directions, the seed-cluster mean MAE is 1.301% SOC, with a bootstrap 95% confidence interval of 0.961–1.677% SOC.

**Table 5. Five-seed cross-rate unseen-profile performance.**

| Direction | MAE (% SOC) | RMSE (% SOC) | R² | Q95-AE (% SOC) |
|---|---:|---:|---:|---:|
| 1C→2C | 1.795 | 3.037 | 0.985689 | 6.804 |
| 2C→1C | **0.806** | **0.988** | **0.998480** | **1.830** |
| Overall | **1.301** | — | — | — |

The two transfer directions are clearly asymmetric. The 2C source data cover a wider range of high-current excitation, so transferring from 2C to 1C keeps most target-current states within the source support. In contrast, 1C→2C requires extrapolation toward stronger and faster current excitation that is not fully represented during training. This interpretation is consistent with the electrical-OOD analysis in Section 4.3 and explains the larger errors and higher seed variability observed in the low-to-high-rate direction.

## 4.5 Wavelength-noise robustness and conformal uncertainty

### 4.5.1 Robustness to FBG wavelength perturbation

Practical FBG interrogation is subject to finite measurement noise. To evaluate sensitivity to such perturbations, independent zero-mean Gaussian noise with standard deviations of 0.5, 1, and 2 pm is added directly to W1 and W2. The trained models and training-derived normalization parameters are kept fixed, and no target-domain transform is re-estimated.

As shown in Fig. 7(a,b), the error increases smoothly with wavelength-noise level rather than exhibiting abrupt degradation. At the largest tested perturbation of 2 pm per channel, the MAE increases by only approximately 1.67% relative to the clean baseline in the 1C→2C direction and 4.57% in the 2C→1C direction. The direct use of native wavelengths therefore does not introduce a strong amplification of small optical measurement errors within the tested perturbation range.

### 4.5.2 95% residual conformal prediction interval

The frozen RA-FBG-TCN point estimator is further calibrated using residual split conformal prediction. Absolute residuals from the independent calibration split define the 95% conformal quantile, after which the same fixed interval rule is applied to the held-out test data.

The resulting 95% nominal interval achieves an empirical PICP of 95.04%, closely matching the target coverage. The mean prediction interval width is 2.075% SOC, the mean interval score is 0.02441, and the corresponding residual quantile is approximately 1.089% SOC. A representative calibrated interval is shown in Fig. 7(c). These results demonstrate that a simple post-hoc conformal layer can complement the high-accuracy deterministic estimator with an uncertainty interval whose empirical coverage is aligned with the nominal level, without introducing a second complex probabilistic model.

Overall, the experimental results show that the proposed framework is most useful when viewed as a compact transfer-oriented electrical–optical estimator rather than as a model designed to minimize interpolation error at all costs. It retains sub-percent conventional accuracy, provides substantial optical benefit in strongly shifted electrical regions, remains effective under simultaneous C-rate and unseen-profile changes, tolerates pm-scale wavelength perturbations, and supports calibrated uncertainty reporting.

# 5. Discussion

## 5.1 Physical decoupling versus predictive representation

A defining feature of the dual-FBG system is that the two measured Bragg wavelengths can be transformed into temperature- and force-related variables using different sensor sensitivities. From a sensing perspective, this decoupling is valuable because it assigns clearer physical meaning to the optical response. The present results nevertheless show that greater physical interpretability does not automatically imply a better predictive representation for cross-condition data-driven modeling.

W1/W2 and T/F describe the same two optical sensing degrees of freedom and are related by an invertible linear transformation. The transformation therefore changes the coordinate system rather than adding new measurements. In the present dataset, it also produces substantially stronger linear coupling between the two coordinates. Such changes in feature geometry and perturbation propagation can alter how a temporal model extracts transferable patterns, even though the underlying sensing information is nominally equivalent.

The matched representation experiments demonstrate this distinction empirically. Native W1/W2 gives lower average cross-condition error than both the same TCN using T/F and a more complex electrical–thermo-mechanical fusion architecture. The result should not be interpreted as evidence that physical decoupling is invalid. Instead, it highlights the difference between a coordinate system that is useful for interpreting sensing mechanisms and one that is well conditioned for predictive transfer. For an end-to-end SOC estimator, retaining the native sensor coordinates and allowing the network to learn task-relevant coupling can be an effective engineering choice.

## 5.2 Why the value of FBG sensing emerges under distribution shift

The conventional blocked-interpolation experiment shows that an electrical-only TCN can outperform the electrical–optical model when the test distribution is well represented during training. This is expected because terminal voltage and current already encode highly informative SOC cues in a supported operating region. Under these conditions, some optical information is redundant with the electrical response, while the additional modality may introduce measurement noise or delayed internal dynamics that are unnecessary for a simple interpolation task.

The situation changes when the electrical operating condition shifts. During 1C→2C transfer, a substantial fraction of the target trajectory extends beyond the current amplitudes observed at the source rate, forcing an electrical-only estimator to extrapolate. The implanted FBG signals are not governed solely by the instantaneous current; they also reflect internal thermo-mechanical evolution coupled to the battery state. Their response therefore remains partially distinct from the terminal electrical variables.

The electrical-OOD stratification provides direct evidence for this interpretation. Optical sensing is not beneficial in the fully supported ID region, but the relative benefit becomes positive as soon as the test windows contain out-of-support electrical states and grows to approximately 48.5% in the most severe OOD bin. This pattern suggests a broader principle for multimodal battery sensing: an auxiliary modality need not dominate the primary modality everywhere to be valuable. Its practical role may instead be to provide complementary information precisely when the primary sensing channel becomes unreliable or poorly supported by the training distribution.

From a BMS perspective, this behavior is relevant because real vehicle operation cannot be fully enumerated during offline training. High-power events, aggressive acceleration, or changing rate demands may drive the electrical signals into previously unseen regions. Internal optical sensing can therefore be interpreted as an additional state constraint for difficult operating conditions rather than as a replacement for conventional voltage and current measurements.

## 5.3 Asymmetric cross-rate transfer and robustness

The strict generalization experiment reveals a pronounced directional asymmetry: 2C→1C transfer is considerably easier than 1C→2C transfer. This behavior can be explained by the support of the source-rate current distribution. Training at 2C exposes the model to a wider range of strong current excitation, so most 1C target states remain within or close to the source support. Training at 1C, in contrast, does not sufficiently cover the highest-current states encountered at 2C and therefore requires extrapolation toward a more demanding electrical regime.

This support-based explanation is consistent with the OOD analysis and indicates that the direction-specific performance is not merely a consequence of random training variation. Even under the more difficult low-to-high-rate transfer, however, the final model preserves useful accuracy across unseen driving profiles. The direct wavelength perturbation experiment further shows that the native optical representation degrades smoothly under pm-scale measurement noise rather than becoming unstable. Combined with the nearly nominal 95% conformal coverage, these results suggest a practical balance among accuracy, sensing robustness, and uncertainty reporting.

## 5.4 Scope and future work

The present study focuses on operating-condition transfer within a fixed dual-FBG sensing configuration. The conclusions therefore apply most directly to scenarios in which the sensor installation and calibration system remain consistent while the discharge rate and driving profile change. Across different physical cells, FBG initial wavelength, bonding condition, strain-transfer efficiency, and sensor-specific sensitivity may vary substantially and can alter both the absolute level and dynamics of the optical response.

Future work should therefore investigate sensor-aware calibration and adaptation across multiple cells, wider temperature ranges, long-term sensor aging, and changing bonding conditions. Such studies would help separate battery-state variation from sensor-installation variation and clarify the conditions under which native optical coordinates can be transferred across sensing systems. Online calibration and lightweight adaptive mechanisms are particularly relevant for translating fiber-optic-assisted SOC estimation from controlled datasets to practical BMS deployment.

More generally, the results indicate that the value of FBG sensing should not be judged solely by whether it reduces interpolation error on every sample. For SOC estimation under meaningful operating-condition change, optical observations can provide a distinct internal-state constraint when the conventional electrical measurements leave their training support. Combining this sensing complementarity with a compact causal estimator and calibrated uncertainty therefore offers a practical route toward more robust multimodal battery-state estimation.

# 6. Conclusion

This study investigated representation-aware dual-FBG optical sensing for battery SOC estimation under operating-condition shift. Rather than treating thermo-mechanically decoupled variables as additional independent measurements, the native wavelength coordinates and the decoupled temperature/force coordinates were analyzed as alternative representations of the same dual-FBG sensing information. Under matched cross-condition evaluation, the native W1/W2 representation exhibited more stable predictive transfer. The final RA-FBG-TCN therefore combines voltage, current, W1, and W2 in a 64-sample causal sequence and uses a compact temporal convolutional network with approximately 11.5k trainable parameters.

Under conventional blocked interpolation, RA-FBG-TCN achieves an MAE of 0.482% SOC, an RMSE of 0.593% SOC, and an R² of 0.999614. More importantly, the electrical-OOD analysis demonstrates that the contribution of FBG sensing depends strongly on operating-condition support. In the 1C→2C transfer setting, incorporating W1/W2 reduces the overall MAE from 2.151% to 1.632% SOC, and the relative optical benefit increases to approximately 48.5% in the most severely shifted electrical region. This result indicates that optical sensing is particularly useful as a complementary internal-state observation when the conventional electrical variables move beyond the source training distribution.

Under the stricter cross-rate plus unseen-profile protocol, the model achieves average MAEs of 1.795% SOC for 1C→2C and 0.806% SOC for 2C→1C, with an overall seed-cluster MAE of 1.301% SOC and a bootstrap 95% confidence interval of 0.961–1.677% SOC. Adding up to 2 pm Gaussian perturbation independently to the two wavelength channels causes only limited error growth, while the 95% residual conformal interval attains 95.04% empirical coverage with a mean width of 2.075% SOC.

Overall, the findings show that the practical significance of internal optical sensing is not simply the addition of more input variables. Its main value lies in providing state information of a different physical origin when the dominant electrical observations become weakly supported by the training distribution. The study also shows that a physically interpretable decoupled coordinate system is not necessarily the most transferable representation for data-driven prediction. Future work will extend the framework toward multi-cell sensor calibration, broader temperature conditions, sensor aging, and online adaptation to support practical deployment of fiber-optic-assisted SOC estimation.