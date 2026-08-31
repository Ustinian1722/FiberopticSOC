# English manuscript V2 — Sections 4–6

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

The resulting 95% nominal interval attains an empirical PICP of 95.04%, closely matching the target coverage. The mean prediction interval width is 2.075% SOC, the mean interval score is 0.02441, and the corresponding absolute-residual quantile is approximately 1.089% SOC. Figure 7(c) illustrates the point estimate and calibrated interval over a representative test segment. These results show that a simple post-hoc conformal layer can supplement the deterministic point estimator with an uncertainty interval whose empirical coverage is aligned with the nominal level, without requiring a second probabilistic neural network.

Taken together, the experiments support a transfer-oriented interpretation of the proposed framework. RA-FBG-TCN retains sub-percent conventional accuracy, provides substantial optical benefit in strongly shifted electrical regions, maintains useful performance under simultaneous C-rate and unseen-profile changes, degrades smoothly under pm-scale wavelength perturbations, and supports calibrated uncertainty reporting.

# 5. Discussion

## 5.1. Physical decoupling versus predictive representation

The dual-FBG system permits the two measured Bragg wavelengths to be transformed into temperature- and force-related variables using sensor sensitivities. From a sensing perspective, this decoupling is valuable because it assigns clearer physical meaning to the optical response. The present results nevertheless demonstrate that stronger physical interpretability does not automatically imply a better representation for cross-condition data-driven prediction.

W1/W2 and T/F describe the same two optical sensing degrees of freedom and are related through an invertible linear transformation. The transformation changes the coordinate system rather than adding measurements. In the present data, it also produces substantially stronger linear coupling between the two variables and changes feature scaling and perturbation propagation. These properties can affect how a temporal model extracts patterns that remain stable outside the source operating condition.

The matched representation experiments illustrate this distinction empirically. Native W1/W2 yields lower average cross-condition error than both the same TCN using T/F and a more complex electrical–thermo-mechanical fusion model. This should not be interpreted as evidence against physical decoupling. Rather, it highlights that sensing interpretation and predictive representation serve different objectives: one seeks physically meaningful variables, whereas the other seeks coordinates that preserve task-relevant information in a form that transfers reliably. For the present SOC task, retaining the native optical coordinates is the more effective engineering choice.

## 5.2. Why optical value emerges under distribution shift

The blocked-interpolation experiment shows that an electrical-only TCN can outperform the electrical–optical estimator when the test distribution is already well supported. Terminal voltage and current contain strong SOC information in this regime, so additional FBG observations can be partly redundant. This result is therefore not contradictory to the proposed sensing strategy; it clarifies where the auxiliary modality is actually useful.

During 1C→2C transfer, the target current increasingly enters regions that were absent or rare during source-rate training. A V/I-only estimator must then extrapolate from unfamiliar electrical excitation. The FBG wavelengths are influenced not only by instantaneous load but also by internal thermo-mechanical evolution coupled to battery state. Their response is therefore not identical to the terminal electrical variables, allowing them to provide additional state constraints when electrical support weakens.

The OOD-stratified results make this mechanism visible at the data level. Optical input is not beneficial in the fully supported ID region, but its relative contribution becomes positive once test windows contain out-of-support electrical states and rises to approximately 48.5% in the most severe OOD region. This pattern suggests a broader interpretation for multimodal battery sensing: an auxiliary modality need not outperform the primary modality everywhere to be valuable. Its practical role can instead be to supply complementary information when the primary sensing pathway becomes poorly supported by the training distribution.

For BMS deployment, this distinction is important because real operating conditions cannot be exhaustively represented during offline training. Rate changes, aggressive acceleration, and other high-power transients may move the electrical measurements outside historical support. Internal optical sensing can therefore be viewed as an additional state constraint for difficult operating conditions rather than as a replacement for voltage and current sensing.

## 5.3. Transfer asymmetry, sensing robustness, and uncertainty

The strict generalization experiment reveals a pronounced directional asymmetry: 2C→1C transfer is considerably easier than 1C→2C transfer. This difference is consistent with the support of the source current distribution. Training at 2C exposes the estimator to a broader high-current range, so lower-rate target states remain largely covered. Training at 1C does not provide equivalent coverage of the highest-current 2C states and therefore requires a more difficult extrapolation.

The wavelength-noise and conformal experiments provide complementary reliability evidence. Native W1/W2 produces smooth rather than abrupt degradation under direct pm-scale perturbation, while the residual conformal interval reaches empirical coverage essentially equal to its 95% nominal target. The point estimator and uncertainty layer are deliberately decoupled: the former remains compact and causal, while the latter uses calibration residuals to add uncertainty information without materially increasing online inference complexity.

Together, these results suggest that robust multimodal SOC estimation should be assessed through more than a single average error measured near the training distribution. Transfer direction, sensing support, perturbation sensitivity, and predictive uncertainty all influence whether a model is useful in realistic operating conditions.

## 5.4. Scope and future work

The present study focuses on operating-condition transfer within a fixed dual-FBG sensing configuration. The conclusions therefore apply most directly when sensor installation and calibration remain consistent while discharge rate and driving profile change. Across different physical cells, FBG initial wavelength, bonding condition, strain-transfer efficiency, and sensor-specific sensitivity may vary and can alter both the absolute level and dynamics of the optical signal.

Future work should therefore investigate sensor-aware calibration and adaptation across multiple cells, broader temperature ranges, long-term sensor aging, and changes in bonding condition. These sensing-calibration and transfer issues may ultimately be more important for practical deployment than further increasing neural-network complexity. Online calibration and lightweight adaptive mechanisms are particularly relevant for translating fiber-optic-assisted SOC estimation from controlled datasets to operational BMS applications.

Overall, the present results indicate that FBG sensing should not be judged solely by whether it reduces interpolation error on every sample. Under meaningful operating-condition shift, optical observations can provide a distinct internal-state constraint when conventional electrical measurements leave their training support. Combining this sensing complementarity with a compact causal estimator and calibrated uncertainty offers a practical route toward more robust multimodal battery-state estimation.

# 6. Conclusion

This study investigated representation-aware dual-FBG optical sensing for battery SOC estimation under operating-condition shift. Native wavelength coordinates and thermo-mechanically decoupled temperature/force coordinates were analyzed as alternative representations of the same dual-FBG sensing information. Under matched cross-condition evaluation, W1/W2 exhibited more stable predictive transfer, motivating a compact RA-FBG-TCN that combines voltage, current, and the two native wavelength channels in a 64-sample causal input sequence.

Under blocked interpolation, RA-FBG-TCN achieved an MAE of 0.482% SOC, an RMSE of 0.593% SOC, and an R² of 0.999614. More importantly, parameter-matched OOD analysis showed that incorporating W1/W2 reduced the overall 1C→2C MAE from 2.151% to 1.632% SOC, while the relative optical benefit increased with electrical-OOD severity and reached approximately 48.5% in the most shifted region. The strict cross-rate unseen-profile evaluation yielded average MAEs of 1.795% SOC for 1C→2C and 0.806% SOC for 2C→1C, with an overall seed-cluster MAE of 1.301% SOC and a bootstrap 95% confidence interval of 0.961–1.677% SOC.

Direct wavelength perturbations up to 2 pm caused only limited error growth. The 95% residual conformal interval attained 95.04% empirical coverage with a mean width of 2.075% SOC, providing calibrated uncertainty without modifying the point estimator. The combined evidence indicates that the main value of internal optical sensing is not simply the addition of more predictor variables, but the provision of state information from a different physical pathway when terminal electrical observations become weakly supported by the source distribution. Future work will extend this framework toward multi-cell sensor calibration, wider temperature conditions, sensor aging, and online adaptation.