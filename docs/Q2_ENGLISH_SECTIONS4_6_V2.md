# English manuscript V2 — Sections 4–6

# 4. Experiments and results

## 4.1. Experimental settings and evaluation metrics

The proposed framework is evaluated through six complementary experiments: conventional mixed-condition accuracy, strict cross-condition backbone comparison, parameter-matched input ablation, electrical-OOD analysis, five-seed cross-rate unseen-profile generalization, and reliability assessment through wavelength perturbation and conformal uncertainty calibration. Unless otherwise stated, all experiments use a 64-sample causal window and training-only normalization. Test data are never used to estimate normalization statistics, select epochs, or tune the uncertainty interval.

Point-estimation performance is evaluated using mean absolute error (MAE), root mean square error (RMSE), coefficient of determination (R²), 95th-percentile absolute error (Q95-AE), and maximum absolute error (MaxAE). MAE and RMSE quantify average prediction accuracy, whereas Q95-AE and MaxAE characterize the upper error tail and extreme deviations. Uncertainty estimates are evaluated using prediction interval coverage probability (PICP), mean prediction interval width (MPIW), and mean interval score (MIS).

Two data regimes are emphasized. The first is a blocked mixed-condition interpolation setting, which evaluates conventional SOC estimation when the operating distribution is well represented during training. The second is a strict cross-rate plus unseen-profile setting. For each held-out case, the model is trained on five complete driving profiles from one C-rate and evaluated on the sixth profile at the opposite C-rate, while the same named profile is excluded from training. Thus, both C-rate and driving-profile changes occur simultaneously.

For the strict backbone comparison, all candidate models receive the same V/I/W1/W2 inputs and use seed 42. To avoid target-dependent early stopping, each held-out split is trained for the same source-only epoch budget frozen before target evaluation. This is an equal-budget architecture comparison rather than a model-specific hyperparameter search. The five-seed input ablation and final RA-FBG-TCN generalization study use seeds 0–4 and the same source-only frozen epoch plan. This separates architecture comparison from initialization-robustness reporting while preserving the same test protocol.

## 4.2. Conventional SOC estimation performance

Table 3 compares representative sequence models under blocked mixed-condition interpolation. CNN, GRU, LSTM, Transformer, and the TCN input variants use the same partitions, training-derived normalization, and early-stopping rule.

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

All models attain high accuracy in this well-supported interpolation setting. The electrical-only VI-TCN gives the lowest error, indicating that voltage and current already provide highly informative SOC cues when the test distribution is well represented. RA-FBG-TCN nevertheless achieves an MAE of 0.482% SOC, an RMSE of 0.593% SOC, and an R² of 0.999614 with only approximately 11.5k trainable parameters. Representative test trajectories and the corresponding error distribution are shown in Fig. 4. This experiment establishes the conventional estimation capability of the proposed model; the contribution of optical sensing under more difficult transfer is examined separately below.

## 4.3. Strict cross-condition backbone comparison

A strict backbone benchmark is conducted to determine whether the compact TCN remains competitive when both C-rate and driving profile change. Seven sequence models are compared using exactly the same V/I/W1/W2 inputs: CNN, GRU, LSTM, Transformer, a matched CNN–GRU–attention comparator (CGA-Matched), a DualTCN-Transformer, and RA-FBG-TCN. All models use seed 42 and the same split-specific source-only frozen training epochs.

**Table 4. Strict cross-rate plus unseen-profile backbone comparison using common V/I/W1/W2 inputs.**

| Model | Parameters | 1C→2C MAE (% SOC) | 2C→1C MAE (% SOC) | Pooled MAE (% SOC) |
|---|---:|---:|---:|---:|
| RA-FBG-TCN | 11,545 | 1.163 | **0.565** | **0.864** |
| GRU | 10,801 | **1.059** | 0.762 | 0.910 |
| LSTM | 14,129 | 1.122 | 0.848 | 0.985 |
| DualTCN-Transformer | 64,705 | 1.288 | 0.732 | 1.010 |
| CNN | 12,081 | 1.274 | 0.757 | 1.015 |
| CGA-Matched | 19,314 | 2.035 | 0.749 | 1.392 |
| Transformer | 20,177 | 2.025 | 1.128 | 1.576 |

In the more difficult 1C→2C direction, GRU gives the lowest MAE at 1.059% SOC, followed by LSTM at 1.122% and RA-FBG-TCN at 1.163%. RA-FBG-TCN ranks third rather than first in this direction, but still outperforms CNN, DualTCN-Transformer, Transformer, and CGA-Matched. In the reverse 2C→1C direction, RA-FBG-TCN achieves the best MAE of 0.565% SOC. Across all 12 direction/profile splits, its pooled MAE is 0.864% SOC, the lowest among the seven compared backbones.

The result also highlights model efficiency. RA-FBG-TCN contains 11,545 trainable parameters, compared with 64,705 for DualTCN-Transformer and 20,177 for the Transformer baseline. The strict benchmark supports the proposed TCN as a compact and competitive electrical–optical estimator without requiring a substantially larger attention-based architecture. It does not imply that RA-FBG-TCN is optimal for every metric: for example, GRU gives a lower pooled RMSE and Q95-AE. The main model claim is limited to competitive cross-condition accuracy with the lowest pooled MAE and low parameter count.

## 4.4. Five-seed input and optical-representation ablation

To isolate the effect of FBG information from architecture capacity, the same 11,545-parameter TCN is evaluated with three input configurations: VI, VI+TF, and VI+W. The VI control receives voltage and current while two fixed zero channels preserve the same input projection size. VI+TF uses the decoupled temperature/force coordinates, whereas VI+W uses the directly measured dual-FBG wavelengths. Every configuration is evaluated over six held-out profiles and five random seeds in both transfer directions.

**Table 5. Five-seed parameter-matched input ablation under strict cross-rate plus unseen-profile transfer.**

| Direction | Input | MAE (% SOC) | RMSE (% SOC) | R² | Q95-AE (% SOC) |
|---|---|---:|---:|---:|---:|
| 1C→2C | VI | 2.162 | 3.824 | 0.977020 | 8.780 |
| 1C→2C | VI+TF | **1.770** | 3.051 | 0.984593 | **6.751** |
| 1C→2C | VI+W | 1.795 | **3.037** | **0.985689** | 6.804 |
| 2C→1C | VI | **0.464** | **0.578** | **0.999587** | **1.127** |
| 2C→1C | VI+TF | 0.493 | 0.605 | 0.999569 | 1.153 |
| 2C→1C | VI+W | 0.806 | 0.988 | 0.998480 | 1.830 |

For 1C→2C transfer, adding native W1/W2 reduces the electrical-only MAE from 2.162% to 1.795% SOC, corresponding to an approximately 17.0% relative reduction. VI+W wins 20 of the 30 matched seed×profile comparisons. Because the results are clustered by random initialization, the primary uncertainty summary is a seed-cluster bootstrap: its 95% confidence interval for the absolute MAE gain is approximately +0.004 to +0.907 percentage points. Thus, dual-FBG information provides measurable benefit in the difficult low-to-high-rate transfer without relying on a single favorable initialization.

The formal ablation also refines the interpretation of the optical representation. VI+TF obtains an MAE of 1.770% SOC and VI+W 1.795% SOC in 1C→2C transfer. Their seed/profile-level performance is closely matched and the seed-cluster comparison does not support a stable practical advantage for either coordinate system. Native W1/W2 remains the retained representation because it was selected in the pre-frozen development comparison, is directly measured, and avoids an additional coordinate inversion. T/F remains useful for physical interpretation and achieves essentially the same level of difficult-transfer performance in the formal multi-seed test.

The reverse 2C→1C direction shows a different pattern. Electrical-only VI achieves the lowest MAE of 0.464% SOC, followed by VI+TF at 0.493% and VI+W at 0.806%. This direction dependence indicates that the benefit of optical sensing is tied to the amount of source-domain electrical support rather than being a universal gain from adding more channels.

## 4.5. Electrical–optical complementarity under increasing distribution shift

The preceding five-seed ablation is the primary evidence for the contribution of FBG sensing. The following OOD analysis is a fixed development diagnostic used only to interpret why the optical contribution changes with transfer direction; it is not used to select or rescue the final model.

An electrical support envelope is defined from the 0.5th–99.5th percentile range of source-training current. For each test window, the proportion of current samples outside this envelope is defined as the electrical-OOD fraction. The score uses only source-current statistics and observed current; target SOC labels are not involved.

In the complete 1C→2C development diagnostic, the parameter-matched VI MAE is 2.151% SOC and the corresponding VI+W MAE is 1.632% SOC. More importantly, the optical contribution changes systematically with OOD severity, as summarized in Table 6 and Fig. 5.

**Table 6. Electrical–optical comparison under increasing electrical-OOD severity for 1C→2C transfer.**

| Electrical-OOD level | VI MAE (% SOC) | VI+W MAE (% SOC) | Relative optical gain |
|---|---:|---:|---:|
| ID | **0.735** | 0.873 | −18.75% |
| OOD 0–25% | 1.550 | **1.470** | +5.17% |
| OOD 25–50% | 2.772 | **2.356** | +15.00% |
| OOD 50–75% | 3.665 | **2.930** | +20.05% |
| OOD 75–100% | 5.529 | **2.846** | **+48.52%** |

Within the fully supported electrical ID region, adding FBG observations does not improve the electrical-only estimator. Once test windows begin to leave source current support, the optical benefit becomes positive and increases monotonically with OOD severity. In the most shifted windows, the relative MAE reduction reaches 48.52%. This diagnostic provides a data-level explanation for the formal ablation: FBG sensing is most useful when the source-trained electrical mapping must extrapolate toward unfamiliar excitation.

## 4.6. Five-seed cross-rate unseen-profile generalization

After the model and raw-wavelength input interface are frozen, RA-FBG-TCN is evaluated over all six held-out profiles in both transfer directions using five independent random seeds. The split-specific epoch counts were selected using source-only data before final target evaluation.

The aggregate results are shown in Table 7 and Fig. 6. For 1C→2C transfer, RA-FBG-TCN achieves an average MAE of 1.795% SOC, RMSE of 3.037% SOC, R² of 0.985689, and Q95-AE of 6.804% SOC. In the reverse 2C→1C direction, the corresponding values are 0.806%, 0.988%, 0.998480, and 1.830% SOC. Across both directions, the seed-cluster mean MAE is 1.301% SOC. A seed-cluster bootstrap gives a 95% confidence interval of 0.961–1.677% SOC for the overall MAE.

**Table 7. Five-seed RA-FBG-TCN performance under strict cross-rate plus unseen-profile transfer.**

| Direction | MAE (% SOC) | RMSE (% SOC) | R² | Q95-AE (% SOC) |
|---|---:|---:|---:|---:|
| 1C→2C | 1.795 | 3.037 | 0.985689 | 6.804 |
| 2C→1C | **0.806** | **0.988** | **0.998480** | **1.830** |
| Overall | **1.301** | — | — | — |

The five-seed values are deliberately reported in addition to the seed-42 architecture benchmark because the 1C→2C direction exhibits meaningful initialization sensitivity. The multi-seed result provides the more conservative estimate of final-model robustness, whereas Table 4 provides a controlled equal-budget architecture comparison under one common seed. The directional asymmetry remains consistent across analyses: 2C→1C is substantially easier than 1C→2C.

## 4.7. Wavelength-noise robustness and conformal uncertainty

### 4.7.1. Robustness to wavelength perturbation

Practical FBG interrogation is subject to finite measurement noise. To evaluate sensitivity to direct optical perturbation, independent zero-mean Gaussian noise with standard deviations of 0.5, 1, and 2 pm is added to W1 and W2 before applying the frozen training-derived normalization. The trained model and all source-domain preprocessing parameters remain unchanged.

As shown in Fig. 7(a,b), error increases smoothly with wavelength-noise level rather than exhibiting abrupt degradation. At 2 pm per channel, the MAE increases by only approximately 1.67% relative to the clean baseline for 1C→2C transfer and 4.57% for 2C→1C transfer. Within the tested perturbation range, directly using native wavelength coordinates does not introduce pronounced sensitivity to small optical measurement errors.

### 4.7.2. 95% residual conformal prediction interval

The frozen RA-FBG-TCN point estimator is further calibrated using residual split conformal prediction. Absolute residuals from an independent calibration split define the 95% conformal quantile, after which the same fixed interval rule is applied to the test data.

The resulting 95% nominal interval attains an empirical PICP of 95.04%, closely matching the target coverage. The mean prediction interval width is 2.075% SOC, the mean interval score is 0.02441, and the corresponding absolute-residual quantile is approximately 1.089% SOC. Figure 7(c) illustrates the point estimate and calibrated interval over a representative test segment. These empirical coverage results pertain to the blocked mixed-condition calibration/test regime; no formal 95% coverage guarantee is claimed here for arbitrary cross-rate or unseen-profile distribution shift. The result nevertheless shows that a simple post-hoc conformal layer can supplement the deterministic point estimator with an uncertainty interval whose empirical coverage is aligned with the nominal level, without requiring a second probabilistic neural network.

Taken together, the experiments establish a conventional battery-estimation evidence chain: the compact model achieves high interpolation accuracy, remains competitive against representative sequence backbones under compound operating-condition change, gains measurable benefit from FBG information in the difficult low-to-high-rate transfer, exhibits interpretable direction-dependent modality utility, tolerates pm-scale wavelength perturbation, and supports lightweight uncertainty reporting.

# 5. Discussion

## 5.1. Effectiveness of dual-FBG information and optical representation

The five-seed input ablation provides the clearest evidence for the role of the optical sensing pathway. In the difficult 1C→2C transfer, both VI+W and VI+TF reduce error relative to the parameter-matched electrical-only model. This supports the practical use of internal FBG information when the deployment condition extends beyond the excitation represented during training. The result is consistent with the physical origin of the measurements: the Bragg wavelengths respond to internal thermo-mechanical evolution associated with electrochemical and thermal processes rather than reproducing terminal voltage or current alone.

The representation results should be interpreted in two stages. During development, the matched representation screen favored native W1/W2 over T/F and therefore froze W1/W2 as the final interface before formal evaluation. In the subsequent five-seed strict ablation, however, W1/W2 and T/F are closely comparable for 1C→2C transfer. The physical decoupling is not invalid, and native wavelengths should not be claimed to universally outperform T/F. The engineering advantage of W1/W2 is instead that they preserve the directly measured optical quantities, require no additional inversion, and deliver comparable formal transfer performance after being selected under the pre-frozen development protocol.

This distinction also avoids treating W1/W2 and T/F as four independent modalities. They are alternative coordinates of the same two optical sensing degrees of freedom. A compact SOC estimator can use the directly measured wavelength pair while the decoupled coordinates remain valuable for thermo-mechanical interpretation.

## 5.2. Cross-condition generalization and transfer asymmetry

The strict experiments show a clear asymmetry between low-to-high-rate and high-to-low-rate transfer. Training at 2C exposes the model to a wider current range, so most 1C target states remain within or close to source electrical support. In this direction, electrical-only VI is already highly effective and additional optical channels do not improve the matched TCN. Conversely, 1C training does not cover the strongest 2C excitation, making 1C→2C a more demanding extrapolation problem. Under this condition, both optical representations improve the electrical-only baseline.

The electrical-OOD diagnostic provides an explanatory link between source support and modality contribution. Optical input is not beneficial in the fully supported ID region, becomes useful once windows contain out-of-support electrical states, and reaches a 48.52% relative MAE reduction in the most severely shifted bin. The analysis does not require the claim that optical sensing always improves SOC estimation. A more practical conclusion is that the additional internal sensing pathway becomes most valuable when conventional electrical observability is weakened by operating-condition change.

The backbone benchmark provides a complementary model-level view. RA-FBG-TCN is not the best model in every direction: GRU gives the lowest seed-42 MAE in the difficult 1C→2C comparison. Nevertheless, RA-FBG-TCN ranks first for 2C→1C and gives the lowest pooled MAE across all 12 strict splits while using only 11.5k parameters. This balance between compactness and cross-condition accuracy is more relevant to the present engineering objective than claiming universal superiority over every recurrent or attention-based baseline.

## 5.3. Robustness, uncertainty, and practical implications

The wavelength-noise experiment shows smooth rather than abrupt degradation under direct 0.5–2 pm perturbations. This result is important because the final model uses native wavelength measurements directly rather than relying on an explicit thermo-mechanical reconstruction. Within the tested range, the direct optical interface does not introduce strong sensitivity to small interrogation errors.

The residual conformal layer complements point accuracy with an empirical uncertainty interval. Under the blocked calibration/test regime, the 95% nominal interval attains 95.04% empirical coverage with a mean width of 2.075% SOC. The point model and uncertainty layer are deliberately decoupled: the online estimator remains a small causal network, while uncertainty is added through a fixed calibration quantile. No formal 95% coverage guarantee is claimed for arbitrary distribution shifts, so the conformal result should be interpreted as calibrated reliability evidence for the evaluated regime rather than as a universal OOD guarantee.

High-accuracy SOC estimation has already been demonstrated on the same sensing platform, including a reported RMSE of 0.635% SOC [5]. Conventional interpolation accuracy is not treated as the primary novelty here, and direct leaderboard comparison is avoided because the evaluation protocols are not identical. The present work instead extends the engineering evidence toward matched sensing ablation, compound rate/profile transfer, representative backbone comparison, sensing perturbation, and post-hoc uncertainty calibration.

## 5.4. Scope and future work

The primary quantitative dataset in this study contains one physical cell instrumented with a fixed dual-FBG sensing configuration. The conclusions apply most directly to operating-condition transfer in which sensor installation and calibration remain consistent while discharge rate and driving profile change. They should not be interpreted as evidence of universal cross-cell optical transfer.

Across different physical cells, FBG initial wavelength, bonding condition, strain-transfer efficiency, sensor position, and sensitivity can vary. Future work should focus on multi-cell validation, sensor-aware calibration, broader temperature conditions, long-term sensor aging, and lightweight adaptation across optical installations. These issues may ultimately be more important for practical deployment than further increasing neural-network complexity.

Overall, the results support dual-FBG sensing as a complementary internal observation pathway for challenging operating-condition transfer. Its value is direction and support dependent rather than universal, while the compact causal TCN, noise-robust optical interface, and conformal calibration provide a practical framework for further multi-cell validation.

# 6. Conclusion

This study developed and evaluated a compact dual-FBG-assisted electrical–optical framework for battery SOC estimation under changing operating conditions. The proposed RA-FBG-TCN combines voltage, current, and two directly measured Bragg-wavelength channels in a 64-sample causal sequence and contains approximately 11.5k trainable parameters. Native W1/W2 and thermo-mechanically decoupled T/F were treated as alternative representations of the same optical sensing information rather than independent modalities.

Under blocked interpolation, RA-FBG-TCN achieved an MAE of 0.482% SOC, an RMSE of 0.593% SOC, and an R² of 0.999614. In the strict seed-42 backbone benchmark, the model achieved a pooled MAE of 0.864% SOC across 12 cross-rate/unseen-profile splits, ranking first in 2C→1C and third in the more difficult 1C→2C direction. The five-seed parameter-matched ablation showed that adding W1/W2 reduced the 1C→2C electrical-only MAE from 2.162% to 1.795% SOC, while W1/W2 and T/F exhibited closely comparable difficult-transfer accuracy. In the reverse 2C→1C direction, electrical-only sensing was sufficient and gave the lowest matched-model error.

The OOD analysis explains this directional behavior: the relative optical benefit increases as test current leaves source support and reaches 48.52% in the most shifted region. Final five-seed RA-FBG-TCN evaluation gives MAEs of 1.795% SOC for 1C→2C and 0.806% SOC for 2C→1C, with an overall seed-cluster MAE of 1.301% SOC and a bootstrap 95% confidence interval of 0.961–1.677% SOC. Direct wavelength perturbations up to 2 pm cause limited degradation, and a 95% residual conformal interval achieves 95.04% empirical coverage with a mean width of 2.075% SOC under the evaluated blocked calibration/test regime.

These results indicate that the practical value of dual-FBG sensing lies in providing an additional internal-state pathway for demanding operating-condition changes rather than universally reducing error in every regime. Future work will extend the framework to multiple cells, sensor-position variability, wider temperatures, aging, and cross-installation calibration.