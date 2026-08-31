# Main-figure captions — English V1

## Fig. 1

**Fig. 1. Synchronized electrical and optical observations during dynamic discharge of the SiOx/C lithium-ion cell.** (a) Placeholder for the battery test platform and implanted dual-FBG sensing configuration; the final schematic will be inserted after the experimental-layout artwork is prepared. (b) Current and reference SOC during the representative NEDC 1C profile. (c) Simultaneously measured W1 and W2 Bragg-wavelength responses. (d) Comparison of the W2–SOC response for NEDC at 1C and 2C. All quantitative panels are generated directly from the released SiC-18 trajectories. The figure illustrates that terminal electrical excitation and internal dual-FBG observations are synchronized but exhibit distinct dynamic characteristics.

## Fig. 2

**Fig. 2. Statistical geometry and predictive-transfer performance of native and thermo-mechanically decoupled dual-FBG representations.** (a) Joint distribution of W1 and W2 over all 12 dynamic trajectories. (b) Joint distribution of the corresponding decoupled temperature- and force-related variables. (c) Absolute Pearson correlation of the two coordinates at 1C and 2C. The native W1/W2 correlations are 0.738 and 0.655, whereas the T/F correlations are 0.982 and 0.985. (d) Mean SOC MAE under the matched 1C→2C unseen-profile development protocol for the native-W TCN, the decoupled-T/F TCN, and the more complex ETMF-TF model. Although T/F provides clearer physical semantics, the native wavelength coordinates yield more stable predictive transfer in the retained causal estimator.

## Fig. 3

**Fig. 3. Overall RA-FBG-TCN framework and residual split-conformal uncertainty calibration.** Placeholder for the final method schematic. The final artwork will show the V/I/W1/W2 inputs, training-only normalization, 64-sample causal window, 4→24 input projection, residual TCN blocks with dilations 1/2/4, SOC regression head, and the 95% residual conformal interval constructed from an independent calibration split. An inset will summarize the two k=3 causal Conv1D layers, GroupNorm, GELU activation, and residual connection used in each TCN block.

## Fig. 4

**Fig. 4. Conventional SOC estimation performance of RA-FBG-TCN under blocked mixed-condition interpolation.** (a,b) Reference and estimated SOC for representative NEDC and NYCC test segments. (c) Absolute SOC error over a representative dynamic segment. (d) Absolute-error distribution over all test windows. RA-FBG-TCN achieves an overall MAE of 0.482% SOC, RMSE of 0.593% SOC, and R² of 0.999614. This experiment characterizes the basic point-estimation performance of the compact causal estimator and is not interpreted as evidence that optical input must improve every in-distribution sample.

## Fig. 5

**Fig. 5. Relationship between electrical distribution-shift severity and the benefit of dual-FBG optical information.** (a) Source-rate electrical support defined from the 0.5th–99.5th percentiles of training current, with a representative 1C→2C test trajectory highlighting regions outside source support. (b) MAE of parameter-matched VI and VI+W estimators across window-level electrical-OOD bins. (c) Relative optical gain obtained by adding W1/W2. The gain is −18.75% in the fully supported ID region and increases to +5.17%, +15.00%, +20.05%, and +48.52% as OOD severity increases. The result shows that the principal value of the FBG modality emerges when conventional electrical observations leave their training support.

## Fig. 6

**Fig. 6. Five-seed generalization performance under the cross-rate unseen-profile protocol.** (a) Profile-wise MAE for 1C→2C transfer; bars and error bars denote the mean and standard deviation over five random seeds. (b) Corresponding results for 2C→1C transfer. (c) Seed-cluster bootstrap 95% confidence intervals for the two transfer directions and the overall MAE. The aggregate MAEs are 1.795% SOC for 1C→2C and 0.806% SOC for 2C→1C; the overall seed-cluster MAE is 1.301% SOC with a bootstrap 95% confidence interval of 0.961–1.677% SOC. The results reveal a clear asymmetry between low-to-high-rate and high-to-low-rate transfer.

## Fig. 7

**Fig. 7. Robustness to direct FBG wavelength perturbation and calibrated SOC uncertainty.** (a) MAE after independently adding zero-mean Gaussian noise with standard deviation 0, 0.5, 1, or 2 pm to W1 and W2. (b) Corresponding Q95 absolute error. (c) Reference SOC, RA-FBG-TCN point estimate, and 95% residual split-conformal prediction interval over a representative blocked-interpolation test segment. Performance degrades smoothly under pm-scale wavelength perturbation. The 95% nominal conformal interval achieves 95.04% empirical coverage with a mean prediction interval width of 2.075% SOC.