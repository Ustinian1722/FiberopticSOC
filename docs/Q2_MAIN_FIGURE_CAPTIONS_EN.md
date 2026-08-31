# Main figure captions — English V1

## Fig. 1

**Fig. 1. Synchronized electrical and optical observations during dynamic discharge of the SiC-based lithium-ion cell.** (a) Placeholder for the battery test platform and implanted dual-FBG sensing configuration; the final artwork will be assembled separately. (b) Current and reference SOC during a representative NEDC 1C trajectory. (c) Synchronously recorded W1 and W2 Bragg-wavelength shifts. (d) W2–SOC response under NEDC at 1C and 2C. The quantitative panels are generated directly from the released SiC-18 trajectories without random subsampling. The figure illustrates that terminal electrical measurements and the two internal optical channels are synchronized but exhibit distinct dynamic evolution.

## Fig. 2

**Fig. 2. Representation characteristics and development-stage transfer comparison of native and thermo-mechanically decoupled dual-FBG coordinates.** (a) Joint distribution of W1 and W2 over all 12 dynamic trajectories. (b) Corresponding joint distribution of the decoupled temperature and force coordinates. (c) Absolute Pearson correlation for the two coordinate systems at 1C and 2C. Native W1/W2 gives |r|=0.738 and 0.655, whereas the decoupled T/F coordinates give |r|=0.982 and 0.985. (d) Average SOC MAE under the matched cross-rate development protocol for raw W, decoupled T/F, and the more complex ETMF-TF representation. This development screen motivated the pre-frozen choice of W1/W2 as the final model interface; the subsequent formal five-seed ablation shows that W1/W2 and T/F provide statistically comparable accuracy in the difficult 1C→2C transfer, so no universal representation superiority is claimed.

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