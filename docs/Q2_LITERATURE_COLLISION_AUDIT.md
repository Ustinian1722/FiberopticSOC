# Q2 literature-collision audit for method positioning

Status: **POSITIONING CONSTRAINT — DO NOT CLAIM GENERIC TCN/TRANSFORMER/CROSSATTENTION NOVELTY**

## Closest architecture-level prior art

Recent Journal of Energy Storage work already includes:

1. A parallel TCN-Transformer multi-task SOC/SOE model in which TCN captures local/multi-scale information, Transformer captures global dependencies, and CrossAttention fuses the two model-branch feature spaces (2025, DOI `10.1016/j.est.2025.118077`).
2. A paper explicitly titled *A temporal convolutional network-Transformer-CrossAttention model for state of charge estimation of lithium-ion batteries* (DOI `10.1016/j.est.2025.119417`).
3. FBG-assisted SOC work using wavelength decoupling and CNN-Transformer with electrical plus internal strain/temperature inputs (2025, DOI `10.1016/j.est.2025.117969`).
4. The companion SiC-18 Energy paper already uses FBG-assisted electrical/thermomechanical inputs with feature engineering, noise augmentation, and CNN-GRU-Attention (2026, DOI `10.1016/j.energy.2026.140609`).

Therefore the present study must not claim any of the following as a standalone novelty:

- first TCN-Transformer SOC estimator;
- first CrossAttention SOC estimator;
- first CNN/TCN local + Transformer global hybrid;
- first FBG-assisted deep-learning SOC estimator;
- first wavelength-decoupled FBG SOC estimator;
- first use of attention for FBG/battery SOC;
- first multi-condition or noise-robust SOC analysis on SiC-18.

## Defensible differentiation if EO-CrossFormer is retained

The proposed `EO-CrossFormer` is organized around **sensor modality interaction**, not fusion of two generic neural-network branches:

- electrical modality: voltage/current causal local representation;
- optical modality: raw dual-FBG wavelength causal local representation;
- electrical -> optical and optical -> electrical causal cross-attention explicitly conditions each sensing modality on the observed history of the other;
- interaction features (`e`, `o`, `e*o`, `|e-o|`) are fused before global causal temporal modeling;
- the central evaluation target is simultaneous C-rate shift **and** unseen-drive-profile shift.

This is distinct from prior TCN-Transformer-CrossAttention papers whose CrossAttention primarily fuses local TCN features and global Transformer features extracted from the same conventional battery input space.

## Defensible differentiation independent of architecture outcome

The stronger paper-level contribution is the combination of:

1. **representation study:** raw optical W1/W2 versus an exactly invertible physics-decoupled T/F coordinate under matched architectures;
2. **compound-shift evaluation:** rate shift and unseen-profile shift occur simultaneously rather than being tested independently or through random/sample-level splitting;
3. **sensor-generalization evidence:** multi-seed and direct raw-wavelength measurement-noise tests, with a preregistered external multi-cell surface-FBG dataset reserved for architecture transfer;
4. **selection integrity:** architecture/representation decisions use the already-designated development direction, while the reverse rate direction is held for stronger confirmatory use after freeze.

These claims remain available even if the final advanced architecture fails to beat the compact IUW-TCN.

## Naming constraint

Avoid a paper title centered on `TCN-Transformer-CrossAttention` or `CrossFormer` alone. If an advanced architecture is retained, the title should emphasize concepts such as:

- representation-aware electrical-optical learning;
- raw optical sensing;
- compound operating-condition shift;
- cross-modal conditioning/fusion;
- robust SOC generalization.

The architecture name should be subordinate to the scientific question rather than presented as the sole novelty.