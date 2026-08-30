# NestedCal99: support-calibrated selective optical assistance

## Scope

NestedCal99 is designed for SOC estimation when the electrical operating condition may extrapolate beyond the source domain, while an implanted dual-FBG representation can provide a complementary expert. The method does **not** assume that optical sensing is universally superior and does not continuously fuse all modalities. Instead, it uses source-domain support statistics to decide when the optical-assisted expert is allowed to replace the electrical-only expert.

The current implementation uses two parameter-matched predictors:

- **Electrical expert**: `VI`, driven by voltage and current. Two fixed zero channels are appended internally so that its encoder has exactly the same input width and parameter count as the optical-assisted model.
- **Optical-assisted expert**: `VI+W-white`, driven by voltage, current, and the two raw FBG wavelength channels after train-only whitening.

`SOC`, `dis_cap`, and absolute `Time_s` are excluded from predictors.

## 1. Train-only FBG representation conditioning

Let the two FBG wavelengths at time t be w_t = [W1_t, W2_t]^T. Normalization statistics are estimated from the five source profiles only. After source-only normalization, a 2-D whitening transform A is estimated from the source FBG pair. The optical-assisted expert receives

z_t = A w_t

alongside normalized voltage and current. No target-profile samples are used to estimate A.

Whitening does not add information. Because the published temperature/force channels are an invertible linear transform of W1/W2 in this dataset, this stage is treated as representation conditioning rather than multimodal information gain.

## 2. Electrical support envelope

For a main split, concatenate source-profile current samples and define the robust source support interval

I_low = Q_0.005(I_source),
I_high = Q_0.995(I_source).

For a causal window W_t of length L=64 ending at time t, define the binary pointwise support violation

b_j = 1[I_j < I_low or I_j > I_high].

The window electrical support-shift score is

s_t = (1/L) sum_{j in W_t} b_j.

Thus s_t is the fraction of samples in the current causal window that fall outside the robust source-current envelope. It uses only measured current up to the present window endpoint.

## 3. Nested source-profile calibration

A fixed rule such as `s_t > 0` is overly sensitive to ordinary profile-to-profile variation. NestedCal99 therefore estimates a normal source-domain shift ceiling without consulting the final target profile.

Suppose the five source profiles are P1,...,P5. For each pseudo-held-out source profile Pk:

1. build the 0.5–99.5% current envelope from the other four source profiles;
2. compute s_t for every causal window of Pk;
3. pool the resulting internal source-profile shift scores over k=1,...,5.

The activation threshold is frozen as

τ_99 = Q_0.99({s_t from all nested source-profile folds}).

No SOC values are used to calculate τ_99.

## 4. Selective expert activation

At inference, compute s_t using the main source envelope. The prediction is

ŷ_t = ŷ_opt,t, if s_t > τ_99;
ŷ_t = ŷ_VI,t, otherwise.

where ŷ_opt,t is the prediction of `VI+W-white` and ŷ_VI,t is the electrical-only prediction.

The gate therefore has a conservative interpretation:

- within the source-calibrated range of ordinary profile shift, retain the electrical expert;
- when the electrical window exhibits shift more severe than 99% of source-only profile variation, permit the optical-assisted expert.

This is a **selective fallback architecture**, not a claim that electrical OOD guarantees optical superiority. Final evaluation explicitly measures negative-transfer cases and compares NestedCal99 with both fixed optical fusion and the more sensitive `AnyOOD` rule.

## 5. Leakage boundary

For every held-out `(rate, profile)` test domain, all of the following are fitted from the five source profiles only:

- channel normalization;
- FBG whitening;
- current support interval;
- nested calibration threshold τ_99;
- model training;
- training epoch, previously frozen by source-profile CV using development seed 42.

The target profile is used only for final forward evaluation. Target SOC is never used for normalization, whitening, gate calibration, model selection, or epoch selection.

## 6. Evaluation protocol

The hardest protocol is directional cross-rate plus unseen-profile extrapolation:

- `1C_to_2C`: train on five 1C drive profiles and test on the sixth profile at 2C;
- `2C_to_1C`: train on five 2C drive profiles and test on the sixth profile at 1C;
- repeat over all six drive profiles;
- final statistical seeds are 0–4, independent of development seed 42.

Primary comparisons are `VI`, `VI+W`, `VI+W-white`, `AnyOOD-VI-or-W-white`, and `NestedCal99-VI-or-W-white`. Statistical reporting is performed on seed×profile paired errors and includes seed-cluster bootstrap intervals so that overlapping windows are not treated as independent experimental replicates.
