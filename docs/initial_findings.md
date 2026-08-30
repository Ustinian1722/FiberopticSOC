# Initial findings: Multi-condition Battery In-situ Sensing Data

Source dataset: **Multi-condition Battery In-situ Sensing Data**, Mendeley Data, DOI `10.17632/ft6rtwt8vm.1`.

This note records the first reproducible audit produced by GitHub Actions run #8 (`33325063009`). Raw dataset bytes are intentionally not committed to Git; the workflow downloads them from the public Mendeley dataset page at runtime and uploads compact analysis outputs as an Actions artifact.

## 1. Dataset structure

The downloaded archive is approximately **5.40 MB** (`5,398,170` bytes) and contains one data package, `SiC-18.zip` / `SiC-18`, with **12 Excel tables** and **68,086 total samples**.

The 12 tables form a clean **6 driving-profile × 2 discharge-rate** design:

| Profile | 1C rows | 2C rows |
|---|---:|---:|
| NYCC | 9,994 | 5,779 |
| NEDC | 8,861 | 5,432 |
| LA92 | 7,907 | 4,477 |
| WLTC | 7,539 | 4,562 |
| US06 | 4,720 | 3,060 |
| HWFET | 3,325 | 2,430 |

All 12 tables contain the same nine channels with no missing canonical channel and no duplicated rows detected in the first audit:

`time, current, voltage, wavelength_1, wavelength_2, temperature, force, dis_cap, SOC`.

## 2. Leakage findings

Two channels must be excluded from the primary SOC estimator.

### `dis_cap`

`SOC` is constructed from discharge capacity. In every table, `dis_cap` and SOC have essentially perfect inverse rank correlation (Spearman ≈ `-1.0`). Using `dis_cap` as an input would therefore amount to target leakage rather than a legitimate SOC-estimation feature.

### `time`

The pooled correlation between time and SOC looks only moderate because different profiles/rates have different durations. This is misleading. **Within every individual table**, time and SOC are almost perfectly monotonic:

- Spearman range: approximately `-1.0000` to `-0.9997`
- median: approximately `-0.99997`

Therefore random row splitting or including absolute elapsed time would make the benchmark severely optimistic. The main benchmark should exclude absolute time and split by complete condition/run, not by rows.

## 3. Optical / thermo-mechanical association with SOC

Within-condition Spearman association with SOC is much more informative than a pooled coefficient.

| Feature | Within-condition Spearman range | Median | Initial interpretation |
|---|---:|---:|---|
| force | `0.902` to `0.993` | `0.989` | Very strong, highly consistent SOC-sensitive mechanical observable |
| voltage | `0.970` to `0.998` | `0.987` | Expected strong electrical SOC coordinate |
| wavelength_2 | `0.924` to `0.984` | `0.961` | Strong and consistent raw FBG SOC response |
| temperature | `-0.969` to `-0.885` | `-0.947` | Strong monotonic thermo-mechanical response |
| wavelength_1 | `-0.008` to `0.656` | `0.361` | Much more condition dependent; weak alone in some regimes |
| current | `-0.677` to `-0.077` | `-0.176` | Primarily excitation/input rather than a direct SOC coordinate |

The pooled Spearman coefficients from all 68,086 samples are:

- force: `0.9869`
- voltage: `0.9840`
- wavelength_2: `0.9431`
- temperature: `-0.9390`
- wavelength_1: `0.4010`
- current: `-0.1785`

These numbers are descriptive only; they are **not** model performance and must not be interpreted as evidence that a random-sample SOC estimator will generalize to unseen conditions.

## 4. Immediate research implications

The first audit strongly supports studying FBG sensing as an auxiliary SOC modality, but the paper story should be framed around **generalization and physical sensing value**, not simply attaining a low random-split error.

A clean first ablation family is:

- E0: `V + I`
- E1: `V + I + temperature`
- E2: `V + I + force`
- E3: `V + I + wavelength_1 + wavelength_2`
- E4: `V + I + temperature + force`
- E5: raw-FBG (`wavelength_1/2`) versus physics-decoupled FBG (`temperature/force`) under exactly the same split

Do **not** combine raw wavelengths and their derived temperature/force channels and describe them as four independent sensor modalities. Temperature and force are derived from the FBG wavelength signals.

The most informative next benchmark should use whole-condition holdouts, for example leave-one-profile-out and/or cross-rate (1C→2C, 2C→1C) testing, followed by sensor drift/offset, noise, and dropout robustness tests. This will test whether FBG adds information beyond voltage/current rather than exploiting time/capacity construction shortcuts.
