# Q2 external E1 cross-cell FBG SOC protocol

Status: **PRE-REGISTERED EXTERNAL CONFIRMATION; NOT ELIGIBLE FOR SiC ARCHITECTURE SELECTION**

External source: Hebenbrock et al., Zenodo `10.5281/zenodo.15388590`, companion to Electrochimica Acta (2025), DOI `10.1016/j.electacta.2025.146975`.

## Structural facts fixed before modeling

- four pouch cells: A1, A2, P1, P2;
- A1/A2 are aged adhesive sets; P1/P2 are pristine sets;
- validation constant-current rates: 0.2C, 0.5C, 1C;
- cycling-unit voltage/current and FBG Bragg wavelength are sampled at 1 Hz;
- central FBG position S5 exists for all four cells in the constant-current validation;
- positive current means discharge;
- validation cycles start from a full charge following the cell datasheet charge procedure;
- SOC in the source publication is determined from the discharge capacity of the initial referenceless strain calibration;
- the initial referenceless strain calibration consists of three 0.04C cycles at 25 C with one-minute rests between charge/discharge changes.

## SOC label reconstruction

For each physical cell independently:

1. identify the three full discharge segments in `Initial_referenceless_strain_calibration.csv`;
2. integrate positive discharge current over timestamp intervals for each segment;
3. set `Q_ref` to the mean of those three discharge capacities;
4. for each eligible constant-current validation discharge starting fully charged,
   `SOC(t) = 1 - cumulative_discharge_Ah(t) / Q_ref`;
5. do **not** rescale each validation discharge to end at zero. The publication explicitly notes that higher-rate validation discharge capacity is smaller than the 0.04C reference, so the valid SOC range becomes narrower at higher C-rate.

The reconstruction must pass a structure audit before training: three consistent initial reference discharges per cell, expected 0.2/0.5/1C validation current levels, lower-voltage termination near 3.0 V, and high S5 availability.

## Sensor and optical-coordinate rule

Use exactly one optical sensor position per cell: the central fixed sensor **S5**.

This rule was declared before external modeling because the publication reports spatially nonuniform pouch-surface strain and uses the central S5 position as the principal fixed-sensor example. No post-hoc sensor-position search is allowed.

For cross-cell modeling, the primary optical predictor is the **causal raw Bragg shift relative to the start of the current full-charge discharge segment**:

`S5_rel(t) = lambda_S5(t) - lambda_S5(t_segment_start)`.

Rationale: different physical FBGs carry manufacturing/installation-dependent absolute Bragg offsets. Cross-cell use of absolute `lambda_S5` would therefore mix sensor identity/zero-point with battery state. The relative coordinate removes only that sensor-specific additive baseline while preserving the measured, non-decoupled Bragg response. It uses the first observation of the current discharge only and no future sample or SOC label.

This is **not** a thermal/strain T/F decoupling. It remains a direct raw optical response coordinate.

Absolute S5 wavelength may be reported later only as a calibration-sensitive ablation; it is not the primary E1 optical coordinate and cannot replace `S5_rel` based on held-out-cell accuracy.

## E1 outer split

Four leave-one-cell-out folds:

- test A1; train A2/P1/P2
- test A2; train A1/P1/P2
- test P1; train A1/A2/P2
- test P2; train A1/A2/P1

All eligible 0.2C, 0.5C and 1C discharge cycles from the three source cells are available for training. All eligible discharge cycles from the held-out cell are test-only.

Normalization statistics are fit on the three source cells only.

## Matched models

No architecture search is conducted on the external test cells.

1. `VI-TCN`: causal TCN template with inputs Voltage and Current.
2. `VI-S5rel-TCN`: the same causal TCN template with Voltage, Current and `S5_rel`.

Both use:

- hidden width 24;
- the same three residual TCN blocks with dilations 1/2/4;
- 64-sample = 64-second causal window;
- train stride 4;
- test stride 1;
- MSE training loss;
- fixed 20-epoch equal budget for the first external comparison;
- seed 42 for the initial E1 screen.

No cell ID, adhesive age label, C-rate label, absolute timestamp, Pt100 temperature, cumulative current, capacity, or reconstructed SOC history is a predictor.

## Reporting

Report for each model:

- overall MAE, RMSE, R2, Q95-AE, MaxAE;
- per held-out cell;
- per C-rate within each held-out cell;
- number of physical discharge segments and windows;
- aged versus pristine cells descriptively;
- absolute S5 baseline/range and `S5_rel` range descriptively;
- source-only normalization provenance.

Primary external evidence question:

> Does adding one pre-declared, zero-point-aligned raw surface-FBG response improve SOC estimation on an unseen physical pouch cell under the same external-data training protocol?

The external experiment tests the **sensing/representation principle and architecture template**, not zero-shot transfer of SiC model weights. External target results cannot reopen SiC model selection.