# Q2 external E1 cross-cell FBG SOC protocol

Status: **PRE-REGISTERED EXTERNAL CONFIRMATION; NOT ELIGIBLE FOR SiC ARCHITECTURE SELECTION**

External source: Hebenbrock et al., Zenodo `10.5281/zenodo.15388590`, companion to Electrochimica Acta (2025), DOI `10.1016/j.electacta.2025.146975`.

## Structural facts fixed before modeling

- four pouch cells: A1, A2, P1, P2;
- A1/A2 are aged adhesive sets; P1/P2 are pristine sets;
- validation constant-current rates: 0.2C, 0.5C, 1C;
- the publication describes the cycler/interrogator acquisition as 1 Hz, while the released merged validation CSV has a denser merged timestamp grid (median contiguous interval about 0.566 s); E1 therefore preserves the official merged grid and does not resample it;
- central FBG position S5 exists for all four cells in the constant-current validation;
- positive current means discharge;
- validation cycles start from a full charge following the cell datasheet charge procedure;
- SOC in the source publication is determined from the discharge capacity of the initial referenceless strain calibration;
- the initial referenceless strain calibration consists of three 0.04C cycles at 25 C with one-minute rests between charge/discharge changes.

## Frozen SOC label reconstruction

The corrected reconstruction is frozen in `config/q2_external_soc_reconstruction_freeze.json` and the exact 52 eligible validation discharge boundaries are frozen in `config/external_e1_frozen_segments.csv`.

For each physical cell independently:

1. use exactly the three long initial discharge segments at approximately 0.4 A (0.04C), each longer than 20 h and above 8 Ah;
2. integrate positive discharge current using the released timestamps;
3. set `Q_ref` to the mean of those three discharge capacities;
4. admit validation discharges only at approximately 2 A, 5 A or 10 A (0.2C, 0.5C, 1C); extra-rate cycles are excluded rather than mapped to the nearest nominal rate;
5. for each frozen validation discharge starting fully charged, use `SOC(t) = 1 - cumulative_discharge_Ah(t) / Q_ref`;
6. do **not** rescale each validation discharge to end at zero. Higher-rate validation discharge capacity may be smaller than the 0.04C reference, so its valid terminal SOC may remain above zero.

Frozen `Q_ref` values (Ah): A1 10.285313, A2 10.270406, P1 10.884552, P2 10.771930. The corrected audit retained 52 discharges and all retained S5 traces have complete optical coverage.

## Sensor and optical-coordinate rule

Use exactly one optical sensor position per cell: the central fixed sensor **S5**.

This rule was declared before external modeling because the publication reports spatially nonuniform pouch-surface strain and uses the central S5 position as the principal fixed-sensor example. No post-hoc sensor-position search is allowed.

For cross-cell modeling, the primary optical predictor is the **causal raw Bragg shift relative to the start of the current full-charge discharge segment**:

`S5_rel(t) = lambda_S5(t) - lambda_S5(t_segment_start)`.

Different physical FBGs carry manufacturing/installation-dependent absolute Bragg offsets. Cross-cell use of absolute `lambda_S5` would therefore mix sensor identity/zero-point with battery state. The relative coordinate removes only that additive sensor baseline while preserving the measured, non-decoupled Bragg response. It uses the first observation of the current discharge only and no future sample or SOC label.

This is **not** a thermal/strain T/F decoupling. It remains a direct raw optical response coordinate.

Absolute S5 wavelength may be reported later only as a calibration-sensitive ablation; it is not the primary E1 optical coordinate and cannot replace `S5_rel` based on held-out-cell accuracy.

## E1 outer split

Four leave-one-cell-out folds:

- test A1; train A2/P1/P2
- test A2; train A1/P1/P2
- test P1; train A1/A2/P2
- test P2; train A1/A2/P1

All frozen 0.2C, 0.5C and 1C discharge cycles from the three source cells are available for training. All frozen discharge cycles from the held-out cell are test-only.

Normalization statistics are fit on the three source cells only.

## Matched models

No architecture search is conducted on the external test cells.

1. `VI-TCN`: causal TCN template with inputs Voltage and Current.
2. `VI-S5rel-TCN`: the same causal TCN template with Voltage, Current and `S5_rel`.

Both use:

- hidden width 24;
- the same three residual TCN blocks with dilations 1/2/4;
- 64-sample causal window on the official merged timestamp grid;
- train stride 4;
- test stride 1;
- MSE training loss;
- fixed 20-epoch equal budget;
- seed 42.

No cell ID, adhesive age label, C-rate label, absolute timestamp, Pt100 temperature, cumulative current, capacity, or reconstructed SOC history is a predictor.

## Pre-registered external evidence gate

The primary aggregation unit is the **held-out physical cell**; pooled-window metrics are secondary because cells and C-rates contain unequal numbers of samples.

`VI-S5rel-TCN` supports a robust cross-cell FBG benefit only if all five conditions hold:

1. mean held-out-cell MAE is lower than `VI-TCN`;
2. MAE improves on at least 3 of 4 held-out cells;
3. mean held-out-cell RMSE is no worse than `VI-TCN`;
4. mean held-out-cell Q95-AE is no worse than `VI-TCN`;
5. the worst single-cell relative MAE increase is at most 10%.

Per-C-rate and per-segment results are descriptive and cannot rescue a failed external evidence gate. External results cannot reopen the SiC representation, architecture, feature, epoch or UQ choices.

## Reporting

Report for each model:

- overall held-out-cell mean MAE, RMSE, R2, Q95-AE, MaxAE;
- each held-out cell;
- each C-rate within each held-out cell;
- number of physical discharge segments and windows;
- aged versus pristine cells descriptively;
- absolute S5 baseline/range and `S5_rel` range descriptively;
- source-only normalization provenance;
- secondary pooled-window metrics.

Primary external evidence question:

> Does adding one pre-declared, zero-point-aligned raw surface-FBG response improve SOC estimation on an unseen physical pouch cell under the same external-data training protocol?

The external experiment tests the **sensing/representation principle and architecture template**, not zero-shot transfer of SiC model weights. External target results cannot reopen SiC model selection.
