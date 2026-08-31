# Q2 external FBG validation protocol

Status: **PRE-REGISTERED ROLE; NOT ELIGIBLE FOR ARCHITECTURE SELECTION**

External source: Hebenbrock et al., Zenodo DOI `10.5281/zenodo.15388590`, raw-data companion to *Referenceless surface FBG sensors: Combined thermal and mechanical monitoring of pouch cells*, Electrochimica Acta (2025), DOI `10.1016/j.electacta.2025.146975`.

## Why this dataset is used

The primary SiC-18 dataset contains one physical cell across six dynamic profiles and two discharge rates. It supports profile/rate domain-shift evaluation but cannot establish cell-to-cell generalization. The external dataset is therefore used only after the proposed architecture and optical representation are frozen, to test whether the learned **modeling principle** transfers to a different FBG deployment and multiple physical cells.

This is not intended as a zero-shot numerical transfer claim between incompatible sensor systems.

## External dataset differences

The Zenodo release contains:

- four pouch-cell/adhesive sets: aged A1/A2 and pristine P1/P2;
- raw cycling-unit voltage/current;
- multiple surface-mounted FBG Bragg-wavelength channels;
- Pt100 cell temperatures;
- constant-current validation at 0.2C, 0.5C, and 1C;
- repeated WLTP validation with maximum C-rate up to 3C; WLTP FBG data are available for P1/P2;
- separate temperature and SOC/strain calibration experiments.

These differences make the dataset suitable for architecture-level external validation, but they prohibit pretending that the SiC-18 dual implanted FBG channels and the external surface-FBG channels have identical physical meaning or absolute calibration.

The external publication further reports that pouch-surface strain is spatially inhomogeneous and that transferability between different sensor positions should not be assumed. For its exemplary fixed-FBG analysis the authors therefore use the **central S5 sensor** on each cell, which showed the highest overall sensitivity to surface temperature/strain changes. This gives the present study a defensible, non-target-selected common optical channel rule.

## Integrity rules

1. External data cannot be inspected to choose between IUW-TCN, Mamba, CrossFormer, or any other architecture.
2. The architecture, raw-W versus T/F decision, windowing philosophy, and main loss must be frozen on SiC-18 first.
3. Discharge capacity may be used only to reconstruct/verify SOC labels according to the external publication; capacity cannot be a predictor.
4. Absolute elapsed time is not a predictor.
5. Any normalization is fit using the external training cells/rates only; held-out external cells/rates cannot contribute normalization statistics.
6. No claim of zero-shot transfer is made unless the actual experiment is zero-shot and sensor-channel dimensionality/calibration are made compatible without target labels.
7. Because the external sensor count differs from SiC-18, the primary external experiment is **architecture transfer**: retrain the frozen architectural template on external-source cells and evaluate on held-out physical cells/conditions.
8. **Optical-channel rule is fixed before model fitting:** use the central fixed FBG `S5` for each cell wherever a valid S5 series exists. Do not search other FBG positions for lower target error. If structure audit shows a missing/unusable S5 for a cell/task, exclude that cell/task from the matched S5 experiment or invoke a separately documented source-independent fallback; do not select a substitute from held-out performance.

## Planned external tasks

### E1 — Cross-cell constant-current validation

Use constant-current data because all four cells provide comparable raw electrical/FBG recordings. Construct leave-one-cell-out or grouped cell holdout experiments using **S5 as the common optical measurement location** whenever available.

Primary question: does adding raw S5 optical response improve held-out-cell SOC estimation relative to the electrical-only counterpart under a different FBG installation system?

The preferred E1 experiment uses the same frozen architectural template for both inputs:

- electrical-only: U/I;
- electrical+optical: U/I + raw S5 Bragg-wavelength response.

Because the external system provides one common fixed optical channel rather than the implanted SiC-18 dual-FBG pair, any required input projection is treated as a dimensional interface adaptation, not as a new architectural search.

### E2 — Cross-rate external validation

Within the external dataset, train on lower/mid C-rates and evaluate a held-out rate, with cell grouping enforced and S5 fixed as above. Exact train/test rate directions will be frozen only after a structure audit confirms complete comparable cycles; the decision must be based on data availability, not target accuracy.

### E3 — Dynamic WLTP validation (secondary)

P1/P2 provide repeated WLTP FBG data and include S5. Use this only as a secondary dynamic-condition test after E1/E2. Because only two cells have WLTP optical data, claims must be limited accordingly.

## Representation policy

The primary external optical input is raw S5 Bragg-wavelength response. The external paper's thermal/strain compensation is physically valid for its own measurement goal, but the present external test asks whether direct optical observations can assist SOC prediction under a different sensing deployment. Any compensated strain/temperature view may be included as a physics-derived comparator only if it can be reconstructed from source-side calibration without target leakage.

The S5 rule deliberately avoids pooling multiple sensor positions because the source publication reports spatially heterogeneous pouch-film strain and limited transferability between positions. A learned multi-sensor attention pool would introduce a second modeling problem and would weaken the intended external-validation interpretation.

## Minimum reporting

For every eligible external task report:

- MAE, RMSE, R2, Q95 absolute error, MaxAE;
- per-cell results;
- electrical-only versus electrical+optical matched ablation;
- train/test cell and rate identities;
- number of windows/cycles;
- exact optical channel rule (`S5` or documented exclusion/fallback);
- whether SOC was supplied or reconstructed and how;
- no mixing of external results into SiC-18 architecture selection.

## Paper role

If successful, this section supports the claim that the proposed representation/fusion principle is not tied to the single SiC-18 cell or implanted dual-FBG installation. If unsuccessful, it is reported as a limitation; the primary SiC-18 result is not retuned using external target labels.