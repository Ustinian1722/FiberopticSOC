# Q2 external four-cell FBG data diagnostics

Status: **DESCRIPTIVE ONLY — NOT USED TO ALTER THE FROZEN E1 MODEL OR GATE**

Source: compact E1 dataset produced by formal external run `33359316457` from the 52 frozen constant-current validation discharges. Labels and boundaries are fixed by `config/q2_external_soc_reconstruction_freeze.json` and `config/external_e1_frozen_segments.csv`.

## Dataset scale and timing

- physical cells: A1, A2, P1, P2
- retained discharge segments: 52
- valid model samples: 781,654
- segment counts: A1 14, A2 12, P1 14, P2 12
- C-rate counts: 0.2C 13, 0.5C 16, 1C 23
- median contiguous interval on the released merged grid: about 0.566 s
- S5 completeness in every retained segment: 100%

The official merged time grid is preserved; there is no E1 resampling. SOC integration uses the released timestamps directly.

## Frozen reference capacities

| Cell | Q_ref (Ah) |
|---|---:|
| A1 | 10.285313 |
| A2 | 10.270406 |
| P1 | 10.884552 |
| P2 | 10.771930 |

Each value is the mean capacity of exactly three long approximately 0.4 A initial calibration discharges. Validation SOC is not forced to reach zero at the high-rate cutoff.

## Cross-cell optical zero point

Absolute S5 Bragg baselines differ materially across physical sensors. Approximate starting-wavelength ranges over the frozen segments are:

- A1: 845.901–846.012 nm
- A2: 845.775–845.954 nm
- P1: 846.132–846.163 nm
- P2: 846.480–846.500 nm

This justifies using the pre-registered causal segment-relative coordinate `S5_rel(t)=S5(t)-S5(start)` for the cross-cell main experiment instead of treating absolute Bragg wavelength as a universal SOC coordinate.

## S5_rel range and direction heterogeneity

Across the frozen segments, approximate S5_rel ranges are:

- A1: −0.066 to +0.084 nm
- A2: −0.125 to +0.161 nm
- P1: −0.128 to +0.030 nm
- P2: −0.105 to +0.060 nm

The optical trajectory is not sign-consistent across physical cells. A1/A2 frequently show positive wavelength excursions, whereas many P1/P2 segments are dominated by negative excursions. No target-aware sign flip is permitted in E1.

## SOC association is cell- and rate-dependent

For each physical discharge independently, Pearson and Spearman correlations were computed between `S5_rel` and the reconstructed SOC, then averaged within each cell/rate group. The approximate mean correlations are:

| Cell | Rate | Pearson | Spearman |
|---|---:|---:|---:|
| A1 | 0.2C | −0.079 | −0.071 |
| A1 | 0.5C | −0.762 | −0.711 |
| A1 | 1C | +0.209 | +0.239 |
| A2 | 0.2C | −0.889 | −0.875 |
| A2 | 0.5C | −0.140 | −0.141 |
| A2 | 1C | +0.171 | +0.167 |
| P1 | 0.2C | +0.897 | +0.945 |
| P1 | 0.5C | +0.849 | +0.757 |
| P1 | 1C | −0.137 | +0.005 |
| P2 | 0.2C | +0.878 | +0.836 |
| P2 | 0.5C | +0.760 | +0.647 |
| P2 | 1C | −0.636 | −0.488 |

The association can therefore change both magnitude and sign with cell and C-rate. A single global linear wavelength-to-SOC calibration is not supported by these external data.

## Scientific implication

The external E1 experiment is intentionally stronger than a same-cell calibration test. It asks whether a causal sequence model can exploit the shape/history of a zero-point-aligned raw FBG response when the simple static S5–SOC association itself is non-universal across cells and rates.

Two interpretations are pre-specified:

- if the frozen E1 gate passes, the result supports transferable information in the dynamic FBG response beyond a cell-specific static wavelength mapping;
- if the gate fails, the result establishes an important boundary: zero-point alignment alone is insufficient for robust cross-cell optical SOC transfer, even though FBG may remain useful within a cell/domain.

Neither outcome may be used to reopen the already frozen SiC model selection.
