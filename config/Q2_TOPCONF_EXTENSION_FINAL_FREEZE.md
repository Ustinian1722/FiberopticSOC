# Q2 top-conference-inspired architecture extension — final freeze

Status: **FROZEN — EXPLORATION CLOSED AFTER USER-DIRECTED IDEA-TRANSFER STUDY**

The purpose of this extension was not to search model names until one beat the compact TCN. It tested whether recent top time-series conference inductive biases could transfer to causal dual-FBG SOC estimation under a leakage-safe source-only protocol.

All architecture-selection screens below used only 1C same-rate leave-one-profile-out validation over HWFET/LA92/NEDC/NYCC/US06/WLTC, seed 42, fixed 20 epochs, window 64, and no 2C metric for the decision.

## Baseline

`IUW-TCN` — V/I/raw W1/W2, 11,545 parameters.

Reference source-only performance:

- MAE: ~`0.005718`
- RMSE: ~`0.007155`
- Q95-AE: ~`0.013761`

## Stage A — generic ModernTCN/non-stationarity transfer

Candidates:

- `MC-TCN`: large-kernel depthwise causal temporal mixing with separated feature/variable mixing;
- `LR-MC-TCN`: local residual normalization plus explicit level path.

Run `33357144571`.

Results:

- MC-TCN MAE `0.009033`, 0/6 wins -> DROP
- LR-MC-TCN MAE `0.012872`, 0/6 wins -> DROP

Conclusion: generic forecasting-oriented large-kernel replacement and local normalization do not transfer directly. Local normalization is especially harmful because SOC relies on absolute operating level as well as dynamics.

## Stage B — fixed modality-specific delay

Candidate `MD-ResTCN`:

- exact strong joint TCN base path;
- short V/I correction RF ~13;
- long W1/W2 correction RF ~57;
- zero-initialized residual correction.

Run `33357531167`.

Results:

- MD MAE `0.005816` vs IUW `0.005718`
- MD wins 3/6
- mean delta MAE `+9.762e-05`
- median delta MAE negative
- RMSE/Q95 slightly worse -> DROP

The profile pattern was heterogeneous: HWFET/LA92/NEDC improved, NYCC/US06/WLTC degraded. This motivated testing whether the useful optical temporal scale is condition dependent rather than globally fixed.

## Stage C — dynamic multi-delay selection

Candidate `DMD-ResTCN`:

- exact strong joint TCN base path;
- fast electrical RF ~13;
- optical RFs ~13/~29/~57;
- per-window softmax selector over the three optical temporal scales;
- zero-initialized residual correction.

Run `33357906774`, artifact `q2-dynamic-multi-delay-tcn-source-summary`.

Aggregate:

| Model | Params | MAE | RMSE | Q95-AE | MaxAE |
|---|---:|---:|---:|---:|---:|
| **IUW-TCN** | 11,545 | **0.00571817** | **0.00715486** | **0.01376095** | **0.02778027** |
| DMD-ResTCN | 25,229 | 0.00572477 | 0.00718016 | 0.01402141 | 0.02852578 |

Pre-registered gate:

- mean delta MAE `+6.60e-06` -> fail
- median delta MAE `-7.62e-05` -> pass
- MAE wins `3/6` -> fail
- lower mean RMSE -> fail
- Q95 no worse -> fail

Decision: **DROP DMD-ResTCN**.

### Selector diagnostic

The dynamic selector did **not** collapse:

- short RF13: mean weight ~`0.416`
- medium RF29: ~`0.278`
- long RF57: ~`0.306`

Profile means vary materially. For example, US06 assigns roughly 0.499 mean weight to the short optical scale, while LA92 is much closer to an even three-scale allocation.

Therefore the data contain learnable condition-dependent temporal-scale structure, but exploiting it with this residual selector does not yield a stable aggregate SOC error improvement.

## Final conclusion

The user-requested top-conference idea-transfer study is complete.

Evidence supports the following interpretation:

1. recent time-series ideas about variable relations, non-stationarity and multi-delay are relevant diagnostic lenses;
2. they should not be assumed to improve a compact SOC estimator simply because they improve forecasting benchmarks;
3. the multi-delay experiments show genuine profile-dependent temporal-scale behavior, but that behavior is not the dominant remaining source of SOC error;
4. the compact raw-W IUW-TCN remains the strongest robust estimator under the source-only selection gate.

No further architecture candidate, lag grid, normalization variant, Transformer/Mamba family, hidden-size search or target-guided rescue is permitted in this extension.

The advanced candidates remain valuable paper ablations: they demonstrate that increasing architectural sophistication and explicitly modeling non-stationarity/delay do not automatically improve compound-domain robustness.

Next priority: external four-cell FBG validation, where the scientific question is stronger than another network search.