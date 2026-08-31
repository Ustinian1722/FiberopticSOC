# Q2 post-architecture freeze

Status: **FROZEN — ARCHITECTURE SEARCH CLOSED**

## Final estimator retained

- estimator: **IUW-TCN**
- inputs: Voltage, Current, raw Wavelength_1, raw Wavelength_2
- optical representation: **raw dual-FBG coordinates W1/W2**
- parameter count: **11,545**
- causal window modeling
- no absolute time, discharge capacity, SOC-derived state, or target-domain normalization

This is an evidence-based retention, not a claim that TCN is architecturally novel.

## Advanced architecture screens

### Mamba family

Development run `33348016670`: all Mamba-family candidates underperformed the compact IUW-TCN on the six 1C->2C unseen-profile development folds. The proposed scalar optical gate also collapsed near a constant. Decision: **DROP Mamba family**.

### CrossFormer family

Canonical development run `33355158650`:

- raw-W EO-CrossFormer: MAE `0.015190`, worse than IUW-TCN `0.013847`; failed all pre-registered retention criteria. **DROP**.
- EO-CrossFormer-TF: aggregate MAE `0.013546`, slightly lower than IUW-TCN, but only 3/6 profile MAE wins and higher cross-profile variability. It was not promoted from target-development evidence alone.

### Independent source-side confirmation of TF-CrossFormer

Run `33355975834`, artifact `q2-tf-crossformer-source-confirm-summary`, 1C-only same-rate LOPO, seed 42, fixed 20 epochs, no 2C metrics used for decision:

| Model | MAE mean | MAE std | RMSE mean | Q95-AE mean |
|---|---:|---:|---:|---:|
| **IUW-TCN** | **0.005718** | **0.001694** | **0.007155** | **0.013761** |
| EO-CrossFormer-TF | 0.007951 | 0.003857 | 0.009379 | 0.016562 |

Paired source-side gate:

- mean delta MAE (TF-IUW): `+0.002233` -> fail
- median delta MAE: `+0.000826` -> fail
- TF wins: `2/6` -> fail
- TF mean RMSE lower: false
- TF mean Q95 no worse: false

Decision: **DROP EO-CrossFormer-TF**.

## Representation conclusion

The study does **not** claim that physical T/F decoupling is invalid. The T/F coordinate remains physically interpretable and is exactly invertibly related to W1/W2 in this release. However, across the complete development evidence it is not sufficiently stable to replace raw W1/W2 for predictive SOC estimation:

- raw W is the clear winner for the compact TCN family;
- T/F can benefit a coordinate-sensitive CrossFormer on selected cross-rate profiles and gives a small aggregate target-development gain;
- that gain does not reproduce under independent 1C source-side profile transfer, where TF-CrossFormer is substantially worse overall.

Therefore the final predictive mainline uses **raw W1/W2**, while T/F is retained as a physics-decoupled representation ablation and a discussion point about physical interpretability versus predictive representation geometry.

## Research-integrity consequence

No further architecture family, hidden size, attention mechanism, Mamba variant, fusion gate, window length, or target-guided rescue search is permitted for the SiC-18 main model in this study.

The paper contribution is explicitly shifted away from generic network novelty toward:

1. rigorous raw-optical versus physics-decoupled representation evidence;
2. simultaneous C-rate + unseen-profile domain shift;
3. compact-model robustness across seeds and direct FBG wavelength noise;
4. uncertainty reporting with residual split conformal;
5. external multi-cell FBG architecture/principle validation on a separately published dataset.

## Next research stage

Proceed to external FBG data audit and validation using the pre-registered S5 rule in `docs/Q2_EXTERNAL_FBG_VALIDATION_PROTOCOL.md`. External results cannot reopen the SiC-18 architecture decision.