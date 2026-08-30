# Same-rate leave-one-profile-out findings

## Protocol

Twelve same-rate leave-one-profile-out (LOPO) splits were evaluated: six held-out profiles at 1C and six at 2C. For each split, five profiles at the same rate were used for training and the sixth profile was used only for testing.

All models used a 64-sample causal history window, train-only normalization, no future-aware statistics, and no `Time_s`, `dis_cap`, or target-derived inputs. The parameter-matched VI control used `[V, I, 0, 0]`; the optical model used `[V, I, W1, W2]`.

The hard selector used only the training-current 0.5%-99.5% envelope: use VI when the current window is fully inside the envelope, otherwise use VI+W. Test SOC was never used by the selector.

## Aggregate result

Across all 12 LOPO splits (seed 42, six epochs):

- VI: MAE 0.009291
- OOD-selective VI-or-W: MAE 0.009692
- VI+TF: MAE 0.011992
- VI+W: MAE 0.013412
- VI+TF-white: MAE 0.014390

At 1C, VI was best on average (MAE 0.008460) and the hard selector was worse (0.009036). The selector failed to beat VI in all six 1C splits.

At 2C, VI was again best on average (MAE 0.010122) and the hard selector was 0.010348. The selector beat or tied VI in only three of six splits.

## Why this matters

Same-rate profile shift is not the same regime as cross-rate electrical extrapolation. The fraction of windows containing any current value outside the five-profile training envelope averaged only about 7.6% across the 12 LOPO tests, compared with 52.6% in the earlier 1C->2C cross-rate test.

Using the LOPO window-level predictions, the association between current-envelope OOD fraction and per-window raw-FBG gain (`AE_VI - AE_VI+W`) essentially vanished:

- all same-rate LOPO windows: Pearson about -0.042, Spearman about -0.034;
- OOD windows only: raw FBG was worse on average, with mean gain about -0.0058 SOC fraction.

Therefore, the earlier result must **not** be generalized into a universal rule that any electrical OOD should activate optical sensing.

## Boundary condition established

The evidence currently supports a narrower claim:

> Raw FBG assistance is especially valuable under strong cross-rate electrical-support collapse, not under ordinary same-rate unseen-profile variation.

This is a useful falsification result. It suggests that the method should be framed as protection against genuine electrical extrapolation rather than as generic adaptive multimodal fusion.

## Next falsification test

Use a combined cross-rate + unseen-profile protocol:

- train five 1C profiles and test the held-out profile at 2C;
- train five 2C profiles and test the held-out profile at 1C;
- repeat for all six profiles in both directions.

This removes the concern that the current cross-rate result benefited from seeing the same driving-profile family during training and provides a substantially stronger generalization test.
