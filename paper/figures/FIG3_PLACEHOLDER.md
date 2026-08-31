# Fig. 3 methodology framework — PLACEHOLDER

Do not produce final artwork yet.

## Locked content flow

`V / I / W1 / W2`
→ train-only normalization
→ 64-sample causal window
→ 1×1 projection (`4 → 24`)
→ residual TCN block (`d=1`, `k=3`)
→ residual TCN block (`d=2`, `k=3`)
→ residual TCN block (`d=4`, `k=3`)
→ final causal state
→ regression head
→ SOC point estimate
→ 95% residual split-conformal interval.

### TCN-block inset

`causal Conv1D(k=3) → GroupNorm → GELU → causal Conv1D(k=3) → GroupNorm → residual add → GELU`

## Layout intent

- horizontal left-to-right flow compatible with a 183-mm two-column figure;
- compact architecture, not a decorative neural-network poster;
- one small inset for the residual TCN block;
- a narrow post-hoc UQ block after the point estimator;
- no claims of extra sensing degrees of freedom from T/F because the final estimator uses raw W1/W2 only.

Final framework drawing is deferred until data figures are stable.
