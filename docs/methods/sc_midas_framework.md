# The SC-MIDAS framework

SC-MIDAS (Staggered-Combination MIDAS) is a two-layer nowcasting recipe
that turns a heterogeneous panel of monthly and quarterly indicators
into a single best-estimate forecast for the quarterly target $y_t$
(typically GDP growth) at several horizons.

## The three building blocks

### 1. Indicator models

Each monthly indicator $x^{(i)}$ is paired with the target through a
direct-forecast MIDAS regression:

$$
y_{t+h} \;=\; \alpha^{(i)}_h \;+\; \beta^{(i)}_h
   \sum_{j=0}^{K_i-1} w_h(j;\theta^{(i)}_h)\, x^{(i)}_{t,j}
   \;+\; \mathbf{d}_{t+h}^{\!\top}\gamma^{(i)}_h
   \;+\; \varepsilon_{t+h}.
$$

Optional outlier dummies $\mathbf{d}_{t+h}$ enter at the **target**
frequency.  Estimation uses OLS for linear-in-parameter weighting
schemes (Almon, U-MIDAS) and Levenberg-Marquardt non-linear least
squares with closed-form analytic Jacobians for the rest (exponential
Almon, Beta).

Each quarterly hard indicator $z^{(k)}$ uses a plain OLS counterpart:

$$
y_{t+h} \;=\; a^{(k)}_h \;+\; \mathbf{z}^{(k)\,\top}_t c^{(k)}_h
   \;+\; \mathbf{d}_{t+h}^{\!\top}\gamma^{(k)}_h \;+\; \varepsilon_{t+h}.
$$

In both cases, the pipeline stores one model per horizon $h$ in
`MidasCombo.midas_models_[var][h]` and `MidasCombo.ols_models_[var][h]`.

### 2. Layer 1 — soft combination

The indicator fits feed into one or more first-layer
[`ComboSpec`](../api.md#nowcast_midas.specs.ComboSpec) nodes whose
combination weights are derived from rolling-window residuals.  Three
inverse-error variants are available (mean absolute error, mean squared
error, root mean squared error) plus equal-weight average.

For inverse-MSE, the weight of source $m$ at time $t$ is the normalised
inverse of its discounted squared-error statistic $S^{(m)}_t$ — defined once
in [Weighting schemes](combo.md#mae-mse-rmse) and evaluated here with $p = 2$.
Complete common rows are selected before the window is applied, so a finite
window contains exactly $W$ = `window` comparable observations once it is
full. Error weighted combinations use available complete rows during warm-up;
regression combinations receive equal weight $1/n$ until `minimum_sample_size`
complete rows exist.

Before this row-level calculation, sources with fewer than
`minimum_sample_size` finite fitted observations are removed. The default
is `10`; the remaining sources are combined using whichever are available
on each date, with weights renormalised over those sources.

### 3. Layer 2 — soft × hard merging

The second layer pools the Layer-1 combo with a quarterly hard
regressor via constrained regression
([`fit_regression_weights`](../../api.md#nowcast_midas.combo_weights.fit_regression_weights)) with
`method='constrained_ls'`:

$$
\min_{\mathbf{w}\,\ge 0,\;\mathbf{1}^{\!\top}\mathbf{w}=1}
   \sum_t \Bigl( y_t - \mathbf{w}^{\!\top}\hat{\mathbf y}_t \Bigr)^2.
$$

`ComboSpec(method='regression', window=None)` uses an expanding window
over the full sample, which is the natural Layer-2 set-up.  When there
are exactly two sources (soft combo + hard) this reduces to the
EViews-style convex combination $w \in [0, 1]$.

## End-to-end recipe

```python
from nowcast_midas import MidasCombo, MidasSpec, OLSSpec, ComboSpec

midas_monthly_1 = MidasSpec("monthly_1", method="almon", n_lags=5)
midas_monthly_2 = MidasSpec("monthly_2", method="almon", n_lags=5)
midas_monthly_3 = MidasSpec("monthly_3", method="unrestricted", n_lags=3)
ols_quarterly_1 = OLSSpec("quarterly_1", n_lags=1)

soft = ComboSpec(
    "soft",
    sources=[midas_monthly_1, midas_monthly_2, midas_monthly_3],
    method="mse",
    window=8,
    discount_rate=0.95,
)
final = ComboSpec(
    "final", sources=[soft, ols_quarterly_1], method="regression", window=None
)

model = MidasCombo(combo_specs=final, horizons=3)
model.fit(target=target_df, regressors=regressors_df)
forecasts = model.forecast()
```

## Relation to the EViews reference

Single-vintage numerical equivalence is a reference-validation target.
The remaining gap is the *recursive* OOS estimation loop (re-fitting at
every historical vintage), which is a workflow-level feature outside
the scope of the estimation API.
