# The SC-MIDAS pipeline

[`MidasCombo`](../api.md#nowcast_midas.midas_combo.MidasCombo) is the
high-level orchestrator: it fits one
[`MIDAS`](../api.md#nowcast_midas.midas.MIDAS) per monthly indicator, one
[`OLS`](../api.md#nowcast_midas.ols.OLS) per quarterly regressor, and pools
their forecasts through a tree of
[`ComboSpec`](../api.md#nowcast_midas.specs.ComboSpec) nodes — across all
horizons in one call.

## Input data

Both `target` and `regressors` are **long-format** DataFrames:

| column      | type        | example       | notes                          |
|-------------|-------------|---------------|--------------------------------|
| `date`      | `datetime`  | `2024-06-30`  | end-of-period date             |
| `variable`  | `str`       | `"quarterly_target"` | one frame can hold many series |
| `frequency` | `str`       | `"QE"` / `"ME"` | quarterly / monthly end      |
| `value`     | `float`     | `1.23`        | the observation                |

* `target` must contain a single quarterly variable (`frequency='QE'`),
  no NaNs in `value`.
* `regressors` may mix monthly (`'ME'`, modelled by `MidasSpec`) and
  quarterly (`'QE'`, modelled by `OLSSpec`); the `frequency` column
  selects the path per variable and each variable must carry a single
  frequency. Trailing NaNs (the ragged edge) in `value` are allowed.
  Name collisions between MIDAS and OLS variables raise `ValueError`.
* `frequency` values are matched case-insensitively (`"qe"` == `"QE"`).

The examples on this page use the built-in simulator so every block runs
as-is. Later blocks continue from the names defined here:

```python
from nowcast_midas import ComboSpec, MidasCombo, MidasSpec, OLSSpec
from nowcast_midas.utils import sample_combo_data

# long-format target + regressors (monthly_1..3 at "ME", quarterly_1 at "QE")
target, regressors, info = sample_combo_data(n_quarters=60, seed=42)
outlier = info["outlier_date"]  # injected -25 shock at 2020-06-30
```

## Specifications

### `MidasSpec` — one per monthly indicator

```python
from nowcast_midas import MidasSpec

MidasSpec(
    variable="monthly_1",
    method="almon",  # 'almon', 'exp_almon', 'beta', 'unrestricted'
    n_lags=5,  # monthly lags X[t-n_lags+1 .. t]
    n_pars_weights=2,  # shape parameters of the weight scheme
    estimator=None,  # 'ols' / 'nls' (auto)
    start_lag=0,  # shift the lag window
    n_ar_lags=0,  # AR lags of y
    dummy_periods=None,  # list[str | pd.Timestamp] outlier quarters
    minimum_sample_size=None,  # minimum fitted observations required before this spec contributes
)
```

All `MidasSpec` arguments map one-for-one to the corresponding
[`MIDAS`](midas_model.md) constructor argument.

### `OLSSpec` — one per quarterly regressor

```python
from nowcast_midas import OLSSpec

OLSSpec(
    variable="quarterly_1",
    n_lags=1,
    start_lag=0,
    n_ar_lags=0,
    dummy_periods=None,
    minimum_sample_size=None,  # minimum fitted observations required before this spec contributes
)
```

Quarterly regressors share the target's frequency, so no temporal
weighting is needed:

$$
y_{t+h} \;=\; \alpha_h \;+\; \mathbf{x}_t^{\!\top}\beta_h
   \;+\; \mathbf{d}_{t+h}^{\!\top}\gamma_h
   \;+\; \sum_{k=1}^{p} \phi_{h,k}\, y_{t+h-k}
   \;+\; \varepsilon_{t+h}.
$$

### `ComboSpec` — the combination tree

```python
from nowcast_midas import ComboSpec

midas_monthly_1 = MidasSpec(
    "monthly_1", method="almon", n_lags=6, dummy_periods=[outlier]
)
midas_monthly_2 = MidasSpec(
    "monthly_2", method="almon", n_lags=6, dummy_periods=[outlier]
)
midas_monthly_3 = MidasSpec(
    "monthly_3", method="unrestricted", n_lags=3, dummy_periods=[outlier]
)
ols_quarterly_1 = OLSSpec("quarterly_1", n_lags=1, dummy_periods=[outlier])

soft = ComboSpec(
    name="soft",
    sources=[midas_monthly_1, midas_monthly_2, midas_monthly_3],
    method="mse",  # 'average' | 'mae' | 'mse' | 'rmse' | 'regression'
    window=8,  # rolling window (None = expanding)
    discount_rate=0.95,  # exponential discount on past errors (error-weighted methods)
    minimum_sample_size=10,  # remove sources with fewer fitted observations
)

final = ComboSpec(
    name="final",
    sources=[soft, ols_quarterly_1],  # nested combo + quarterly regressor
    method="regression",
    window=None,
    estimator="constrained_ls",  # 'constrained_ls' (default) or 'clipped_ols'
)
```

Sources may be `MidasSpec` / `OLSSpec` / `MultiMidasSpec` instances or
nested `ComboSpec` nodes. Plain string names can also be used, but only
when that name matches a variable or combo that already appears as an
object elsewhere in the tree — the pipeline resolves names by walking
the spec tree; there is no separate registration step.

The pipeline **flattens** the tree into dependency order (leaves first)
and **collects** indicator specs automatically, so you never need to
list them separately.

Combination methods are detailed on the
[weighting schemes page](../methods/combo.md).

## Fitting

```python
from nowcast_midas import MidasCombo

model = MidasCombo(
    combo_specs=final,
    horizons=3,  # produces 1qa, 2qa, 3qa models
)
model.fit(target=target, regressors=regressors)
```

`horizons` is a **count**, not an index. `horizons=3` fits a separate
model for $h \in \{0,1,2\}$ — i.e. the 1qa, 2qa, 3qa direct
forecasts.

Internally `fit()` runs three stages:

1. **MIDAS stage** — one `MIDAS` instance per `MidasSpec` is fitted on
   all horizons at once. Insufficient-data variables emit a
   `RuntimeWarning` and produce all-NaN fitted values, but do not
   abort the pipeline.
2. **OLS stage** — same for each `OLSSpec`.
3. **Combo stage** — every `ComboSpec` node, at every horizon,
   computes its time-varying weights from in-sample residuals of its
   sources and stores the combined fitted vector.

### Inspecting the fit

| attribute               | type / shape                           | contents                                      |
|-------------------------|----------------------------------------|-----------------------------------------------|
| `fitted_`               | `dict[name][h] -> pd.Series (T,)`      | in-sample fitted values per spec and horizon  |
| `midas_models_`         | `dict[var][h] -> FittedMidas`          | per-horizon MIDAS fit objects                 |
| `ols_models_`           | `dict[var][h] -> FittedOLS`            | per-horizon OLS fit objects                   |
| `midas_instances_`      | `dict[var] -> MIDAS`                   | the underlying fitted MIDAS estimators        |
| `ols_instances_`        | `dict[var] -> OLS`                     | the underlying fitted OLS estimators          |
| `multi_midas_instances_`| `dict[name] -> MultiMIDAS`             | the underlying fitted MultiMIDAS estimators   |
| `combo_weights_`        | `dict[name][h] -> dict[src, ndarray]`  | time-varying combination weights              |
| `fits_df_`              | `DataFrame` — `spec, date, horizon, value` | long-format in-sample fitted values       |
| `weights_df_`           | `DataFrame` — `spec, source, horizon, value` | long-format combination weights         |

```python
model.summary(horizon=0)  # prints and returns the text
```

## Out-of-sample forecast

```python
forecasts = model.forecast()
```

`forecasts` is a **long-format** DataFrame with columns
`date`, `horizon`, `spec`, `value` — one row per spec per step:

```
         date  horizon     spec  value
    2025-06-30        0  monthly_1   1.23
   2025-06-30        0     soft   1.19
   2025-06-30        0    final   1.31
    2025-09-30        1  monthly_1   1.05
   2025-09-30        1     soft   1.11
   2025-09-30        1    final   1.28
```

* `date` — forecast target date. `horizon=0` → $y_{T+1}$,
  `horizon=1` → $y_{T+2}$, etc.
* `spec` — indicator or combo name.
* Step $s$ uses the horizon-$s$ model. This alignment needs no future
    regressor values, so the pipeline fills every step by construction. A
    spec still returns `NaN` when its inputs are missing at the ragged edge.

After `forecast()`, the full in-sample + OOS series is available as
`model.fits_and_forecasts_df_` — a long DataFrame with the same
`date`, `horizon`, `spec`, `value` columns covering both the training
window and the OOS extension. The OOS block alone is kept in
`model.forecasts_df_`.

### Forecast decomposition

```python
dec = model.forecast_decomp()  # root combo, per sub-component
dec = model.forecast_decomp("soft", aggregate=True)  # one row per indicator model
```

`forecast_decomp(spec_name=None, regressors=None, aggregate=False)` splits
each horizon's combined forecast into additive components that sum back to
the value returned by `forecast()`. Pass a different `regressors` frame to
compute counterfactual decompositions (e.g. an older model evaluated on a
newer vintage). See
[Interpreting decompositions](decomposition.md) for the full column
contract.

## Plotting

After fitting, `MidasCombo` provides three plotting methods:

```python
# Actual vs fitted for one or more combos / individual indicators
model.plot_fit(
    combo_names="soft",
    indicator_names=["monthly_1", "monthly_2"],
    horizon=0,
)

# Time-varying combination weights per source
model.plot_weights(combo_name="soft", horizon=0)

# In-sample fit + OOS forecast (call forecast() first)
model.plot_forecast(combo_names="final", horizon=0)
```

Common arguments:

| argument                        | default | behaviour |
|---------------------------------|---------|-----------|
| `combo_names` / `combo_name`    | `None`  | defaults to the root `ComboSpec` name |
| `indicator_names`               | `None`  | overlay individual MIDAS / OLS / MultiMIDAS sources (`plot_fit`, `plot_forecast`) |
| `horizon`                       | `None`  | plots all horizons stacked vertically; pass an `int` for a single panel |
| `ax`                            | `None`  | provide an existing `Axes` when `horizon` is a single int |
| `ylim` / `xlim`                 | `None`  | fix the axis bounds on every subplot (`plot_fit`, `plot_forecast`) |

All three return `(fig, ax)` so you can customise or save the figure.

## End-to-end example

```python
from nowcast_midas import ComboSpec, MidasCombo, MidasSpec, OLSSpec
from nowcast_midas.utils import sample_combo_data

target, regressors, info = sample_combo_data(n_quarters=60, seed=42)
outlier = info["outlier_date"]

# sample_combo_data() injects a -25 outlier by default; without a dummy the
# fit degrades badly (RMSE ~3.5 vs ~1.1), so pass it to every spec.
midas_monthly_1 = MidasSpec(
    "monthly_1", method="almon", n_lags=6, dummy_periods=[outlier]
)
midas_monthly_2 = MidasSpec(
    "monthly_2", method="almon", n_lags=6, dummy_periods=[outlier]
)
ols_quarterly_1 = OLSSpec("quarterly_1", n_lags=1, dummy_periods=[outlier])

soft = ComboSpec(
    "soft",
    sources=[midas_monthly_1, midas_monthly_2],
    method="mse",
    window=8,
    discount_rate=0.95,
)
final = ComboSpec(
    "final", sources=[soft, ols_quarterly_1], method="regression", window=None
)

model = MidasCombo(combo_specs=final, horizons=3).fit(target, regressors)
forecasts = model.forecast()
print(forecasts.head())
```

A runnable end-to-end version is in [Worked examples](../examples/core_models.md).
