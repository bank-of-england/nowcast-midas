# The MIDAS model

This page covers the single-indicator
[`MIDAS`](../api.md#nowcast_midas.midas.MIDAS) class: how to specify it,
estimate it (OLS vs NLS), add AR lags and outlier dummies, fit multiple
horizons in one call, and produce out-of-sample forecasts.

## Model

The direct $h$-step MIDAS regression of a low-frequency target
$y_{t+h}$ on a single high-frequency indicator $x_{t,j}$
($j=0,\dots,K-1$) is

$$
y_{t+h} \;=\; \alpha_h \;+\; \beta_h \sum_{j=0}^{K-1} w_h(j;\theta_h)\, x_{t,j}
   \;+\; \mathbf{d}_{t+h}^{\!\top}\gamma_h
   \;+\; \sum_{k=1}^{p} \phi_{h,k}\, y_{t+h-k}
   \;+\; \varepsilon_{t+h}.
$$

* $K$ = `n_lags` high-frequency lags per low-frequency observation.
* $w_h(j;\theta_h)$ = weight scheme controlled by a small number of
  shape parameters $\theta_h$.
* $\mathbf{d}_{t+h}$ = optional outlier-dummy block.
* $p$ = `n_ar_lags` autoregressive lags of the target.
* All parameters are **horizon-specific** (one set per $h$).

## Input data

`fit` takes two **2-column** DataFrames — a quarterly `target` and a
monthly `regressors` frame, each with exactly the columns `date` and
`value`. The frequencies are inferred from the date spacing:

```python
# illustrative shape — build these from your own data
target = pd.DataFrame({"date": q_dates, "value": y})  # quarterly
regressors = pd.DataFrame({"date": m_dates, "value": x})  # monthly
```

The examples on this page use the built-in simulator so every block runs
as-is. Each later block continues from the names defined here:

```python
from nowcast_midas import MIDAS
from nowcast_midas.utils import sample_data

# date/value quarterly target + date/value monthly indicator
target, regressors = sample_data(n_obs=100, seed=42)
```

## Specifying the model

```python
from nowcast_midas import MIDAS

model = MIDAS(
    method="almon",  # 'almon', 'exp_almon', 'beta', 'unrestricted'
    n_lags=6,  # high-frequency lags per low-frequency obs
    n_pars_weights=2,  # shape parameters of the weight scheme
    estimator=None,  # 'ols' / 'nls' (auto-selected from method)
    horizons=[0, 1, 4],  # explicit list of horizons to fit (default None -> [0])
    start_lag=0,  # shift the lag window forward by k high-frequency periods
    n_ar_lags=0,  # autoregressive lags of y (0 = no AR term)
    dummy_periods=None,  # list[pd.Timestamp] of outlier quarters
)
```

!!! note "`horizons` here is a list, not a count"
    `MIDAS(horizons=...)` takes an explicit list of horizon indices to fit
    (`None` is treated as `[0]`, i.e. nowcast only). This is different from
    `MidasCombo(horizons=...)`, which takes an `int` — the *number* of
    application forecast steps to produce.

### Weighting schemes (`method`)

| method          | weight $w(j;\theta)$                                   | parameters        | typical estimator |
|-----------------|--------------------------------------------------------|-------------------|-------------------|
| `unrestricted`  | $\theta_j$                                             | $K$               | OLS               |
| `almon`         | $\sum_{i=0}^{P-1} \theta_i\, j^{i}$                    | `n_pars_weights`  | OLS (linear in $\theta$) |
| `exp_almon`     | $\exp\!\big(\theta_1 j + \theta_2 j^2\big) / Z$        | `n_pars_weights`  | NLS               |
| `beta`          | Beta-density profile on $[0,1]$                        | 2                 | NLS               |

`unrestricted` and `almon` are **linear in their parameters**, so the
model reduces to a closed-form least-squares problem (`estimator='ols'`).
`exp_almon` and `beta` are non-linear; the package estimates them by
`least_squares` with an analytical Jacobian (`estimator='nls'`).

See the [weighting schemes page](../methods/combo.md#temporal-weights)
for full definitions.

### Estimator (`estimator`)

* `None` (default) — `ols` for `almon`/`unrestricted`, `nls` otherwise.
* `'ols'` — only valid for `almon`/`unrestricted`; a `ValueError` is
  raised otherwise.
* `'nls'` — always available; uses Levenberg-Marquardt with an
  analytical Jacobian. This method is slower but required for the
  non-linear schemes.

### Autoregressive lags (`n_ar_lags`)

Setting `n_ar_lags=p` appends $p$ lags of the dependent variable
$y_{t+h-1},\dots,y_{t+h-p}$ to the regressor block. The coefficients
$\phi_{h,k}$ are estimated **jointly** with the MIDAS parameters
(closed-form OLS for `almon`/`unrestricted`, NLS otherwise) and
available as `fit.phi`. At forecast time the model automatically picks
up the most recent $p$ target values strictly before the latest available
regressor date $T_X$.

### Outlier dummies (`dummy_periods`)

`dummy_periods=[ts1, ts2, ...]` adds one binary column per listed
quarter to the design matrix. The estimated coefficients live in
`fit.gamma` and represent the additive shift on $y_{t+h}$ at those
quarters. Dummies travel with the **target** (so they are aligned with
$y_{t+h}$, not with $X_t$).

## Fitting and direct multi-horizon estimation

```python
model.fit(target, regressors)
```

For each $h$ in `horizons`, `fit()` independently estimates a model on
the direct alignment $y_{t+h} \sim X_t$. For non-negative horizons this
uses the pair $(y[h:],\, X[:T-h])$. Negative horizons instead build a
regressor window at the origin quarter `target quarter - h` for each
target date. Each fit is stored in `model.fits_`:

```python
model.fits_[0]  # 1qa model: y[t]   ~ X[t]
model.fits_[1]  # 2qa model: y[t+1] ~ X[t]
model.fits_[4]  # 5qa model: y[t+4] ~ X[t]
```

Negative horizons can be requested explicitly, for example
`MIDAS(horizons=[-1, 0, 1])`. With October as the latest finite monthly
observation, `h=-1` estimates Q3 using October's lag window. Historical Q3
targets likewise use October windows; historical Q2 targets use July
windows. Available windows beyond the end of the target sample can be used
for estimation, while unavailable future windows are excluded. Dummies
and negative-horizon AR terms align with the target, so AR predictors never
include that target itself during estimation.

The forecast date is the latest finite indicator quarter plus `h`.
An October-informed Q3 estimate therefore retains a Q3 output date.
Backtests must respect the October information date.

Each entry is a `FittedMidas` dataclass with:

| attribute        | shape           | meaning                              |
|------------------|-----------------|--------------------------------------|
| `alpha`          | scalar          | intercept $\alpha_h$                 |
| `beta`           | scalar          | scale $\beta_h$ (always $1$ in OLS, absorbed into `weights`) |
| `theta`          | $(\text{n\_pars\_weights},)$ | weight-shape parameters $\theta_h$ |
| `weights`        | `(n_lags,)`     | resolved weight vector $w_h(\cdot)$  |
| `gamma`          | `(n_dummies,)`  | dummy coefficients                   |
| `phi`            | `(n_ar_lags,)`  | AR coefficients                      |
| `fitted_values`  | `(T_h,)`        | $\hat y$ on the valid rows          |
| `residuals`      | `(T_h,)`        | $y - \hat y$                         |
| `nobs`           | int             | number of observations used          |
| `y`, `X`         | arrays          | the data fitted at this horizon      |

The model drops rows with missing required lags, including rows affected by
insufficient monthly data or AR warm-up. `model.valid_mask_` records the
target rows used for estimation.

## Forecasting

```python
fc = model.forecast(regressors)
```

`forecast()` produces a **long-format** DataFrame with columns
`date`, `horizon`, `spec`, `value` — one row per fitted horizon. `spec`
is the indicator name (the regressor `variable` value, or `"target"`
when the frame has no `variable` column):

```
        date  horizon    spec  value
0 2025-03-31        0  target   1.23
1 2025-06-30        1  target   1.18
2 2026-03-31        4  target   1.05
```

Mechanics:

* The **latest available regressor date** $T_X$ is the latest date in
  `regressors_forecast` with a finite value. This date supplies the most
  timely regressor observation; the model applies no additional cap.
* Direct forecasting fits $y_{t+h} \sim X_t$, so for each horizon $h$ the
  lag row is built at $T_X$ and the forecast target date is
  $T_X + h$ quarters (`latest_regressor_date + h * 3 months`).
* Equivalently, to forecast $y_{T+\text{step}}$ where $T$ is the last
  observed target, the horizon used is the gap between the forecast date and
  the latest regressor: $h = (T + \text{step}) - T_X$.
* If the model uses dummies, the dummy row is built at the
  **forecast target date** (consistent with the in-sample alignment).
* If the model uses AR lags, the $p$ most recent target values at
  dates strictly **before** the latest available regressor date are pulled from the training
  target and used as $y_{t-1}, \dots, y_{t-p}$.

To get a true OOS forecast at $h=0$ the regressors must extend
**beyond** the training period; for $h>0$ the model already embeds the
lead, so a forecast is OOS even with training-period regressors.

### Long-format outputs

`fit()` also stores the in-sample fitted values as
`model.fits_df_` (`date`, `horizon`, `value`), and `forecast()` stores its
return value as `model.forecasts_df_` (`date`, `horizon`, `spec`, `value`).

### Forecast decomposition

```python
dec = model.forecast_decomp(regressors, regressor_name="indicator")
```

Splits every horizon's point forecast into additive components
(`intercept`, the weighted MIDAS block, `dummy_i`, `ar_lagk`) that sum
back to the value returned by `forecast()`. Columns: `horizon`, `date`,
`component`, `contribution`, `weight`. See
[Interpreting decompositions](decomposition.md).

## Inspecting and plotting

```python
model.summary()  # text summary of every fitted horizon
model.summary(horizon=0)  # only h=0
model.plot_fit(horizon=0)  # in-sample fit
model.plot_weights(horizon=0)  # resolved weight vector w(j;θ)
```

## Synthetic data with a built-in DGP

`utils.sample_data(horizon=h, ...)` simulates a target that follows the
MIDAS DGP at horizon $h$:

$$
y_{t+h} \;=\; \alpha \;+\; \beta\, X_t\, w \;+\; \varepsilon_{t+h}.
$$

Fitting `MIDAS(..., horizons=[h])` on this data should recover the
generating parameters and produce essentially zero residuals when
`noise=0`. The package's exact-recovery tests rely on this.

```python
from nowcast_midas.utils import sample_data

target, regressors = sample_data(
    n_obs=200,
    n_lags=6,
    noise=1e-10,
    seed=0,
    horizon=4,
    method="almon",
)
model = MIDAS(method="almon", n_lags=6, horizons=[4]).fit(target, regressors)
```
