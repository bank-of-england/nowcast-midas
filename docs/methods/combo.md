# Weighting Schemes

SC-MIDAS uses two kinds of weights: temporal weights within each MIDAS
regression and combination weights across fitted sources. This page defines
both kinds.

Each non-leaf `ComboSpec` node produces a time-varying weight vector over its
sources that sums to one. For inverse-error weighting the source weight is the
normalised inverse of a discounted error statistic, defined once under
[`'mae'`, `'mse'`, `'rmse'`](#mae-mse-rmse) below and referenced from the
[SC-MIDAS framework](sc_midas_framework.md) page.

## Temporal weights

A MIDAS model is

$$
y_t \;=\; \alpha \;+\; \beta \sum_{j=0}^{K-1} w(j;\theta)\, x_{t,j}
   \;+\; \varepsilon_t
$$

with $K$ = `n_lags`.  The weights are normalised to sum to one for the
restricted schemes; `unrestricted` lifts that constraint.

| `method`        | $w(j;\theta)$                                  | parameters                | estimator |
|-----------------|------------------------------------------------|---------------------------|-----------|
| `almon`         | polynomial $\sum_i \theta_i j^i$               | `n_pars_weights` coeffs   | OLS       |
| `exp_almon`     | normalised $\exp(\sum_i \theta_i j^{i+1})$     | `n_pars_weights` shape    | NLS       |
| `beta`          | Beta density on $(0,1)$ grid                   | $2$ (shape $a, b$)        | NLS       |
| `unrestricted`  | one coefficient per lag (U-MIDAS)               | `n_lags`                  | OLS       |

OLS-estimable schemes (`almon`, `unrestricted`) are linear in the
parameters and solved via `numpy.linalg.lstsq`.  Non-linear schemes use
`scipy.optimize.least_squares(method='lm')` with analytic Jacobians
computed in closed form.

The Almon basis (Vandermonde):

$$
V = \begin{bmatrix}
j^0 & j^1 & \dots & j^{p-1}
\end{bmatrix}_{j=0}^{K-1},
\qquad
\hat\theta = (V^{\!\top}X^{\!\top}XV)^{-1}V^{\!\top}X^{\!\top}y,
\qquad
\hat w = V\hat\theta.
$$

This matches the EViews `polynomial=p` parameterisation exactly.

## Combination weights

Choose a `ComboSpec.method`:

| Method | Weighting | Typical use |
|---|---|---|
| `average` | Equal weights | Baseline combination |
| `mae` | Inverse mean absolute error | Less emphasis on large errors than MSE |
| `mse` | Inverse mean squared error | Pooling monthly indicators |
| `rmse` | Inverse root mean squared error | Less concentrated weights than MSE |
| `regression` | Joint least-squares fit | Combining a pooled indicator with quarterly data |

Weights are non-negative and sum to one over available sources:

$$
\hat y_t = \sum_{m=1}^{n} w^{(m)}_t\, \hat y^{(m)}_t.
$$

Error and regression weights are estimated separately for each horizon.
Fitted values use weights based on earlier observations; forecasts use weights
estimated through the final observation, subject to the chosen window.

### `'average'`

Each available source receives weight $1/n$, where $n$ is the number of
available sources.

### `'mae'`, `'mse'`, `'rmse'`

Sources with smaller past errors receive more weight. Each source uses its
own prior residuals, excluding missing targets, missing fitted values, and
`dummy_periods`.

- `window=W`: use the latest $W$ usable residuals per source.
- `window=None`: use all usable prior residuals.
- `discount_rate`: set to `1` for equal treatment of past residuals, or between
  `0` and `1` to give older residuals less influence.

For selected dates $s_1 < \dots < s_{N_m}$ and discount
$\delta =$ `discount_rate`, the error statistic is:

$$
S^{(m)}_t \;=\; \frac{1}{N_m} \sum_{j=1}^{N_m}
   \delta^{\,N_m-j}\, \bigl|\,y_{s_j} - \hat y^{(m)}_{s_j}\,\bigr|^{p}.
$$

Use $p=1$ for `mae` and $p=2$ for `mse` or `rmse`. Weights are proportional
to $1/S$ for `mae` and `mse`, or $1/\sqrt{S}$ for `rmse`.

With `window=W`, an available source with fewer than $W$ usable prior
residuals receives $1/n$. Sources with at least $W$ residuals divide the
remaining weight in proportion to inverse error.

For example, with `window=8` and three available sources, a source with only
five residuals receives $1/3$. The other two share $2/3$ according to their
errors, provided each has at least eight residuals.

Use a finite window when sources start at different dates: expanding error
windows can produce missing combinations when a source has no prior residuals.

### `'regression'`

Choose weights jointly to minimise the combined squared error:

$$
\hat{\mathbf w}_t \;=\; \arg\min_{\mathbf w \,\ge\, 0,\; \mathbf{1}^{\!\top}\mathbf w = 1}
   \sum_{s \in C_t}\bigl( y_s - \mathbf w^{\!\top} \hat{\mathbf y}_s\bigr)^2.
$$

The sample $C_t$ contains prior rows with a finite target and fitted values
for every retained source, excluding `dummy_periods`. Set `window=W` for the
latest $W$ common rows, or `window=None` for all common rows.

- `estimator="constrained_ls"` (default): solve with non-negative weights that
   sum to one. Set `minimum_sample_size` and any finite `window` at least as
   large as the number of retained sources.
- `estimator="clipped_ols"`: fit OLS, clip weights to $[0,1]$, then normalise.

## Minimum sample size (`minimum_sample_size`)

On `MidasSpec`, `OLSSpec`, and `MultiMidasSpec`, this sets the required number
of fitted quarterly observations. The default, `None`, adds no threshold.

On `ComboSpec`, the default is `10`. Sources with fewer finite fitted values
are excluded. Regression combinations also use equal weights until the prior
common history reaches this count, before applying `window`.
