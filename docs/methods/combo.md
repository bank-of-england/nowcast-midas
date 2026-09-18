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

Each non-leaf `ComboSpec` produces a time-varying weight vector
$\mathbf{w}_t \in \Delta^{n-1}$ (the unit simplex) over its sources.
Combined fits are then

$$
\hat y_t = \sum_{m=1}^{n} w^{(m)}_t\, \hat y^{(m)}_t.
$$

The available `method` values are:

### `'average'`

Equal weight on every source that is non-NaN at $t$:

$$
w^{(m)}_t \;=\; \mathbb{1}\{\hat y^{(m)}_t \in \mathbb{R}\}
   \,\big/\, n_{\text{avail}}(t).
$$

### `'mae'`, `'mse'`, `'rmse'`

Inverse-error weighting with an exponential discount on past residuals. Rows
containing a missing value in any source are removed before the latest $W$
complete rows are selected; $C_t$ is that common sample. The source weight is
the normalised inverse of a discounted error statistic:

$$
S^{(m)}_t \;=\; \frac{1}{|C_t|} \sum_{s \in C_t}
   \delta^{\,t-s}\, \bigl|\,y_s - \hat y^{(m)}_s\,\bigr|^{p},
\qquad
w^{(m)}_t \;=\; \frac{1/S^{(m)}_t}{\sum_{m'} 1/S^{(m')}_t},
$$

with $p = 1$ for `mae`, $p = 2$ for `mse` / `rmse` (`rmse` takes $\sqrt{S}$
before inverting), discount $\delta$ = `discount_rate` $\in (0, 1]$, and
window $|C_t| \le W$ = `window` (equal to `window` once the window is full).
The most recent residual carries weight $\delta^{0} = 1$; older residuals
decay geometrically.

The function is called once for each direct forecasting horizon, after the
horizon-specific source fits have been selected. It returns in-sample weight
rows only. For each fitted value at t, the corresponding weight at t is
based on prior performance up to and including t-1. For true out-of-sample
projections, `forecast()` reuses the final in-sample weight row.

`window=None` uses an expanding window from the first observation,
which is the typical Layer-2 set-up. During warm-up, error weighted methods
use the available complete common rows. Regression methods use equal weights
until `minimum_sample_size` complete common rows exist.

### `'regression'`

Constrained least squares with non-negativity and sum-to-one
constraints:

$$
\hat{\mathbf w}_t \;=\; \arg\min_{\mathbf w \,\ge\, 0,\; \mathbf{1}^{\!\top}\mathbf w = 1}
   \sum_{s=t-W+1}^{t}\bigl( y_s - \mathbf w^{\!\top} \hat{\mathbf y}_s\bigr)^2.
$$

Solved by [`constrained_least_squares`](../api.md#nowcast_midas.combo_weights.constrained_least_squares)
(the `estimator="constrained_ls"` default): Levenberg-Marquardt on a
softmax reparameterization $\mathbf w = \mathrm{softmax}(\mathbf z)$ with
an analytical Jacobian, so the constraints hold by construction.  With
exactly two sources this reduces to a convex combination
$w_1 \in [0, 1]$, $w_2 = 1 - w_1$, matching the EViews soft-vs-hard
merge equation.

Passing `estimator="clipped_ols"` instead uses
[`clipped_ols`](../api.md#nowcast_midas.combo_weights.clipped_ols): plain OLS,
weights clipped to $[0, 1]$ and renormalized to sum to one. This method
avoids iterative optimisation, which makes it faster but less stable on
ill-conditioned designs.

The regression sample can be restricted with `estimation_start` /
`estimation_end`, and `window=None` triggers an expanding window covering
the full sample.

## Which to pick?

* **Layer 1 (soft pooling):** `'mse'` or `'rmse'` with a moderate
  rolling window (e.g. `window=8`, `discount_rate=0.95`).  This gives
  more weight to indicators with recently low forecast errors and
  reacts smoothly to regime changes.
* **Layer 2 (soft × hard):** `'regression'` with `window=None`.  The
  expanding window uses the whole sample to estimate one stable mixing
  weight between soft signal and hard data, matching the EViews
  convention.
* **Average:** baseline for diagnostics — useful when you want to
  isolate the gain from error weighting.

## Minimum sample size (`minimum_sample_size`)

`MidasSpec`, `OLSSpec`, and `MultiMidasSpec` accept a
`minimum_sample_size` argument (default `None`). When set, a source model is considered
insufficient for forecasting until it has that many **fitted quarterly
observations**. This prevents early, poorly estimated models from
entering a combination during the warm-up period.

`ComboSpec` uses the same argument, defaulting to `10`, to remove sources
with fewer than this many finite fitted observations before combination
rows are filtered. For regression combinations, it also prevents weight
estimation until the common sample reaches this size. The remaining sources
can still be missing on individual dates; those dates use the available
sources and renormalise their weights.
