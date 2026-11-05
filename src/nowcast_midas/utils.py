"""Utility functions for MIDAS models and example data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .midas import FittedMidas

__all__ = ["sample_combo_data", "sample_data"]


def _residual_std(fit: FittedMidas, estimator: str) -> float:
    """Residual standard deviation (df-corrected).

    Parameters
    ----------
    fit : FittedMidas
        Fitted result for a single horizon.
    estimator : str
        ``'ols'`` or ``'nls'``.

    Returns
    -------
    sigma : float
    """
    # NLS estimates alpha, beta, and theta; OLS fixes beta at 1.0.
    # Count the corresponding parameters for the degrees-of-freedom correction.
    n_free = len(fit.theta) + (2 if estimator == "nls" else 1)
    return float(np.sqrt((fit.residuals @ fit.residuals) / (fit.nobs - n_free)))


def _ols_fit_se(fit: FittedMidas, estimator: str) -> np.ndarray:
    """Analytical SE of in-sample fitted values (OLS only).

    Uses the hat matrix: SE_i = sigma * sqrt(h_ii).

    Parameters
    ----------
    fit : FittedMidas
        Fitted result for a single horizon (must have ``A`` populated).
    estimator : str
        ``'ols'`` or ``'nls'``.

    Returns
    -------
    se : np.ndarray
    """
    A = fit.A
    sigma = _residual_std(fit, estimator)
    ATA_inv = np.linalg.pinv(A.T @ A)
    H_diag = np.einsum("ij,jk,ik->i", A, ATA_inv, A)
    return sigma * np.sqrt(H_diag)


def _ols_weight_se(
    fit: FittedMidas,
    estimator: str,
    method: str,
    n_lags: int,
    n_pars_weights: int,
) -> np.ndarray:
    """Analytical SE of estimated lag weights (OLS only).

    For ``unrestricted``: SE comes directly from the diagonal of cov(phi).
    For ``almon``: SE is propagated through the polynomial basis via the
    delta method: cov(w) = V @ cov(theta) @ V^T.

    Parameters
    ----------
    fit : FittedMidas
        Fitted result for a single horizon (must have ``A`` populated).
    estimator : str
        ``'ols'`` or ``'nls'``.
    method : str
        Weighting scheme (``'unrestricted'`` or ``'almon'``).
    n_lags : int
        Number of high-frequency lags.
    n_pars_weights : int
        Number of weight-shape parameters.

    Returns
    -------
    se : np.ndarray
    """
    A = fit.A
    sigma = _residual_std(fit, estimator)
    cov_phi = sigma**2 * np.linalg.pinv(A.T @ A)

    if method == "unrestricted":
        return np.sqrt(np.diag(cov_phi)[1 : 1 + n_lags])

    j = np.arange(n_lags, dtype=float)
    V = np.column_stack([j**i for i in range(n_pars_weights)])
    cov_w = V @ cov_phi[1 : 1 + n_pars_weights, 1 : 1 + n_pars_weights] @ V.T
    return np.sqrt(np.diag(cov_w))


# ======================================================================== #
#  MIDAS lag-matrix builder                                                 #
# ======================================================================== #


def _build_lag_matrix(
    target_dates: pd.DatetimeIndex | pd.Series | np.ndarray | list | pd.Timestamp,
    regressors: pd.DataFrame,
    n_lags: int,
    start_lag: int = 0,
) -> np.ndarray:
    """Build a ``(T, n_lags)`` MIDAS regressor matrix.

    For each target date, selects the *n_lags* most recent monthly
    observations on or before that date.  Column 0 is the most recent
    lag (j = start_lag), consistent with the weight functions in
    `temporal_weights`.  Positions where a lag falls
    before the start of the monthly series are filled with ``NaN``.

    Parameters
    ----------
    target_dates : pd.DatetimeIndex | pd.Series | np.ndarray | list | pd.Timestamp
        Low-frequency target dates.  Accepts any date-like input:
        ``pd.DatetimeIndex``, ``pd.Series``, ``np.ndarray``, list of
        strings/timestamps, or a single date for forecasting.
    regressors : pd.DataFrame
        High-frequency (monthly) data for a **single** indicator.
        Must contain ``date`` and ``value`` columns.  Rows are sorted by
        date before the lag matrix is built.
    n_lags : int
        Number of high-frequency lags to include.
    start_lag : int
        Index of the first lag to include (default 0).  When
        ``start_lag=1`` the most recent monthly observation (lag 0)
        is skipped.

    Returns
    -------
    X : np.ndarray
        Lag matrix.  ``X[t, 0]`` is lag *start_lag*; ``X[t, k]`` is
        lag *start_lag + k*.

    Raises
    ------
    ValueError
        If *n_lags* < 1.
    """
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1.")

    target_dates = np.sort(
        pd.to_datetime(np.atleast_1d(target_dates)).to_numpy(dtype="datetime64[ns]")
    )
    regressors_sorted = regressors.sort_values("date")
    monthly_dates = pd.to_datetime(regressors_sorted["date"]).to_numpy(
        dtype="datetime64[ns]"
    )
    monthly_values = regressors_sorted["value"].to_numpy(dtype=float)

    valid_monthly = np.isfinite(monthly_values)
    if not valid_monthly.any():
        return np.full((len(target_dates), n_lags), np.nan)

    lag_offsets = np.arange(start_lag, start_lag + n_lags)

    # --- Ragged edge logic ---
    # Match each quarter's data availability to the same month within the quarter.
    # For example, February 2025 anchors February 2024 and November 2024.
    latest_month = pd.Timestamp(monthly_dates[valid_monthly][-1])
    latest_month_end = latest_month + pd.offsets.MonthEnd(0)
    month_within_quarter = ((latest_month_end.month - 1) % 3) + 1  # 1, 2, or 3

    # Vectorized calculation of quarter start for each target date
    target_dates_pd = pd.to_datetime(target_dates)
    quarter_end = (target_dates_pd + pd.offsets.QuarterEnd(0)) - pd.offsets.MonthEnd(0)
    anchor_months = (
        quarter_end
        - pd.offsets.MonthEnd(3 - month_within_quarter)
        + pd.offsets.MonthEnd(0)
    )

    # Use np.searchsorted to align anchor months to available monthly data
    anchor_indices = np.searchsorted(monthly_dates, anchor_months, side="right")
    col_indices = anchor_indices[:, None] - 1 - lag_offsets

    in_range = col_indices >= 0
    safe_indices = np.clip(col_indices, 0, len(monthly_values) - 1)
    X = monthly_values[safe_indices].astype(float)
    X[~in_range] = np.nan
    return X


def _build_horizon_lag_matrix(
    target_dates: np.ndarray,
    regressors: pd.DataFrame,
    n_lags: int,
    start_lag: int,
    horizon: int,
    *,
    quarterly: bool = False,
) -> np.ndarray:
    """Build target-aligned lag windows at the specified direct horizon.

    Parameters
    ----------
    target_dates : np.ndarray
        Quarterly target dates in chronological order.
    regressors : pd.DataFrame
        Single-indicator data with ``date`` and ``value`` columns.
    n_lags : int
        Number of indicator lags in each window.
    start_lag : int
        First indicator lag to include.
    horizon : int
        Target quarter minus regressor-origin quarter. Negative values
        select windows in quarters after the target.
    quarterly : bool
        Whether to build quarterly rather than monthly lag windows.

    Returns
    -------
    np.ndarray
        Target-aligned lag matrix. Origins beyond the latest finite
        indicator quarter are missing rather than reusing an older window.
    """
    origin_periods = pd.DatetimeIndex(target_dates).to_period("Q") - horizon
    origin_dates = origin_periods.to_timestamp(how="end").normalize()
    builder = _build_quarterly_lag_matrix if quarterly else _build_lag_matrix
    matrix = builder(origin_dates, regressors, n_lags, start_lag)
    finite_dates = regressors.loc[np.isfinite(regressors["value"]), "date"]
    if finite_dates.empty:
        matrix[:] = np.nan
    else:
        latest_period = pd.Timestamp(finite_dates.max()).to_period("Q")
        matrix[origin_periods > latest_period] = np.nan
    return matrix


def _build_quarterly_lag_matrix(
    target_dates: pd.DatetimeIndex | pd.Series | np.ndarray | list | pd.Timestamp,
    regressors: pd.DataFrame,
    n_lags: int,
    start_lag: int = 0,
) -> np.ndarray:
    """Build a ``(T, n_lags)`` quarterly lag matrix.

    Aligns a quarterly regressor with quarterly target dates.  Column 0
    is lag ``start_lag`` (i.e. ``x_{t-start_lag}``) and column ``k`` is
    lag ``start_lag + k``.  Positions where a lag is missing are set
    to ``NaN``.

    Parameters
    ----------
    target_dates : pd.DatetimeIndex | pd.Series | np.ndarray | list | pd.Timestamp
        Quarterly target dates.
    regressors : pd.DataFrame
        Quarterly regressor for a **single** indicator, with ``date``
        and ``value`` columns.
    n_lags : int
        Number of quarterly lags (``>= 1``).
    start_lag : int
        Index of the first lag to include (default 0, contemporaneous).

    Returns
    -------
    X : np.ndarray

    Raises
    ------
    ValueError
        If *n_lags* is less than 1.
    """
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1.")

    target_dates = pd.to_datetime(np.atleast_1d(target_dates))
    regressors_sorted = regressors.sort_values("date")
    q_dates = pd.to_datetime(regressors_sorted["date"]).to_numpy(dtype="datetime64[ns]")
    q_values = regressors_sorted["value"].to_numpy(dtype=float)
    finite_dates = q_dates[np.isfinite(q_values)]

    if len(finite_dates) == 0:
        return np.full((len(target_dates), n_lags), np.nan)

    # Align by calendar quarter rather than by the position of the rows.
    # This preserves an omitted or nonfinite interior quarter as a missing lag.
    values_by_quarter = {
        pd.Timestamp(date).to_period("Q"): value
        for date, value in zip(q_dates, q_values)
    }
    first_finite = pd.Timestamp(finite_dates[0]).to_period("Q")
    last_finite = pd.Timestamp(finite_dates[-1]).to_period("Q")
    lag_offsets = range(start_lag, start_lag + n_lags)
    rows = []
    for date in target_dates:
        target_quarter = pd.Timestamp(date).to_period("Q")
        if target_quarter < first_finite:
            rows.append([np.nan] * n_lags)
            continue
        anchor = min(target_quarter, last_finite)
        rows.append(
            [values_by_quarter.get(anchor - lag, np.nan) for lag in lag_offsets]
        )
    X = np.asarray(rows, dtype=float)
    return X


def _build_ar_lag_matrix(y: np.ndarray, n_ar_lags: int) -> np.ndarray:
    """Build a ``(T, n_ar_lags)`` matrix of lagged dependent values.

    Row ``t`` contains ``[y[t-1], y[t-2], ..., y[t-n_ar_lags]]``.
    Entries that fall before the start of the series are ``NaN``.
    Returns an empty ``(T, 0)`` matrix when ``n_ar_lags == 0``.
    """
    if n_ar_lags < 0:
        raise ValueError("n_ar_lags must be >= 0.")
    T = len(y)
    out = np.full((T, n_ar_lags), np.nan)
    for k in range(1, n_ar_lags + 1):
        if k < T:
            out[k:, k - 1] = y[: T - k]
    return out


def sample_data(
    n_obs: int = 100,
    n_lags: int = 6,
    alpha: float = 2.0,
    beta_: float = 1.0,
    noise: float = 0.5,
    seed: int = 42,
    horizon: int = 0,
    method: str = "exp_almon",
    theta_true: list[float] | np.ndarray | None = None,
    n_ar_lags: int = 0,
    phi_true: list[float] | np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate sample quarterly target and monthly regressor data.

    Generates synthetic data from a specified MIDAS structure.

    The target is generated with a direct-forecast lead relationship:

    ``y[t+h] = alpha + beta * X[t] @ w
              + sum_{k=1..p} phi[k-1] * y[t+h-k]
              + noise``,

    where *h* is the *horizon* parameter and *p* = ``n_ar_lags``.  When
    ``horizon=0`` and ``n_ar_lags=0`` (default) this reduces to the
    contemporaneous ``y[t] ~ X[t]``.

    Parameters
    ----------
    n_obs : int
        Number of quarterly observations (default 100).
    n_lags : int
        Number of monthly lags (default 6).
    alpha : float
        Intercept (default 2.0).
    beta_ : float
        Slope coefficient (default 1.0).
    noise : float
        Noise standard deviation (default 0.5).
    seed : int
        Random seed (default 42).
    horizon : int
        Forecast lead used to generate the relationship
        ``y[t+h] ~ X[t]`` (default 0).
    method : str
        Weighting scheme used to generate the true weights
        (default ``'exp_almon'``).  Any scheme accepted by
        `get_weights()` is valid.
    theta_true : list[float] | np.ndarray | None
        Weight-shape parameters.  When ``None`` (default), uses
        ``[-0.5, -0.1]``.
    n_ar_lags : int
        Number of autoregressive lags of the dependent variable to
        embed in the DGP (default 0 = no AR term).
    phi_true : list[float] | np.ndarray | None
        AR coefficients of length ``n_ar_lags``.  When ``None`` and
        ``n_ar_lags > 0``, defaults to ``[0.3, -0.1, ...]`` truncated
        / zero-padded to ``n_ar_lags`` entries.  Ignored when
        ``n_ar_lags == 0``.

    Returns
    -------
    target : pd.DataFrame
        Quarterly target with ``date`` and ``value`` columns.
    regressors : pd.DataFrame
        Monthly regressors with ``date`` and ``value`` columns.

    Raises
    ------
    ValueError
        If the requested autoregressive lags, forecast horizon, or
        autoregressive coefficient length is invalid for the sample.

    Examples
    --------
    >>> target, regressors = sample_data(n_obs=100, n_lags=6, seed=0)
    >>> from nowcast_midas.midas import MIDAS
    >>> m = MIDAS(n_lags=6).fit(target, regressors)
    """
    from .temporal_weights import get_weights

    if n_ar_lags < 0:
        raise ValueError("n_ar_lags must be >= 0")

    # Set up RNG and constants
    rng = np.random.default_rng(seed)

    # Set default temporal-weight parameters.
    if theta_true is None:
        theta_true = [-0.5, -0.1]
    true_weights = np.asarray(get_weights(method, np.array(theta_true), n_lags))

    # Set default autoregressive coefficients.
    if n_ar_lags > 0:
        if phi_true is None:
            default_phi = np.array([0.3, -0.1])
            phi_arr = np.zeros(n_ar_lags)
            phi_arr[: min(n_ar_lags, len(default_phi))] = default_phi[:n_ar_lags]
            phi_true = phi_arr
        else:
            phi_true = np.asarray(phi_true, dtype=float)
            if len(phi_true) != n_ar_lags:
                raise ValueError(
                    f"phi_true has length {len(phi_true)}, expected {n_ar_lags}."
                )
    else:
        phi_true = np.array([])

    # Build date columns first
    dates_q = pd.date_range("2000-03-31", periods=n_obs, freq="QE")
    dates_m = pd.date_range(
        dates_q[0] - pd.DateOffset(months=n_lags),
        end=dates_q[-1],
        freq="ME",
    )

    # Monthly regressor values (random)
    monthly_vals = rng.standard_normal(len(dates_m))

    # Build lag matrix from monthly values to get X
    regressors_df = pd.DataFrame({"date": dates_m, "value": monthly_vals})
    X = _build_lag_matrix(dates_q, regressors_df, n_lags)

    # Drop rows where X has missing lags
    valid = ~np.any(np.isnan(X), axis=1)
    X = X[valid]
    dates_q = dates_q[valid]

    T = X.shape[0]
    if horizon < 0 or horizon >= T:
        raise ValueError(f"horizon must be in [0, {T - 1}], got {horizon}.")

    # Generate target from known X, weights, and (optionally) own lags.
    # DGP:  y[t+h] = alpha + beta * X[t] @ w
    #               + sum_{k=1..p} phi[k-1] * y[t+h-k] + noise
    # When MIDAS.fit() processes horizon h it pairs y[h:] with X[:T-h],
    # so the signal is recovered at the correct horizon.
    p = n_ar_lags
    if p == 0:
        if horizon > 0:
            y = np.empty(T)
            y[:horizon] = alpha + rng.standard_normal(horizon)
            y[horizon:] = (
                alpha
                + beta_ * (X[: T - horizon] @ true_weights)
                + noise * rng.standard_normal(T - horizon)
            )
        else:
            y = alpha + beta_ * (X @ true_weights) + noise * rng.standard_normal(T)
    else:
        warmup = p + horizon
        if warmup >= T:
            raise ValueError(
                f"n_obs too small for n_ar_lags={p} and horizon={horizon}."
            )
        y = np.empty(T)
        y[:warmup] = alpha + rng.standard_normal(warmup)
        for t in range(warmup, T):
            i = t - horizon
            ar_term = float(phi_true @ y[i - p : i][::-1])
            y[t] = (
                alpha
                + beta_ * float(X[i] @ true_weights)
                + ar_term
                + noise * float(rng.standard_normal())
            )

    # Assemble DataFrames
    target = pd.DataFrame({"date": dates_q, "value": y})
    regressors = pd.DataFrame({"date": dates_m, "value": monthly_vals})

    return target, regressors


def sample_combo_data(
    n_quarters: int = 60,
    n_lags: int = 6,
    monthly_vars: list[str] | None = None,
    quarterly_vars: list[str] | None = None,
    alpha: float = 1.0,
    betas: dict[str, float] | None = None,
    gammas: dict[str, float] | None = None,
    noise: float = 0.5,
    seed: int = 42,
    method: str = "exp_almon",
    theta_true: list[float] | np.ndarray | None = None,
    horizon: int = 0,
    outlier_date: str | pd.Timestamp | None = "2020-06-30",
    outlier_size: float = -25.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Generate sample data for the ``MidasCombo`` pipeline.

    Each monthly indicator ``v`` is drawn iid ``N(0, 1)`` and enters the
    target through its own MIDAS DGP; each quarterly indicator ``z``
    enters linearly.  With lead ``h = horizon`` the full DGP is:

    ``y[t+h] = alpha + sum_v beta_v * X[v, t] @ w + sum_z gamma_z * Z[z, t]
              + outlier[t+h] + eps``,  with ``eps ~ N(0, noise**2)``.

    The lag weights ``w`` come from ``get_weights()`` with shape parameters
    ``theta_true`` (default ``[-0.5, -0.1]``).

    Parameters
    ----------
    n_quarters : int
        Number of quarterly observations (default 60).
    n_lags : int
        Number of monthly lags used both to build the regressor matrix
        and to generate the DGP weights (default 6).
    monthly_vars : list[str] | None
        Names of the monthly (``ME``) indicators.  Default
        ``['monthly_1', 'monthly_2', 'monthly_3']``.
    quarterly_vars : list[str] | None
        Names of the quarterly (``QE``) indicators.  Default
        ``['quarterly_1']``.
    alpha : float
        DGP intercept (default 1.0).
    betas : dict[str, float] | None
        Per-indicator MIDAS coefficients.  Defaults to
        ``{'PMI': 1.0, 'IP': 0.8, 'GDPM': 0.5}``; any missing entry
        falls back to ``1.0``.
    gammas : dict[str, float] | None
        Per-indicator OLS coefficients for quarterly regressors.
        Defaults to ``{'UNEMP': -0.5}``; missing entries fall back to
        ``1.0``.
    noise : float
        Standard deviation of the Gaussian target noise (default 0.5).
    seed : int
        Random seed (default 42).
    method : str
        Weighting scheme used to generate the true weights.
    theta_true : list[float] | np.ndarray | None
        Forwarded to `get_weights()` to
        produce the true monthly lag weights.
    horizon : int
        Forecast lead used to generate the relationship
        ``y[t+h] ~ X[t]`` (default 0 = contemporaneous).  Must satisfy
        ``0 <= horizon < n_quarters``.
    outlier_date : str | pd.Timestamp | None
        Quarter at which to inject a one-off additive shock to the
        target.  Pass ``None`` to skip.  Default ``'2020-06-30'``.
    outlier_size : float
        Size of the additive shock at ``outlier_date`` (default ``-25``).

    Returns
    -------
    target : pd.DataFrame
        Quarterly target in long format with columns
        ``date``, ``variable``, ``frequency``, ``value``.
    regressors : pd.DataFrame
        Monthly and quarterly regressors in the same long format.
    info : dict
        Ground-truth metadata: ``outlier_date``, ``alpha``, ``betas``,
        ``gammas``, ``weights`` (length ``n_lags``), ``noise``,
        ``monthly_vars``, ``quarterly_vars``.

    Raises
    ------
    ValueError
        If the requested horizon or outlier date is outside the simulated
        sample, or if autoregressive settings are inconsistent.
    """
    from .temporal_weights import get_weights

    if monthly_vars is None:
        monthly_vars = ["monthly_1", "monthly_2", "monthly_3"]
    if quarterly_vars is None:
        quarterly_vars = ["quarterly_1"]

    default_betas = {"monthly_1": 1.0, "monthly_2": 0.8, "monthly_3": 0.5}
    default_gammas = {"quarterly_1": -0.5}
    betas = {v: (betas or {}).get(v, default_betas.get(v, 1.0)) for v in monthly_vars}
    gammas = {
        v: (gammas or {}).get(v, default_gammas.get(v, 1.0)) for v in quarterly_vars
    }

    if theta_true is None:
        theta_true = [-0.5, -0.1]
    true_weights = np.asarray(get_weights(method, np.array(theta_true), n_lags))

    if horizon < 0 or horizon >= n_quarters:
        raise ValueError(f"horizon must be in [0, {n_quarters - 1}], got {horizon}.")

    rng = np.random.default_rng(seed)

    n_months = n_quarters * 3 + n_lags  # buffer for the deepest lag
    dates_q = pd.date_range("2010-03-31", periods=n_quarters, freq="QE")
    dates_m = pd.date_range(end=dates_q[-1], periods=n_months, freq="ME")

    # ------------------------------------------------------------------
    # Build monthly regressors (iid N(0,1)) and their per-quarter lag
    # matrices, then the target contribution from each.
    # ------------------------------------------------------------------
    # Signal contribution: regressor entry at row ``t`` drives y[t+horizon].
    monthly_data: dict[str, np.ndarray] = {}
    signal = np.full(n_quarters, alpha, dtype=float)
    for v in monthly_vars:
        x_m = rng.standard_normal(n_months)
        monthly_data[v] = x_m
        X = _build_lag_matrix(
            dates_q, pd.DataFrame({"date": dates_m, "value": x_m}), n_lags
        )
        # The leading buffer supplies every monthly lag in X.
        signal += betas[v] * (X @ true_weights)

    quarterly_data: dict[str, np.ndarray] = {}
    for v in quarterly_vars:
        z = rng.standard_normal(n_quarters)
        quarterly_data[v] = z
        signal += gammas[v] * z

    # Shift the signal forward by ``horizon`` quarters: y[t+h] = signal[t] + eps.
    # The first ``horizon`` quarters of y are unrelated noise (warm-up).
    y = np.empty(n_quarters)
    eps = noise * rng.standard_normal(n_quarters)
    if horizon == 0:
        y[:] = signal + eps
    else:
        y[:horizon] = alpha + noise * rng.standard_normal(horizon)
        y[horizon:] = signal[: n_quarters - horizon] + eps[horizon:]

    # ------------------------------------------------------------------
    # Inject outlier
    # ------------------------------------------------------------------
    outlier_ts: pd.Timestamp | None = None
    if outlier_date is not None:
        outlier_ts = pd.Timestamp(outlier_date)
        idx = np.where(dates_q == outlier_ts)[0]
        if len(idx) == 0:
            raise ValueError(
                f"outlier_date {outlier_ts!r} is not one of the simulated quarters"
            )
        y[int(idx[0])] += outlier_size

    # ------------------------------------------------------------------
    # Assemble long-format DataFrames
    # ------------------------------------------------------------------
    target = pd.DataFrame(
        {
            "date": dates_q,
            "variable": "quarterly_target",
            "frequency": "QE",
            "value": y,
        }
    )
    reg_frames = [
        pd.DataFrame(
            {
                "date": dates_m,
                "variable": v,
                "frequency": "ME",
                "value": monthly_data[v],
            }
        )
        for v in monthly_vars
    ] + [
        pd.DataFrame(
            {
                "date": dates_q,
                "variable": v,
                "frequency": "QE",
                "value": quarterly_data[v],
            }
        )
        for v in quarterly_vars
    ]
    regressors = pd.concat(reg_frames, ignore_index=True)

    info = {
        "outlier_date": outlier_ts,
        "alpha": alpha,
        "betas": betas,
        "gammas": gammas,
        "weights": true_weights,
        "noise": noise,
        "horizon": horizon,
        "monthly_vars": list(monthly_vars),
        "quarterly_vars": list(quarterly_vars),
    }
    return target, regressors, info


def _build_dummy_matrix(
    target_dates: pd.DatetimeIndex | pd.Series | np.ndarray | list | pd.Timestamp,
    dummy_quarters: list,
    variable_name: str = "quarterly_dummy",
    frequency: str = "QE",
) -> np.ndarray:
    """Build a dummy matrix with one column per specified quarter.

    Parameters
    ----------
    target_dates : pd.DatetimeIndex | pd.Series | np.ndarray | list | pd.Timestamp
        Quarterly target dates.
    dummy_quarters : list
        List of quarters for which dummies should be created.
        Each quarter gets its own column (1 at that date, 0 elsewhere).
    variable_name : str
        Unused, kept for API compat.
    frequency : str
        Unused, kept for API compat.

    Returns
    -------
    D : np.ndarray
        One column per dummy quarter.

    Notes
    -----
    *variable_name* and *frequency* are accepted for compatibility and do not
    affect the returned matrix.
    """
    target_dates = pd.to_datetime(np.atleast_1d(target_dates))
    dummy_quarters = pd.to_datetime(dummy_quarters)

    D = np.zeros((len(target_dates), len(dummy_quarters)))
    for j, dq in enumerate(dummy_quarters):
        D[:, j] = (target_dates == dq).astype(float)
    return D
