"""Weighting functions for combining MIDAS forecasts."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

__all__ = ["clipped_ols", "constrained_least_squares", "fit_average", "fit_weights"]


def fit_average(
    source_fitted: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Equal-weight average.

    Parameters
    ----------
    source_fitted : pd.DataFrame
        Fitted value matrix with source names as column names.

    Returns
    -------
    combined : np.ndarray
        Equally-weighted average of sources.
    weights : dict[str, np.ndarray]
        Source names to time-varying equal weights over available sources.
    """
    names = source_fitted.columns.tolist()
    mat = source_fitted.to_numpy(dtype=float)
    if mat.shape[1] == 0:
        return np.full(len(source_fitted), np.nan), {}
    all_nan_rows = np.all(np.isnan(mat), axis=1)
    combined = np.full(mat.shape[0], np.nan)
    if (~all_nan_rows).any():
        combined[~all_nan_rows] = np.nanmean(mat[~all_nan_rows], axis=1)
    T, n_models = mat.shape
    weights_mat = np.full((T, n_models), np.nan)
    for t in range(T):
        avail = np.isfinite(mat[t])
        if avail.any():
            weights_mat[t] = np.where(avail, 1.0 / avail.sum(), 0.0)
    weights = {name: weights_mat[:, m] for m, name in enumerate(names)}
    return combined, weights


def _dummy_period_mask(
    index: pd.Index,
    dummy_periods: list | None,
) -> np.ndarray:
    """Return a mask for dummy quarters on the supplied date index."""
    if dummy_periods is None:
        return np.zeros(len(index), dtype=bool)

    dummy_quarters = set()
    for period in dummy_periods:
        try:
            dummy_quarters.add(pd.Period(period, freq="Q"))
        except (TypeError, ValueError):
            dummy_quarters.add(pd.Timestamp(period).to_period("Q"))
    return np.asarray(pd.DatetimeIndex(index).to_period("Q").isin(dummy_quarters))


def _filter_sources(
    source_fitted: pd.DataFrame,
    minimum_sample_size: int,
) -> pd.DataFrame:
    """Remove sources with too few finite fitted values."""
    source_counts = np.isfinite(source_fitted.to_numpy(dtype=float)).sum(axis=0)
    keep = source_counts >= minimum_sample_size
    return source_fitted.loc[:, keep]


def _equal_weights(available: np.ndarray) -> np.ndarray:
    """Return equal weights over the currently available sources."""
    n_available = int(available.sum())
    if n_available == 0:
        return np.zeros(len(available))
    return np.where(available, 1.0 / n_available, 0.0)


def _mask_weights(weights: np.ndarray, available: np.ndarray) -> np.ndarray:
    """Remove unavailable sources and renormalise the remaining weights."""
    weights = np.where(available, weights, 0.0)
    weight_sum = weights.sum()
    if weight_sum > 0:
        return weights / weight_sum
    return _equal_weights(available)


def _source_complete_counts(
    history_fitted: np.ndarray,
    history_target: np.ndarray,
    dummy_slice: np.ndarray,
) -> np.ndarray:
    """Return, per source, the count of usable historical rows."""

    valid = (
        np.isfinite(history_target)[:, None]
        & np.isfinite(history_fitted)
        & ~dummy_slice[:, None]
    )
    return valid.sum(axis=0)


def _source_error_stat(
    fitted_col: np.ndarray,
    target_col: np.ndarray,
    valid_col: np.ndarray,
    window: int | None,
    discount_rate: float,
    method: str,
) -> float:
    """Discounted error statistic for one source over its own valid history.

    Parameters
    ----------
    fitted_col : np.ndarray
        One source's fitted-value history, shape ``(t,)``.
    target_col : np.ndarray
        Target history, shape ``(t,)``.
    valid_col : np.ndarray
        Boolean mask of usable rows for this source, shape ``(t,)``.
    window : int | None
        Rolling window size, or ``None`` for an expanding window.
    discount_rate : float
        Exponential discount rate applied to older rows.
    method : str
        One of ``'mae'``, ``'mse'``, or ``'rmse'``.

    Returns
    -------
    stat : float
        The discounted error statistic for this source.
    """
    idx = np.flatnonzero(valid_col)
    if window is not None and len(idx) > window:
        idx = idx[-window:]

    residual = target_col[idx] - fitted_col[idx]
    errors = np.abs(residual) if method == "mae" else residual**2

    n = len(idx)
    discount = discount_rate ** (n - 1 - np.arange(n))
    stat = np.mean(errors * discount)
    return float(np.sqrt(stat)) if method == "rmse" else float(stat)


def _blend_warm_computed_weights(
    fitted_available: np.ndarray,
    warm_mask: np.ndarray,
    stats: np.ndarray,
) -> np.ndarray:
    """Blend equal weights (warm sources) with inverse-error weights.

    Sources still inside their discount-window warm-up period receive an
    equal share ``1 / n_total`` of the available sources. The remaining
    probability mass is split among the sources past warm-up, proportional
    to their inverse error statistic, so the full vector sums to 1.

    Parameters
    ----------
    fitted_available : np.ndarray
        Boolean mask of sources with a finite fitted value at *t*.
    warm_mask : np.ndarray
        Boolean mask of available sources still inside warm-up.
    stats : np.ndarray
        Per-source error statistic; only read for sources past warm-up.

    Returns
    -------
    weights : np.ndarray
        Normalised weights, zero for unavailable sources.
    """
    n_total = int(fitted_available.sum())
    weights = np.zeros(len(fitted_available))
    if n_total == 0:
        return weights

    computed_mask = fitted_available & ~warm_mask
    n_warm = int(warm_mask.sum())
    weights[warm_mask] = 1.0 / n_total

    if computed_mask.any():
        inv_error = 1.0 / np.maximum(stats[computed_mask], 1e-10)
        remaining_mass = 1.0 - (n_warm / n_total)
        weights[computed_mask] = remaining_mass * inv_error / inv_error.sum()

    return weights


def _fit_weight(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Dispatch to the requested regression-based weight estimator.

    ``X`` is the fitted-value matrix and ``y`` is the target vector.
    """
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be a 2-D array with one row for each y value.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("All sources must have fitted values in the fitting window.")

    if method == "clipped_ols":
        return clipped_ols(X, y)
    if method == "constrained_ls":
        return constrained_least_squares(X, y)
    raise ValueError(f"Weighting method '{method}' is not supported.")


def _weight_row(
    method: str,
    history_fitted: np.ndarray,
    history_target: np.ndarray,
    dummy_slice: np.ndarray,
    fitted_available: np.ndarray,
    window: int | None,
    discount_rate: float,
    regression_methods: set[str],
    minimum_regression_rows: int,
) -> np.ndarray:
    """Compute one weight vector from a block of history.

    Shared by the main in-sample loop (history strictly before row *t*) and
    by the extra one-step-ahead row appended after it (history covering the
    entire in-sample sample), so both use identical warm-up/estimation
    logic.

    Parameters
    ----------
    method : str
        Weighting method.
    history_fitted : np.ndarray
        Fitted-value history to estimate from, shape ``(n_history, n_models)``.
    history_target : np.ndarray
        Target history aligned to ``history_fitted``, shape ``(n_history,)``.
    dummy_slice : np.ndarray
        Dummy-period mask aligned to the history rows, shape ``(n_history,)``.
    fitted_available : np.ndarray
        Boolean mask of sources considered available for this row.
    window : int | None
        Rolling window size, or ``None`` for an expanding window.
    discount_rate : float
        Exponential discount rate for error weighting.
    regression_methods : set[str]
        Method names estimated via a shared joint-row design matrix.
    minimum_regression_rows : int
        Minimum jointly-complete rows required before regression weights
        are estimated.

    Returns
    -------
    weights : np.ndarray
        Normalised weight vector, shape ``(n_models,)``.
    """
    if method in regression_methods:
        complete_history = (
            np.isfinite(history_target)
            & np.isfinite(history_fitted).all(axis=1)
            & ~dummy_slice
        )
        complete_indices = np.flatnonzero(complete_history)
        if len(complete_indices) < minimum_regression_rows:
            return _equal_weights(fitted_available)

        if window is not None and len(complete_indices) > window:
            complete_indices = complete_indices[-window:]

        fitted_window = history_fitted[complete_indices]
        target_window = history_target[complete_indices]

        if len(fitted_window) == 0:
            return _equal_weights(fitted_available)

        return _mask_weights(
            _fit_weight(method, fitted_window, target_window),
            fitted_available,
        )

    # mae / mse / rmse: per-source warm-up and error stats
    counts = _source_complete_counts(history_fitted, history_target, dummy_slice)
    if window is not None:
        warm_mask = fitted_available & (counts < window)
    else:
        warm_mask = np.zeros(len(fitted_available), dtype=bool)
    computed_mask = fitted_available & ~warm_mask

    stats = np.full(len(fitted_available), np.nan)
    for m in np.flatnonzero(computed_mask):
        valid_col = (
            np.isfinite(history_target)
            & np.isfinite(history_fitted[:, m])
            & ~dummy_slice
        )
        stats[m] = _source_error_stat(
            history_fitted[:, m],
            history_target,
            valid_col,
            window,
            discount_rate,
            method,
        )

    return _blend_warm_computed_weights(fitted_available, warm_mask, stats)


def fit_weights(
    target: pd.Series,
    source_fitted: pd.DataFrame,
    *,
    method: str,
    window: int | None = None,
    discount_rate: float | None = None,
    dummy_periods: list | None = None,
    minimum_sample_size: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Inverse-error weighted combination with exponential discounting.

    At each time *t* the weight for model *m* is proportional to the
    inverse of its discounted error statistic computed over the latest
    complete residual rows before *t*.
    Residuals are model-specific (``target - source_fitted.iloc[:, m]``),
    using whichever source forecast table the caller provides.

    The error statistic is ``mean(err² * disc)`` (i.e. sum of discounted
    squared errors divided by the count of observations, not by the sum
    of discount weights).

    For error-weighted methods (``'mae'``, ``'mse'``, ``'rmse'``), each
    source's warm-up is evaluated independently: a source with fewer than
    ``window`` of its own usable historical rows receives an equal share
    ``1 / n_available`` at that date, regardless of how long its peers'
    histories are. Sources with at least ``window`` usable rows have their
    error statistic computed from their own most recent ``window`` rows,
    and the probability mass left over after the equal-weighted sources
    are accounted for is split between them in proportion to inverse
    error. Regression-based methods (``'clipped_ols'``, ``'constrained_ls'``)
    keep the existing joint-row estimation sample and use equal weights
    until ``minimum_sample_size`` jointly-complete rows exist.
    Models whose fitted value is NaN at *t* receive zero weight.

    Parameters
    ----------
    target : pd.Series
        Target values.
    source_fitted : pd.DataFrame
        Fitted value matrix with source names as column names.
    method : str
        Error metric for weighting.
    window : int | None
        Lookback window size for computing weights.
    discount_rate : float | None
        Discount factor for exponential weighting (0 < value < 1).
    dummy_periods : list | None
        Quarters to exclude from the error statistic computation.
        These rows are masked out of the residual window so that
        outlier quarters (e.g. COVID) do not distort the weights.
        Default is None (no exclusions).
    minimum_sample_size : int | None
        Minimum number of complete common rows required before regression
        based weights are estimated. Error weighted methods do not require
        this warm-up threshold; their warm-up is instead governed
        per-source by ``window``.

    Returns
    -------
    combined : np.ndarray
        Time-varying weighted combination, shape ``(T,)``.
    weights : dict[str, np.ndarray]
        Source names to weight arrays of shape ``(T + 1,)``. Rows
        ``0..T-1`` are the in-sample weights described above. Row ``T`` is
        an extra one-step-ahead weight, estimated from the *entire*
        in-sample history (rows ``0..T-1``), for `forecast()` to use on
        the out-of-sample step rather than reusing the stale row ``T-1``.

    Raises
    ------
    ValueError
        If *minimum_sample_size* is less than 1 or a weighting method
        receives invalid input.
    """
    names = source_fitted.columns.tolist()
    T = len(target)
    n_models = len(names)

    if minimum_sample_size is not None and minimum_sample_size < 1:
        raise ValueError("minimum_sample_size must be >= 1 when provided.")

    regression_methods = {"clipped_ols", "constrained_ls"}
    minimum_regression_rows = (
        n_models if minimum_sample_size is None else minimum_sample_size
    )

    fitted_values = source_fitted.to_numpy(dtype=float)

    # Exclude dummy periods from the error-estimation sample.
    dummy_bool = _dummy_period_mask(
        pd.DatetimeIndex(source_fitted.index), dummy_periods
    )

    if discount_rate is None:
        discount_rate = 1.0

    n_rows = T
    # Row T (appended after the loop) is an extra one-step-ahead weight,
    # estimated from the entire in-sample history, for out-of-sample use.
    weights_matrix = np.full((n_rows + 1, n_models), np.nan)

    # Calculate one in-sample weight vector per date.
    for t in range(n_rows):
        fitted_available = np.isfinite(fitted_values[t])

        if not fitted_available.any():
            continue

        if t == 0:
            weights_matrix[t] = _equal_weights(fitted_available)
            continue

        history_end = t
        history_index = source_fitted.index[:history_end]
        history_fitted = fitted_values[:history_end]
        history_target = target.reindex(history_index).to_numpy(dtype=float)
        dummy_slice = dummy_bool[:history_end]

        weights_matrix[t] = _weight_row(
            method,
            history_fitted,
            history_target,
            dummy_slice,
            fitted_available,
            window,
            discount_rate,
            regression_methods,
            minimum_regression_rows,
        )

    # Row T: the weight to apply to an out-of-sample forecast, estimated
    # from the full in-sample history (rows 0..T-1). All sources that
    # survived filtering are considered available here; downstream OOS
    # consumers renormalise over whichever sources actually produced a
    # forecast at that date.
    fitted_available_next = np.ones(n_models, dtype=bool)
    weights_matrix[n_rows] = _weight_row(
        method,
        fitted_values,
        target.to_numpy(dtype=float),
        dummy_bool,
        fitted_available_next,
        window,
        discount_rate,
        regression_methods,
        minimum_regression_rows,
    )

    # Combine -----------------------------------------------------------------
    # Apply each in-sample weight row to its fitted values.
    fitted_safe = np.where(np.isnan(fitted_values), 0.0, fitted_values)

    w_safe = np.where(np.isnan(weights_matrix), 0.0, weights_matrix)

    combined = np.where(
        np.any(np.isfinite(weights_matrix[:T]), axis=1),
        (w_safe[:T] * fitted_safe).sum(axis=1),
        np.nan,
    )

    weights_dict = {name: w_safe[:, m] for m, name in enumerate(names)}
    return combined, weights_dict


def clipped_ols(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """OLS with clipping to [0, 1] and sum-to-one normalisation.

    Estimates weights via OLS ``min ||y - X w||^2``, clips each weight
    to [0, 1], and normalises to sum to 1.

    Parameters
    ----------
    X : np.ndarray
        Design matrix with p regressors.
    y : np.ndarray
        Target values.

    Returns
    -------
    weights : np.ndarray
        Non-negative weights clipped to [0, 1] and summing to 1.
    """
    n_sources = X.shape[1]
    if len(y) == 0:
        return np.full(n_sources, np.nan, dtype=float)

    # Estimate the weights with OLS.
    try:
        weights = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.full(n_sources, 1.0 / n_sources, dtype=float)

    # Restrict the weights to [0, 1].
    weights = np.clip(weights, 0, 1)

    # Normalize the weights to sum to 1.
    weight_sum = weights.sum()
    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        weights = np.full(n_sources, 1.0 / n_sources, dtype=float)

    return weights


def constrained_least_squares(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Estimate non-negative weights that sum to one.

    Solves ``min ||y - X w||^2`` subject to ``w >= 0`` and ``sum(w) = 1``.

    Parameters
    ----------
    X : np.ndarray
        Design matrix.
    y : np.ndarray
        Target vector.

    Returns
    -------
    weights : np.ndarray
        Non-negative weights summing to 1.
    """
    n_sources = X.shape[1]
    if len(y) == 0:
        return np.full(n_sources, np.nan, dtype=float)

    scaling_factor = float(np.std(y))
    if np.isclose(scaling_factor, 0.0):
        return np.full(n_sources, 1.0 / n_sources, dtype=float)

    Xs = X / scaling_factor
    ys = y / scaling_factor

    def _softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - z.max())
        return e / e.sum()

    def residuals_np(z: np.ndarray) -> np.ndarray:
        """Return residuals for an unconstrained weight parameter vector."""
        return ys - Xs @ _softmax(z)

    def jac_np(z: np.ndarray) -> np.ndarray:
        """Return the residual Jacobian for an unconstrained parameter vector."""
        s = _softmax(z)  # (p,)
        # J_softmax[i, j] = s[i] * (delta_ij - s[j])  →  shape (p, p)
        J_softmax = np.diag(s) - np.outer(s, s)
        return -Xs @ J_softmax  # shape (n, p)

    # softmax(0) = 1/n for all entries → equal-weight initialisation
    z0 = np.zeros(n_sources)

    result = least_squares(
        residuals_np,
        z0,
        jac=jac_np,
        method="lm",  # Levenberg-Marquardt
    )

    if not result.success:
        warnings.warn(
            "Constrained least-squares did not converge: " + result.message,
            RuntimeWarning,
            stacklevel=2,
        )
        return np.full(n_sources, 1.0 / n_sources, dtype=float)

    return np.asarray(_softmax(result.x), dtype=float)
