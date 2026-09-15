"""Weighting functions for combining MIDAS forecasts."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

__all__ = [
    "clipped_ols",
    "constrained_least_squares",
    "fit_average",
    "fit_error_based_weights",
    "fit_regression_weights",
]


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


def _give_equal_weights(available: np.ndarray) -> np.ndarray:
    """Return equal weights over the currently available sources."""
    n_available = int(available.sum())
    if n_available == 0:
        return np.zeros(len(available))
    return np.where(available, 1.0 / n_available, 0.0)


def _get_discounted_error_stat_at_t(
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


def _combine_equal_and_error_weights_at_t(
    fitted_available: np.ndarray,
    equal_weight_mask: np.ndarray,
    stats: np.ndarray,
) -> np.ndarray:
    """Combine equal weights for warm sources with inverse-error weights.

    Sources still inside their discount-window warm-up period receive an
    equal share ``1 / n_total`` of the available sources. The remaining
    probability mass is split among the sources past warm-up, proportional
    to their inverse error statistic, so the full vector sums to 1.

    Parameters
    ----------
    fitted_available : np.ndarray
        Boolean mask of sources with a finite fitted value at *t*.
    equal_weight_mask : np.ndarray
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

    error_weighted_mask = fitted_available & ~equal_weight_mask
    n_equal_weighted = int(equal_weight_mask.sum())
    weights[equal_weight_mask] = 1.0 / n_total

    if error_weighted_mask.any():
        inv_error = 1.0 / np.maximum(stats[error_weighted_mask], 1e-10)
        remaining_mass = 1.0 - (n_equal_weighted / n_total)
        weights[error_weighted_mask] = remaining_mass * inv_error / inv_error.sum()

    return weights


def fit_regression_weights(
    target: pd.Series,
    source_fitted: pd.DataFrame,
    *,
    method: str,
    window: int | None = None,
    dummy_periods: list | None = None,
    minimum_sample_size: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit time-varying weights using clipped or constrained regression."""
    if method not in ("clipped_ols", "constrained_ls"):
        raise ValueError(f"Regression method '{method}' is not supported.")
    if minimum_sample_size is not None and minimum_sample_size < 1:
        raise ValueError("minimum_sample_size must be >= 1 when provided.")

    names = source_fitted.columns.tolist()
    T = len(target)
    n_models = len(names)
    minimum_regression_rows = (
        n_models if minimum_sample_size is None else minimum_sample_size
    )
    fitted_values = source_fitted.to_numpy(dtype=float)
    target_values = target.reindex(source_fitted.index).to_numpy(dtype=float)
    dummy_bool = _dummy_period_mask(
        pd.DatetimeIndex(source_fitted.index), dummy_periods
    )
    weights_matrix = np.full((T + 1, n_models), np.nan)
    for t in range(T + 1):
        if t < T:
            fitted_available = np.isfinite(fitted_values[t])
            if not fitted_available.any():
                continue
        else:
            fitted_available = np.ones(n_models, dtype=bool)
        if t == 0:
            weights_matrix[t] = _give_equal_weights(fitted_available)
            continue
        complete = (
            np.isfinite(target_values[:t])
            & np.isfinite(fitted_values[:t]).all(axis=1)
            & ~dummy_bool[:t]
        )
        indices = np.flatnonzero(complete)
        if len(indices) < minimum_regression_rows:
            weights_matrix[t] = _give_equal_weights(fitted_available)
            continue
        if window is not None and len(indices) > window:
            indices = indices[-window:]
        X = fitted_values[indices]
        y = target_values[indices]
        weights = (
            clipped_ols(X, y)
            if method == "clipped_ols"
            else constrained_least_squares(X, y)
        )
        weights = np.where(fitted_available, weights, 0.0)
        total = weights.sum()
        weights_matrix[t] = (
            weights / total if total > 0 else _give_equal_weights(fitted_available)
        )
    return _calculate_combined_series_using_weights(
        fitted_values, weights_matrix, names
    )


def fit_error_based_weights(
    target: pd.Series,
    source_fitted: pd.DataFrame,
    *,
    method: str,
    window: int | None = None,
    discount_rate: float = 1.0,
    dummy_periods: list | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit time-varying inverse-error weights using MAE, MSE, or RMSE."""
    if method not in ("mae", "mse", "rmse"):
        raise ValueError(f"Error weighting method '{method}' is not supported.")
    names = source_fitted.columns.tolist()
    T = len(target)
    fitted_values = source_fitted.to_numpy(dtype=float)
    target_values = target.reindex(source_fitted.index).to_numpy(dtype=float)
    dummy_bool = _dummy_period_mask(
        pd.DatetimeIndex(source_fitted.index), dummy_periods
    )
    weights_matrix = np.full((T + 1, len(names)), np.nan)
    for t in range(T + 1):
        if t < T:
            fitted_available = np.isfinite(fitted_values[t])
            if not fitted_available.any():
                continue
        else:
            fitted_available = np.ones(len(names), dtype=bool)
        if t == 0:
            weights_matrix[t] = _give_equal_weights(fitted_available)
            continue
        valid_history = (
            np.isfinite(target_values[:t])[:, None]
            & np.isfinite(fitted_values[:t])
            & ~dummy_bool[:t, None]
        )
        counts = valid_history.sum(axis=0)
        equal_mask = (
            fitted_available & (counts < window)
            if window is not None
            else np.zeros(len(names), dtype=bool)
        )
        weighted_mask = fitted_available & ~equal_mask
        stats = np.full(len(names), np.nan)
        for m in np.flatnonzero(weighted_mask):
            valid_col = (
                np.isfinite(target_values[:t])
                & np.isfinite(fitted_values[:t, m])
                & ~dummy_bool[:t]
            )
            stats[m] = _get_discounted_error_stat_at_t(
                fitted_values[:t, m],
                target_values[:t],
                valid_col,
                window,
                discount_rate,
                method,
            )
        weights_matrix[t] = _combine_equal_and_error_weights_at_t(
            fitted_available, equal_mask, stats
        )
    return _calculate_combined_series_using_weights(
        fitted_values, weights_matrix, names
    )


def _calculate_combined_series_using_weights(
    fitted_values: np.ndarray,
    weights_matrix: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Calculate the combined fitted series using the source weights."""
    fitted_safe = np.where(np.isnan(fitted_values), 0.0, fitted_values)
    weights_safe = np.where(np.isnan(weights_matrix), 0.0, weights_matrix)
    combined_fitted = np.where(
        np.any(np.isfinite(weights_matrix[:-1]), axis=1),
        (weights_safe[:-1] * fitted_safe).sum(axis=1),
        np.nan,
    )
    return combined_fitted, {name: weights_safe[:, m] for m, name in enumerate(names)}


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
