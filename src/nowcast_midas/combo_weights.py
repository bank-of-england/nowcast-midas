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


def _fit_error_window(
    fitted_window: np.ndarray,
    target_window: np.ndarray,
    discount_slice: np.ndarray,
    method: str,
) -> np.ndarray:
    """Return weights for one complete fitting window.

    Parameters
    ----------
    fitted_window : np.ndarray
        Fitted values for the models in the window.
    target_window : np.ndarray
        Target values for the observations in the window.
    discount_slice : np.ndarray
        Discount factors for the observations in the window.
    method : str
        Error statistic used to calculate the weights.
    Returns
    -------
    weights : np.ndarray
        Normalised source weights.  Unavailable sources have weight zero.

    Raises
    ------
    ValueError
        If *method* is not a supported error-weighting method.
    """
    residual_window = target_window[:, None] - fitted_window
    n_models = fitted_window.shape[1]
    weights = np.zeros(n_models)

    if method == "mae":
        errors = np.abs(residual_window)
    elif method in ("mse", "rmse"):
        errors = residual_window**2
    else:
        raise ValueError(f"Method '{method}' is not supported.")

    stat = np.mean(errors * discount_slice[:, None], axis=0)
    if method == "rmse":
        stat = np.sqrt(stat)

    inv_error = 1.0 / np.maximum(stat, 1e-10)
    weights = inv_error / inv_error.sum()

    return weights


def _fit_weight(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    discount_slice: np.ndarray | None = None,
) -> np.ndarray:
    """Dispatch to the requested combination-weight estimator.

    ``X`` is the fitted-value matrix and ``y`` is the target vector for every
    method. Error-weighted methods additionally use ``discount_slice``.
    """
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be a 2-D array with one row for each y value.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("All sources must have fitted values in the fitting window.")

    if method in ("mae", "mse", "rmse"):
        if discount_slice is None:
            raise ValueError("discount_slice is required for error-weighted methods.")
        return _fit_error_window(
            X,
            y,
            discount_slice,
            method,
        )
    if method == "clipped_ols":
        return clipped_ols(X, y)
    if method == "constrained_ls":
        return constrained_least_squares(X, y)
    raise ValueError(f"Weighting method '{method}' is not supported.")


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

    Rows containing a NaN in any source are excluded before the lookback
    window is selected, so a finite ``window`` always contains exactly that
    many comparable observations once the window is full. During warm-up,
    error weighted methods use all available complete rows. Regression based
    methods use equal weights until ``minimum_sample_size`` complete rows
    exist.
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
        this warm-up threshold.

    Returns
    -------
    combined : np.ndarray
        Time-varying weighted combination.
    weights : dict[str, np.ndarray]
        Source names to weight arrays of shape (T,). ``forecast()`` uses
        the final in-sample weight for the out-of-sample forecast.

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
    weights_matrix = np.full((n_rows, n_models), np.nan)

    # Calculate one in-sample weight vector per date. Forecasting reuses the
    # final row for the out-of-sample forecast. For backseries, pseudo out of sample
    # weights are used such that each fitted value at t is weighted by performance
    # in the period up to t-1 (with discounting if applicable).
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
        complete_history = (
            np.isfinite(history_target)
            & np.isfinite(history_fitted).all(axis=1)
            & ~dummy_bool[:history_end]
        )

        complete_indices = np.flatnonzero(complete_history)
        if (
            method in regression_methods
            and len(complete_indices) < minimum_regression_rows
        ):
            weights_matrix[t] = _equal_weights(fitted_available)
            continue

        if window is not None and len(complete_indices) > window:
            complete_indices = complete_indices[-window:]

        fitted_window = history_fitted[complete_indices]
        target_window = history_target[complete_indices]
        window_length = len(complete_indices)
        discount_weights = np.array(
            [discount_rate ** (window_length - 1 - i) for i in range(window_length)]
        )
        discount_slice = discount_weights

        if len(fitted_window) == 0:
            weights_matrix[t] = _equal_weights(fitted_available)
            continue

        weights_matrix[t] = _mask_weights(
            _fit_weight(
                method,
                fitted_window,
                target_window,
                discount_slice=discount_slice,
            ),
            fitted_available,
        )

    # Combine -----------------------------------------------------------------
    # Apply each in-sample weight row to its fitted values.
    fitted_safe = np.where(np.isnan(fitted_values), 0.0, fitted_values)

    w_safe = np.where(np.isnan(weights_matrix), 0.0, weights_matrix)

    combined = np.where(
        np.any(np.isfinite(weights_matrix), axis=1),
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
