"""Quarterly OLS models and fitted result objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._compat import legacy_forecast_alias
from .utils import (
    _build_ar_lag_matrix,
    _build_dummy_matrix,
    _build_horizon_lag_matrix,
    _build_quarterly_lag_matrix,
)

__all__ = ["OLS", "FittedOLS"]


def _latest_finite_quarterly_date(regressors: pd.DataFrame) -> pd.Timestamp:
    """Return the date of the latest quarterly regressor with a finite value."""
    finite = regressors.loc[np.isfinite(regressors["value"]), "date"]
    if finite.empty:
        raise ValueError("regressors contain no finite observations")
    return pd.Timestamp(finite.max())


def _fit_parameters_are_finite(fit: FittedOLS) -> bool:
    """Return whether every fitted parameter is finite."""
    arrays = [fit.coef, fit.gamma, fit.phi]
    return np.isfinite(fit.intercept) and all(
        np.all(np.isfinite(array)) for array in arrays
    )


@dataclass
class FittedOLS:
    """Fitted result for one forecast horizon.

    Attributes
    ----------
    intercept : float
        Estimated intercept.
    coef : np.ndarray
        Estimated quarterly lag coefficients.
    gamma : np.ndarray
        Estimated dummy coefficients, or an empty array when no dummies are
        included.
    phi : np.ndarray
        Estimated autoregressive coefficients, or an empty array when no AR
        lags are included.
    fitted_values : pd.Series
        In-sample fitted values indexed by target date.
    residuals : np.ndarray
        In-sample residuals.
    nobs : int
        Number of observations used for estimation.
    dates : np.ndarray | None
        Target dates aligned with the fitted values.
    """

    intercept: float
    coef: np.ndarray  # shape (n_lags,)
    gamma: np.ndarray  # shape (n_dummies,)
    phi: np.ndarray  # shape (n_ar_lags,)
    fitted_values: pd.Series
    residuals: np.ndarray
    nobs: int
    dates: np.ndarray | None = None


class OLS:
    """Plain OLS regression for a single quarterly regressor.

    Parameters
    ----------
    n_lags : int
        Number of quarterly lags (default 1, i.e. contemporaneous only).
    start_lag : int
        Index of the first lag to include (default 0).
    horizons : list[int] | None
        Direct-forecast horizons (default ``[0]``). Negative horizons use
        regressor windows after the target quarter. Dummies and, for
        negative horizons, AR lags remain aligned to the target quarter.
    n_ar_lags : int
        Number of autoregressive lags of the dependent variable to include
        (default 0 = no AR terms).  When > 0, the model becomes
        ``y[t+h] = alpha + X[t] @ coef + D[t+h] @ gamma
        + sum_{k=1..p} phi_k * y[t+h-k] + eps`` with ``p = n_ar_lags``.
    dummy_periods : list[pd.Timestamp] | None
        Optional outlier-dummy quarters.

    Raises
    ------
    ValueError
        If a model setting is invalid.
    """

    def __init__(
        self,
        n_lags: int = 1,
        start_lag: int = 0,
        horizons: list[int] | None = None,
        n_ar_lags: int = 0,
        dummy_periods: list[pd.Timestamp] | None = None,
    ) -> None:
        if n_lags < 1:
            raise ValueError("n_lags must be >= 1")
        if n_ar_lags < 0:
            raise ValueError("n_ar_lags must be >= 0")

        if horizons is not None and (
            not isinstance(horizons, list)
            or not all(isinstance(h, int) for h in horizons)
        ):
            raise ValueError("horizons must be a list of integers.")

        self.n_lags = n_lags
        self.start_lag = start_lag
        self.horizons = horizons if horizons is not None else [0]
        self.n_ar_lags = n_ar_lags
        self.dummy_periods = dummy_periods

        # Populated by fit() — one FittedOLS per horizon
        self.fits_: dict[int, FittedOLS] = {}
        # Populated by fit() — long-format DataFrame (date, horizon, value)
        self.fits_df_: pd.DataFrame | None = None
        # Mask (length T) of target rows with finite X and D values.
        self.valid_mask_: np.ndarray | None = None
        # Indicator name reported in the ``spec`` column of ``forecast()``.
        self._variable_name: str = "target"

    def fit(self, target: pd.DataFrame, regressors: pd.DataFrame) -> OLS:
        """Fit the OLS model from target and quarterly regressor frames.

        Parameters
        ----------
        target : pd.DataFrame
            Quarterly target with (at least) ``date`` and ``value`` columns.
        regressors : pd.DataFrame
            Quarterly regressor for a single variable, with (at least)
            ``date`` and ``value`` columns.

        Returns
        -------
        self : OLS

        Raises
        ------
        ValueError
            If either input has invalid columns, missing target values, or
            an unsupported horizon.
        """
        if not {"date", "value"}.issubset(target.columns):
            raise ValueError("target DataFrame must have 'date' and 'value' columns.")
        if not {"date", "value"}.issubset(regressors.columns):
            raise ValueError(
                "regressors DataFrame must have 'date' and 'value' columns."
            )

        # Name used in the ``spec`` column of ``forecast()``.
        if "variable" in regressors.columns and regressors["variable"].notna().any():
            self._variable_name = str(regressors["variable"].iloc[0])
        else:
            self._variable_name = "target"

        target = target.sort_values("date")
        regressors = regressors.sort_values("date")
        self.target_ = target.reset_index(drop=True)

        if target["value"].isna().any():
            raise ValueError("target contains NaN values")

        target_dates_np = pd.to_datetime(target["date"]).to_numpy(
            dtype="datetime64[ns]"
        )
        y = target["value"].to_numpy(dtype=float)

        X = _build_quarterly_lag_matrix(
            target_dates_np, regressors, self.n_lags, self.start_lag
        )

        D = None
        if self.dummy_periods is not None and len(self.dummy_periods) > 0:
            D = _build_dummy_matrix(target_dates_np, dummy_quarters=self.dummy_periods)

        Y_ar = _build_ar_lag_matrix(y, self.n_ar_lags)

        # Keep rows with finite target, lag, dummy, and own-lag values. Keep
        # this mask aligned with the full target so direct horizons retain
        # their original calendar pairing.
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if D is not None:
            valid &= np.all(np.isfinite(D), axis=1)
        if self.n_ar_lags > 0:
            valid &= np.all(np.isfinite(Y_ar), axis=1)
        self.valid_mask_ = valid

        T = len(y)
        max_horizon = max(self.horizons)
        if any(
            h >= T or (h >= 0 and not (valid[: T - h] & np.isfinite(y[h:])).any())
            for h in self.horizons
        ):
            raise ValueError(
                f"Maximum horizon {max_horizon} exceeds valid observations "
                f"{int(valid.sum())}."
            )

        self.fits_ = {}
        all_dates = []
        all_horizons = []
        all_values = []

        for h in self.horizons:
            if h < 0:
                X_origin = _build_horizon_lag_matrix(
                    target_dates_np,
                    regressors,
                    self.n_lags,
                    self.start_lag,
                    h,
                    quarterly=True,
                )
                origin_valid = np.isfinite(y) & np.all(np.isfinite(X_origin), axis=1)
                origin_valid &= np.all(np.isfinite(Y_ar), axis=1)
                X_h = X_origin[origin_valid]
                y_h = y[origin_valid]
                D_h = D[origin_valid] if D is not None else None
                Y_ar_h = Y_ar[origin_valid]
                dates_h = target_dates_np[origin_valid]
            else:
                origin_valid = valid[: T - h] & np.isfinite(y[h:])
                X_h = X[: T - h][origin_valid]
                y_h = y[h:][origin_valid]
                D_h = D[h:][origin_valid] if D is not None else None
                Y_ar_h = Y_ar[: T - h][origin_valid]
                dates_h = target_dates_np[h:][origin_valid]
            if not len(y_h):
                raise ValueError(f"No valid observations for horizon {h}.")
            fit = self._fit_single_horizon(y_h, X_h, D_h, Y_ar_h)
            fit.dates = dates_h
            fit.fitted_values = pd.Series(
                fit.fitted_values, index=pd.DatetimeIndex(dates_h)
            )
            self.fits_[h] = fit

            # Accumulate arrays for long-format DataFrame
            all_dates.extend(dates_h)
            all_horizons.extend([h] * len(dates_h))
            all_values.extend(fit.fitted_values.values)

        # Create long-format DataFrame once
        self.fits_df_ = pd.DataFrame(
            {
                "date": all_dates,
                "horizon": all_horizons,
                "value": all_values,
            }
        )

        return self

    def _fit_single_horizon(
        self,
        y: np.ndarray,
        X: np.ndarray,
        D: np.ndarray | None = None,
        Y_ar: np.ndarray | None = None,
    ) -> FittedOLS:
        """Fit one horizon from pre-aligned arrays."""
        y = np.asarray(y, dtype=float).ravel()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        T, n_regressors = X.shape

        if n_regressors != self.n_lags:
            raise ValueError(f"X has {n_regressors} cols, expected {self.n_lags}.")
        if len(y) != T:
            raise ValueError("y and X row counts differ.")

        if Y_ar is None:
            Y_ar = np.empty((T, 0))
        n_ar = Y_ar.shape[1]

        cols = [np.ones((T, 1)), X]
        if D is not None:
            cols.append(D)
        if n_ar > 0:
            cols.append(Y_ar)
        A = np.concatenate(cols, axis=1)

        n_free = A.shape[1]
        if T < n_free:
            raise ValueError(
                f"Insufficient observations ({T}) for {n_free} parameters."
            )

        beta_hat, *_ = np.linalg.lstsq(A, y, rcond=None)
        fitted_values = A @ beta_hat
        residuals = y - fitted_values

        intercept = float(beta_hat[0])
        coef = beta_hat[1 : 1 + self.n_lags]
        n_d = D.shape[1] if D is not None else 0
        gamma = (
            beta_hat[1 + self.n_lags : 1 + self.n_lags + n_d]
            if D is not None
            else np.array([])
        )
        phi = beta_hat[1 + self.n_lags + n_d :] if n_ar > 0 else np.array([])

        return FittedOLS(
            intercept=intercept,
            coef=coef,
            gamma=gamma,
            phi=phi,
            fitted_values=fitted_values,
            residuals=residuals,
            nobs=T,
        )

    @legacy_forecast_alias
    def forecast(
        self,
        regressors_forecast: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute one out-of-sample point forecast per fitted horizon.

        Direct-forecasting date convention (see `MIDAS.forecast()`):
        with ``T_X`` the latest available regressor date and the model
        fitted as ``y[t+h] ~ X[t]``, the horizon-h forecast targets
        ``forecast_date = T_X + h``.

        Parameters
        ----------
        regressors_forecast : pd.DataFrame
            Quarterly regressor data with ``date`` and ``value`` columns.

        Returns
        -------
        pd.DataFrame
            Long-format forecasts with columns ``date``, ``horizon``,
            ``spec``, ``value`` — one row per fitted horizon.  ``spec`` is
            the regressor ``variable`` value, else ``"target"``.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self.fits_:
            raise RuntimeError("Not fitted.  Call fit() first.")

        # Date of the latest finite regressor (``T_X``); horizon-h forecasts
        # the target at ``T_X + h``.
        forecast_origin = _latest_finite_quarterly_date(regressors_forecast)

        x_date = np.array([np.datetime64(forecast_origin, "ns")])
        x_row = _build_quarterly_lag_matrix(
            x_date, regressors_forecast, self.n_lags, self.start_lag
        )

        y_ar = np.array([])
        if self.n_ar_lags > 0:
            past = self.target_["value"].to_numpy(dtype=float)
            if len(past) >= self.n_ar_lags:
                y_ar = past[-self.n_ar_lags :][::-1]
            else:
                y_ar = np.full(self.n_ar_lags, np.nan)

        rows: list[dict] = []
        base_period = forecast_origin.to_period("Q")
        for h, fit in sorted(self.fits_.items()):
            forecast_date = (base_period + h).end_time.normalize()
            if not _fit_parameters_are_finite(fit) or np.any(~np.isfinite(x_row)):
                forecast_val = np.nan
            else:
                forecast_val = float(fit.intercept + x_row[0] @ fit.coef)
                if len(fit.gamma) > 0 and self.dummy_periods is not None:
                    d_row = _build_dummy_matrix(
                        forecast_date, dummy_quarters=self.dummy_periods
                    )
                    forecast_val += float(d_row[0] @ fit.gamma)
                if len(fit.phi) > 0:
                    if np.any(~np.isfinite(y_ar)):
                        forecast_val = np.nan
                    else:
                        forecast_val += float(y_ar @ fit.phi)
                if not np.isfinite(forecast_val):
                    forecast_val = np.nan
            rows.append(
                {
                    "date": forecast_date,
                    "horizon": h,
                    "spec": self._variable_name,
                    "value": forecast_val,
                }
            )

        self.forecasts_df_ = pd.DataFrame(
            rows, columns=["date", "horizon", "spec", "value"]
        )
        return self.forecasts_df_

    # -- Forecast decomposition -------------------------------------------------

    def forecast_decomp(
        self,
        regressors_forecast: pd.DataFrame,
        regressor_name: str = "X",
    ) -> pd.DataFrame:
        """Additive component decomposition of each out-of-sample forecast.

        Splits every horizon's point forecast into additive components that
        sum back to the forecast value produced by `forecast()`:

            forecast[h] = intercept + sum_k coef_k * X[t-k] + dummies + AR-lags

        Each quarterly lag is reported as its own component with its scalar
        coefficient in ``weight``.  When ``n_lags == 1``, the component is
        named after the regressor.  Intercept, dummy, and AR components also
        include their scalar coefficient in ``weight``.

        Parameters
        ----------
        regressors_forecast : pd.DataFrame
            Quarterly regressor data with ``date`` and ``value`` columns
            (same input as `forecast()`).
        regressor_name : str
            Component label for the regressor block (default ``"X"``).

        Returns
        -------
        pd.DataFrame
            Long-format decomposition with columns ``horizon``, ``date``,
            ``component``, ``contribution`` and ``weight``.  Contributions
            sum to the forecast value at every horizon.  Horizons whose
            forecast is undefined (e.g. missing lags or AR lags) emit no rows.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self.fits_:
            raise RuntimeError("Not fitted.  Call fit() first.")

        # Identical date convention to forecast(): the latest finite
        # regressor date ``T_X``; horizon-h forecasts target ``T_X + h``.
        forecast_origin = _latest_finite_quarterly_date(regressors_forecast)

        x_date = np.array([np.datetime64(forecast_origin, "ns")])
        x_row = _build_quarterly_lag_matrix(
            x_date, regressors_forecast, self.n_lags, self.start_lag
        )

        # AR predictors (mirror forecast()).
        y_ar = np.array([])
        if self.n_ar_lags > 0:
            past = self.target_["value"].to_numpy(dtype=float)
            if len(past) >= self.n_ar_lags:
                y_ar = past[-self.n_ar_lags :][::-1]
            else:
                y_ar = np.full(self.n_ar_lags, np.nan)

        rows: list[dict] = []
        base_period = forecast_origin.to_period("Q")
        for h, fit in sorted(self.fits_.items()):
            forecast_date = (base_period + h).end_time.normalize()

            # If a required block is unavailable, forecast() returns NaN for
            # this horizon; emit no decomposition rows for it.
            if not _fit_parameters_are_finite(fit) or np.any(~np.isfinite(x_row)):
                continue
            if len(fit.phi) > 0 and np.any(~np.isfinite(y_ar)):
                continue

            horizon_rows: list[dict] = [
                {
                    "horizon": h,
                    "date": forecast_date,
                    "component": "intercept",
                    "contribution": float(fit.intercept),
                    "weight": 1.0,
                }
            ]

            # Quarterly regressor lags (one genuine-linear component per lag).
            for k, c in enumerate(fit.coef):
                lag = self.start_lag + k
                name = (
                    regressor_name if self.n_lags == 1 else f"{regressor_name}_lag{lag}"
                )
                horizon_rows.append(
                    {
                        "horizon": h,
                        "date": forecast_date,
                        "component": name,
                        "contribution": float(x_row[0, k] * c),
                        "weight": float(c),
                    }
                )

            # Dummy components (one per active dummy).
            if len(fit.gamma) > 0 and self.dummy_periods is not None:
                d_row = _build_dummy_matrix(
                    forecast_date, dummy_quarters=self.dummy_periods
                )
                for i, g in enumerate(fit.gamma):
                    contrib = float(d_row[0, i] * g)
                    if contrib != 0.0:
                        horizon_rows.append(
                            {
                                "horizon": h,
                                "date": forecast_date,
                                "component": f"dummy_{i}",
                                "contribution": contrib,
                                "weight": float(g),
                            }
                        )

            # AR components (one per lag).
            for k, p in enumerate(fit.phi):
                horizon_rows.append(
                    {
                        "horizon": h,
                        "date": forecast_date,
                        "component": f"ar_lag{k + 1}",
                        "contribution": float(y_ar[k] * p),
                        "weight": float(p),
                    }
                )

            rows.extend(horizon_rows)

        return pd.DataFrame(
            rows, columns=["horizon", "date", "component", "contribution", "weight"]
        )
