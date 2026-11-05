"""MIDAS regression models and fitted result objects."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ._compat import legacy_forecast_alias
from .plots.midas import _MIDASPlots
from .temporal_weights import _weight_jacobian, get_weights
from .utils import (
    _build_ar_lag_matrix,
    _build_dummy_matrix,
    _build_horizon_lag_matrix,
    _build_lag_matrix,
)

__all__ = ["MIDAS", "FittedMidas"]

_METHODS = ("exp_almon", "beta", "almon", "unrestricted")
_OLS_METHODS = ("unrestricted", "almon")


def _latest_finite_regressor_date(regressors: pd.DataFrame) -> pd.Timestamp:
    """Return the date of the latest regressor with a finite value."""
    finite = regressors.loc[np.isfinite(regressors["value"]), "date"]
    if finite.empty:
        raise ValueError("regressors contain no finite observations")
    return pd.Timestamp(finite.max())


class MIDAS(_MIDASPlots):
    """
    MIDAS regression.

    Parameters
    ----------
    method : str
        Weighting scheme (default ``'almon'``).
    n_lags : int
        Number of high-frequency lags (default 6).
    n_pars_weights : int
        Weight-shape parameters for exp_almon/almon (default 2).
    estimator : str | None
        ``'ols'`` or ``'nls'``.  Defaults to ``'ols'`` for
        unrestricted/almon, ``'nls'`` otherwise.
    horizons : list[int] | None
        Explicit list of forecast horizons to fit, e.g. ``[0, 1, 4]``.
        ``None`` (default) is treated as ``[0]``, i.e. nowcast only / no
        multi-step forecasts.  This is a *list of horizon indices*, unlike
        ``MidasCombo(horizons: int)``, which is a *count* of application
        forecast steps.  The MIDAS model uses a direct forecasting approach,
        so each horizon is estimated as a separate model with the target
        variable shifted accordingly y[t+h] ~ X[t]  (direct h-step
        forecasting). Negative horizons are supported: ``h=-1`` pairs
        each target with the following quarter's indicator window, allowing
        October data to estimate Q3. Dummies and, for negative horizons,
        AR lags remain aligned to the target quarter.
    start_lag : int
        Index of the first high-frequency lag to include (default 0).
    n_ar_lags : int
        Number of autoregressive lags of the dependent variable to include
        as additional regressors (default 0 = no AR terms).  When > 0, the
        model becomes ``y[t+h] = alpha + beta * X[t]'w + gamma'D[t+h]
        + sum_{k=1..p} phi_k * y[t+h-k] + eps`` with ``p = n_ar_lags``.
    dummy_periods : list[pd.Timestamp] | None
        Quarter-end dates to include as dummy variables (default ``None``).

    Raises
    ------
    ValueError
        If a model setting is invalid.

    """

    def __init__(
        self,
        method: str = "almon",
        n_lags: int = 6,
        n_pars_weights: int = 2,
        estimator: str | None = None,
        horizons: list[int] | None = None,
        start_lag: int = 0,
        n_ar_lags: int = 0,
        dummy_periods: list[pd.Timestamp] | None = None,
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got '{method}'")
        if n_lags < 1:
            raise ValueError("n_lags must be >= 1")
        if n_ar_lags < 0:
            raise ValueError("n_ar_lags must be >= 0")

        self.method = method
        self.n_lags = n_lags
        self.n_pars_weights = n_pars_weights
        self.n_ar_lags = n_ar_lags
        self.dummy_periods = dummy_periods

        if estimator is None:
            estimator = "ols" if method in _OLS_METHODS else "nls"
        elif estimator == "ols" and method not in _OLS_METHODS:
            raise ValueError(f"OLS only valid for {_OLS_METHODS}, not '{method}'.")

        if horizons is not None and (
            not isinstance(horizons, list)
            or not all(isinstance(h, int) for h in horizons)
        ):
            raise ValueError("horizons must be a list of integers.")

        self.horizons = horizons if horizons is not None else [0]
        self.estimator = estimator
        self.start_lag = start_lag

        # Populated by fit() — one FittedMidas per horizon
        self.fits_: dict[int, FittedMidas] = {}
        self.fits_df_: pd.DataFrame | None = None
        # Indicator name reported in the ``spec`` column of ``forecast()``.
        self._variable_name: str = "target"

        # Populated by forecast() (DataFrame, multi-horizon)
        self.forecasts_df_: pd.DataFrame | None = None

    # -- Fitting -----------------------------------------------------------
    def fit(
        self,
        target: pd.DataFrame,
        regressors: pd.DataFrame,
    ) -> MIDAS:
        """Fit the MIDAS model from target and regressor DataFrames.

        Parameters
        ----------
        target : pd.DataFrame
            Low-frequency target with (at least) ``date`` and ``value``
            columns.  Column order does not matter and extra columns are
            ignored.
        regressors : pd.DataFrame
            High-frequency (monthly) regressor for a single indicator, with
            (at least) ``date`` and ``value`` columns.  Column order does not
            matter and extra columns are ignored.

        Returns
        -------
        self : MIDAS

        Raises
        ------
        ValueError
            If either input is missing a required column, the target has
            missing values, or a horizon is unsupported.
        """

        if not {"date", "value"}.issubset(target.columns):
            raise ValueError("target DataFrame must have 'date' and 'value' columns.")

        if not {"date", "value"}.issubset(regressors.columns):
            raise ValueError(
                "regressors DataFrame must have 'date' and 'value' columns."
            )

        # Name used in the ``spec`` column of ``forecast()``.  Taken from the
        # regressor ``variable`` column when present, else ``"target"``.
        if "variable" in regressors.columns and regressors["variable"].notna().any():
            self._variable_name = str(regressors["variable"].iloc[0])
        else:
            self._variable_name = "target"

        target = target.sort_values("date")
        regressors = regressors.sort_values("date")
        self.target_ = target.reset_index(drop=True)

        target_dates = pd.to_datetime(target["date"])

        # Require finite target values.
        if target["value"].isna().any():
            raise ValueError("target contains NaN values")

        target_dates_np = target_dates.values
        y = target["value"].to_numpy(dtype=float)

        X = _build_lag_matrix(target["date"], regressors, self.n_lags, self.start_lag)

        D = None
        if self.dummy_periods is not None:
            D = _build_dummy_matrix(target_dates_np, dummy_quarters=self.dummy_periods)

        # AR lag matrix of the target (empty when n_ar_lags == 0).
        Y_ar = _build_ar_lag_matrix(y, self.n_ar_lags)

        # Keep rows with complete monthly and autoregressive lag blocks.
        valid = ~np.any(np.isnan(X), axis=1)
        if self.n_ar_lags > 0:
            valid &= ~np.any(np.isnan(Y_ar), axis=1)
        self.valid_mask_ = valid

        # Fit one model per horizon: y[t+h] ~ X[t]  (direct h-step forecasting)
        T = len(y)
        max_horizon = max(self.horizons)
        if max_horizon >= T:
            raise ValueError(
                f"Maximum horizon {max_horizon} exceeds number of observations {T}."
            )

        self.fits_ = {}
        all_dates = []
        all_horizons = []
        all_values = []

        for h in self.horizons:
            if h < 0:
                X_origin = _build_horizon_lag_matrix(
                    target_dates_np, regressors, self.n_lags, self.start_lag, h
                )
                source_valid = np.all(np.isfinite(X_origin), axis=1)
                source_valid &= np.all(np.isfinite(Y_ar), axis=1)
                X_h = X_origin[source_valid]
                y_h = y[source_valid]
                D_h = D[source_valid] if D is not None else None
                Y_ar_h = Y_ar[source_valid]
                dates_h = target_dates_np[source_valid]
            else:
                source_valid = valid[: T - h]
                X_h = X[: T - h][source_valid]
                y_h = y[h:][source_valid]
                D_h = D[h:][source_valid] if D is not None else None
                Y_ar_h = Y_ar[: T - h][source_valid]
                dates_h = target_dates_np[h:][source_valid]
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
    ) -> FittedMidas:
        """Fit one horizon from pre-built NumPy arrays.

        Parameters
        ----------
        y : np.ndarray
            Low-frequency target variable.
        X : np.ndarray
            High-frequency lag matrix.
        D : np.ndarray | None
            Dummy matrix at target frequency.
        Y_ar : np.ndarray | None
            Autoregressive lag matrix of the target.  ``None`` is treated
            as an empty ``(T, 0)`` block (no AR terms).

        Returns
        -------
        FittedMidas

        Raises
        ------
        ValueError
            If *X* has an unexpected number of columns or its row count
            does not match *y*.
        """
        y = np.asarray(y, dtype=float).ravel()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        T, n_regressors = X.shape

        if n_regressors != self.n_lags:
            raise ValueError(f"X has {n_regressors} cols, expected {self.n_lags}.")
        if len(y) != T:
            raise ValueError("y and X row counts differ.")

        if Y_ar is None:
            Y_ar = np.empty((T, 0))

        if self.estimator == "ols":
            alpha, beta, theta, weights, A, gamma, phi = self._fit_ols(y, X, D, Y_ar)
        else:
            alpha, beta, theta, weights, gamma, phi = self._fit_nls(y, X, D, Y_ar)
            A = None

        fitted_values = alpha + X @ np.asarray(beta * weights)
        if len(gamma) > 0:
            fitted_values += D @ gamma
        if len(phi) > 0:
            fitted_values += Y_ar @ phi
        residuals = y - fitted_values

        fit = FittedMidas(
            alpha=alpha,
            beta=beta,
            theta=theta,
            weights=weights,
            A=A,
            fitted_values=fitted_values,
            residuals=residuals,
            nobs=T,
            y=y,
            X=X,
            gamma=gamma,
            phi=phi,
        )
        return fit

    def _fit_ols(self, y: np.ndarray, X: np.ndarray, D=None, Y_ar=None) -> tuple:
        """Fit an unrestricted or Almon model with OLS.

        Returns (alpha, beta, theta, weights, A, gamma, phi).
        """
        ones = np.ones((len(y), 1))

        if self.method == "unrestricted":
            A = np.hstack([ones, X])
        else:
            j = np.arange(self.n_lags, dtype=float)
            V = np.column_stack([j**i for i in range(self.n_pars_weights)])
            A = np.hstack([ones, X @ V])

        if D is not None:
            A = np.hstack([A, D])
        n_ar = 0 if Y_ar is None else Y_ar.shape[1]
        if n_ar > 0:
            A = np.hstack([A, Y_ar])

        phi_hat, *_ = np.linalg.lstsq(A, y, rcond=None)

        alpha = float(phi_hat[0])
        beta = 1.0

        n_weight_pars = (
            self.n_lags if self.method == "unrestricted" else self.n_pars_weights
        )
        n_d = D.shape[1] if D is not None else 0
        theta = phi_hat[1 : 1 + n_weight_pars]
        weights = theta.copy() if self.method == "unrestricted" else V @ theta
        gamma = phi_hat[1 + n_weight_pars : 1 + n_weight_pars + n_d]
        phi = phi_hat[1 + n_weight_pars + n_d :] if n_ar > 0 else np.array([])
        return alpha, beta, theta, weights, A, gamma, phi

    def _fit_nls(self, y: np.ndarray, X: np.ndarray, D=None, Y_ar=None) -> tuple:
        """Fit a nonlinear MIDAS model.

        Returns (alpha, beta, theta, weights, gamma, phi).
        """
        n_dummies = D.shape[1] if D is not None else 0
        n_ar = 0 if Y_ar is None else Y_ar.shape[1]

        n_lags = self.n_lags
        n_theta = n_lags if self.method == "unrestricted" else self.n_pars_weights
        # Parameter layout: [alpha, beta, theta..., gamma..., phi...]
        gamma_start = 2 + n_theta
        phi_start = gamma_start + n_dummies
        ones = np.ones(len(y))

        def residuals_np(pars):
            """Return residuals for a parameter vector."""
            theta_pars = pars[2 : 2 + n_theta]
            w = get_weights(self.method, theta_pars, n_lags)
            fitted = pars[0] + pars[1] * (X @ w)
            if n_dummies > 0:
                fitted = fitted + D @ pars[gamma_start:phi_start]
            if n_ar > 0:
                fitted = fitted + Y_ar @ pars[phi_start:]
            r = y - fitted
            return np.where(np.isfinite(r), r, 1e3)

        def jac_np(pars):
            """Return the residual Jacobian for a parameter vector."""
            theta_pars = pars[2 : 2 + n_theta]
            w = get_weights(self.method, theta_pars, n_lags)
            dw = _weight_jacobian(self.method, theta_pars, n_lags)  # (n_lags, n_theta)

            cols = [
                -ones,  # d/d alpha
                -(X @ w),  # d/d beta
                -pars[1] * (X @ dw),  # d/d theta (n, n_theta)
            ]
            if n_dummies > 0:
                cols.append(-D)
            if n_ar > 0:
                cols.append(-Y_ar)
            J = np.column_stack(cols)
            return np.where(np.isfinite(J), J, 0.0)

        # Initial parameters: [alpha, beta, theta..., gamma..., phi...]
        if self.method == "beta":
            init_params = np.array([0.0, 1.0, 1.0, 2.0])
        elif self.method == "almon":
            theta0 = np.zeros(self.n_pars_weights)
            theta0[0] = 1.0
            init_params = np.concatenate([[0.0, 1.0], theta0])
        else:
            n = n_lags if self.method == "unrestricted" else self.n_pars_weights
            init_params = np.concatenate([[0.0, 1.0], np.zeros(n)])

        if n_dummies > 0:
            init_params = np.concatenate([init_params, np.zeros(n_dummies)])
        if n_ar > 0:
            init_params = np.concatenate([init_params, np.zeros(n_ar)])

        result = least_squares(
            residuals_np,
            init_params,
            jac=jac_np,
            method="lm",  # Levenberg-Marquardt
        )

        alpha = float(result.x[0])
        beta = float(result.x[1])
        theta = np.array(result.x[2 : 2 + n_theta])
        weights = np.asarray(get_weights(self.method, theta, self.n_lags))
        gamma = (
            np.array(result.x[gamma_start:phi_start]) if n_dummies > 0 else np.array([])
        )
        phi = np.array(result.x[phi_start:]) if n_ar > 0 else np.array([])
        return alpha, beta, theta, weights, gamma, phi

    # -- Forecast ---------------------------------------------------------------

    @legacy_forecast_alias
    def forecast(
        self,
        regressors_forecast: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute one out-of-sample point forecast per fitted horizon.

        Notes
        -----
        Direct-forecasting date convention:
        Let ``T_X`` be the date of the **latest finite regressor observation**
        (trailing rows with missing values do not advance the information
        date).  The fitted relation is ``y[t+h] ~ X[t]``.  Evaluating
        the horizon-``h`` model on ``X`` at ``T_X`` therefore forecasts the
        target ``h`` periods after ``T_X``:

            forecast_date(h) = T_X + h    (in quarters)

        Equivalently, if ``T`` is the date of the last observed ``y`` and
        ``step = forecast_date - T``, then the horizon needed to reach
        ``y[T+step]`` from ``X[T_X]`` is ``h = (T + step) - T_X``.  So a
        nowcast (``h=0``) of ``y[T+1]`` is only possible when ``X[T+1]``
        is available (``T_X = T+1``).

        Parameters
        ----------
        regressors_forecast : pd.DataFrame
            Monthly regressor data with ``date`` and ``value`` columns.

        Returns
        -------
        pd.DataFrame
            Long-format forecasts with columns ``date``, ``horizon``,
            ``spec``, ``value`` — one row per fitted horizon.  ``spec`` is
            the indicator name (the regressor ``variable`` value, else
            ``"target"``).

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self.fits_:
            raise RuntimeError("Not fitted.  Call fit() first.")

        # Date of the latest available regressor (``T_X``).  Direct
        # forecasting fits ``y[t+h] ~ X[t]``, so the horizon-h model
        # evaluated on X at T_X forecasts the target at ``T_X + h``.
        forecast_origin = _latest_finite_regressor_date(regressors_forecast)

        x_row = _build_lag_matrix(
            forecast_origin, regressors_forecast, self.n_lags, self.start_lag
        )

        # AR predictors: the p most recent target values from the training data.
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
            w = np.asarray(fit.beta * fit.weights)
            forecast_val = float(fit.alpha)

            # Add MIDAS regressor block when available.
            if not np.any(np.isnan(x_row)):
                forecast_val += float(x_row[0] @ w)

            # Add dummy contribution if model was fitted with dummies.
            gamma = fit.gamma
            if len(gamma) > 0 and self.dummy_periods is not None:
                d_row = _build_dummy_matrix(
                    forecast_date,
                    dummy_quarters=self.dummy_periods,
                )
                forecast_val += float(d_row[0] @ gamma)

            # Add AR contribution.
            phi = fit.phi
            if len(phi) > 0:
                if np.any(np.isnan(y_ar)):
                    forecast_val = np.nan
                else:
                    forecast_val += float(y_ar @ phi)

            # store
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

            forecast[h] = intercept + MIDAS-block + dummies + AR-lags

        The MIDAS regressor block ``x'(beta * w)`` is reported as a single
        component (the weighted high-frequency regressor) with ``weight``
        set to ``NaN``.  The intercept, dummy, and AR components carry their
        scalar coefficient as ``weight``.

        Parameters
        ----------
        regressors_forecast : pd.DataFrame
            Monthly regressor data with ``date`` and ``value`` columns
            (same input as `forecast()`).
        regressor_name : str
            Component label for the MIDAS regressor block (default ``"X"``).

        Returns
        -------
        pd.DataFrame
            Long-format decomposition with columns ``horizon``, ``date``,
            ``component``, ``contribution`` and ``weight``.  Contributions
            sum to the forecast value at every horizon.  Horizons whose
            forecast is undefined (e.g. missing AR lags) emit no rows.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self.fits_:
            raise RuntimeError("Not fitted.  Call fit() first.")

        # Identical date convention to forecast(): the latest available
        # regressor date ``T_X``; horizon-h forecasts target ``T_X + h``.
        forecast_origin = _latest_finite_regressor_date(regressors_forecast)

        x_row = _build_lag_matrix(
            forecast_origin, regressors_forecast, self.n_lags, self.start_lag
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
            phi = fit.phi

            # If the AR block is required but unavailable, forecast() returns
            # NaN for this horizon; emit no decomposition rows for it.
            if len(phi) > 0 and np.any(np.isnan(y_ar)):
                continue

            horizon_rows: list[dict] = [
                {
                    "horizon": h,
                    "date": forecast_date,
                    "component": "intercept",
                    "contribution": float(fit.alpha),
                    "weight": 1.0,
                }
            ]

            # MIDAS regressor block (single weighted-regressor component).
            if not np.any(np.isnan(x_row)):
                w = np.asarray(fit.beta * fit.weights)
                horizon_rows.append(
                    {
                        "horizon": h,
                        "date": forecast_date,
                        "component": regressor_name,
                        "contribution": float(x_row[0] @ w),
                        "weight": np.nan,
                    }
                )

            # Dummy components (one per active dummy).
            gamma = fit.gamma
            if len(gamma) > 0 and self.dummy_periods is not None:
                d_row = _build_dummy_matrix(
                    forecast_date, dummy_quarters=self.dummy_periods
                )
                for i, g in enumerate(gamma):
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
            for k, p in enumerate(phi):
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

    # -- Summary ----------------------------------------------------------------

    def summary(self, horizon: int | None = None) -> str:
        """Print a formatted text summary of the fitted model and return it.

        Parameters
        ----------
        horizon : int | None
            Which horizon to summarise.  When ``None`` (default) all
            fitted horizons are printed in sequence.

        Returns
        -------
        str
            The formatted summary text (also printed to stdout).

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self.fits_:
            raise RuntimeError("Not fitted. Call fit() first.")

        horizons = [horizon] if horizon is not None else sorted(self.fits_.keys())
        text = "\n".join(self._summary_single(h) for h in horizons)
        print(text)
        return text

    def _summary_single(self, horizon: int) -> str:
        """Build the summary text for a single horizon."""
        fit = self.fits_[horizon]
        sse = float(fit.residuals @ fit.residuals)
        rmse = float(np.sqrt(sse / fit.nobs))
        est = "scipy LM + analytic Jacobian" if self.estimator == "nls" else "OLS"

        sep = "=" * 52
        thin = "-" * 52

        lines = [
            sep,
            "         MIDAS Regression Results         ",
            sep,
            f"  Method / Estimator : {self.method} / {est}",
            f"  Lags (n_lags)      : {self.n_lags}",
            f"  Horizon            : {horizon}",
            f"  Observations       : {fit.nobs}",
            thin,
            f"  alpha : {fit.alpha: .6f}",
        ]

        if self.method != "unrestricted":
            lines.append(f"  beta  : {fit.beta: .6f}")

        for i, t in enumerate(fit.theta):
            lines.append(f"  theta[{i}] = {t: .6f}")

        for i, w in enumerate(fit.weights):
            lines.append(f"  w[{i:2d}]    = {w: .6f}")

        for i, g in enumerate(fit.gamma):
            lines.append(f"  gamma[{i}] = {g: .6f}")

        for i, p in enumerate(fit.phi):
            lines.append(f"  phi[{i + 1}]   = {p: .6f}")

        lines.extend([thin, f"  SSE  : {sse: .6f}", f"  RMSE : {rmse: .6f}", sep])

        return "\n".join(lines)


@dataclass
class FittedMidas:
    """Stores the fitted parameters and diagnostics for a single
    MIDAS regression at a specific horizon.

    Attributes
    ----------
    alpha : float
        Estimated intercept.
    beta : float
        Estimated slope on the weighted regressor (1.0 for OLS).
    theta : np.ndarray
        Weight-shape parameters.
    weights : np.ndarray
        Evaluated lag weights.
    A : np.ndarray | None
        OLS design matrix (``None`` for NLS).
    fitted_values : pd.Series
        In-sample fitted values with DatetimeIndex.
    residuals : np.ndarray
        In-sample residuals.
    nobs : int
        Number of observations used in estimation.
    y : np.ndarray
        Target vector used in estimation.
    X : np.ndarray
        Regressor matrix used in estimation.
    dates : np.ndarray | None
        Low-frequency target dates aligned to the fitted sample.
        Populated by ``fit()``; equivalent to ``fitted_values.index``.
    gamma : np.ndarray
        Estimated outlier-dummy coefficients (empty when no ``dummy_periods``).
    phi : np.ndarray
        Estimated autoregressive coefficients (empty when ``n_ar_lags == 0``).
    """

    alpha: float = 0.0
    beta: float = 1.0
    theta: np.ndarray = field(default_factory=lambda: np.array([]))
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    A: np.ndarray | None = None
    fitted_values: pd.Series = field(default_factory=pd.Series)
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    nobs: int = 0
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    X: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    dates: np.ndarray | None = None
    gamma: np.ndarray = field(default_factory=lambda: np.array([]))
    phi: np.ndarray = field(default_factory=lambda: np.array([]))
