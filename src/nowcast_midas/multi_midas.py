"""Multi-regressor MIDAS models and fitted result objects."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ._compat import legacy_forecast_alias
from .specs import VariableSpec
from .temporal_weights import get_weights
from .utils import (
    _build_ar_lag_matrix,
    _build_dummy_matrix,
    _build_horizon_lag_matrix,
    _build_lag_matrix,
    _build_quarterly_lag_matrix,
)

__all__ = ["FittedMultiMidas", "MultiMIDAS", "VariableFit"]

jax.config.update("jax_enable_x64", True)

_METHODS = ("exp_almon", "beta", "almon", "unrestricted")
_OLS_METHODS = ("unrestricted", "almon")


def _latest_finite_date(data: pd.DataFrame) -> pd.Timestamp | None:
    finite = data.loc[np.isfinite(data["value"]), "date"]
    return pd.Timestamp(finite.max()) if not finite.empty else None


def _fit_parameters_are_finite(fit: FittedMultiMidas) -> bool:
    arrays = [
        np.asarray([fit.alpha]),
        np.asarray(fit.gamma),
        np.asarray(fit.phi),
    ]
    arrays.extend(
        np.asarray(values)
        for variable_fit in fit.variable_fits.values()
        for values in (variable_fit.beta, variable_fit.theta, variable_fit.weights)
    )
    return all(np.all(np.isfinite(array)) for array in arrays)


def _forecast_context(
    regressors: pd.DataFrame,
    specs: list[VariableSpec],
) -> tuple[pd.Timestamp, dict[str, pd.DataFrame]]:
    """Return capped variable data and the shared latest finite origin."""
    data_by_variable: dict[str, pd.DataFrame] = {}
    latest_dates: list[pd.Timestamp] = []

    for spec in specs:
        data = regressors.loc[
            regressors["variable"] == spec.variable, ["date", "value"]
        ].copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values("date")
        data_by_variable[spec.variable] = data
        latest = _latest_finite_date(data)
        if latest is not None:
            latest_dates.append(latest)

    if latest_dates:
        common_origin = max(latest_dates)
    else:
        common_origin = pd.Timestamp(pd.to_datetime(regressors["date"]).max())

    for variable, data in data_by_variable.items():
        data_by_variable[variable] = data.loc[data["date"] <= common_origin]
    return common_origin, data_by_variable


class MultiMIDAS:
    """Multi-regressor MIDAS regression.

    Extends MIDAS regression to multiple high-frequency (monthly)
    regressors.  Each regressor can use either a shared weighting method
    or its own per-variable specification via
    `VariableSpec`.

    Parameters
    ----------
    variables : list[str | VariableSpec]
        Regressors to include.  Pass a plain string to use the shared
        defaults; pass a `VariableSpec` to
        override any parameter for that regressor.  Use
        ``VariableSpec(..., frequency='QE')`` for quarterly regressors.
    method : str
        Shared weighting scheme for **monthly** variables given as plain
        strings (default ``'almon'``).  Ignored for quarterly regressors.
    n_lags : int
        Shared number of lags (default 3).  Monthly lags for monthly
        regressors, quarterly lags for quarterly regressors.
    n_pars_weights : int
        Shared weight-shape parameters for polynomial schemes (default 2).
    estimator : str | None
        Shared estimator override.  ``None`` (default) chooses
        automatically per variable based on method.
    horizons : list[int] | None
        Direct-forecast horizons (default ``[0]``). Negative horizons use
        regressor windows after the target quarter. Dummies and, for
        negative horizons, AR lags remain aligned to the target quarter.
    start_lag : int
        Shared starting lag index (default 0).
    n_ar_lags : int
        Number of AR lags of the target to include (default 0).
    dummy_periods : list[pd.Timestamp] | None
        Outlier-dummy quarters (default ``None``).

    Raises
    ------
    ValueError
        If a model setting or variable specification is invalid.

    Examples
    --------
    All regressors share the same method:

        mc = MultiMIDAS(["PMI", "IP", "CLI"], method="almon", n_lags=6)
        mc.fit(target, regressors)

    Per-variable specs with a quarterly regressor:

        mc = MultiMIDAS([
            VariableSpec("PMI",  method="exp_almon", n_lags=6),
            VariableSpec("IP",   method="almon",     n_lags=3),
            VariableSpec("UNEMP", frequency="QE",    n_lags=1),
        ])
        mc.fit(target, regressors)
    """

    def __init__(
        self,
        variables: list[str | VariableSpec],
        method: str = "almon",
        n_lags: int = 3,
        n_pars_weights: int = 2,
        estimator: str | None = None,
        horizons: list[int] | None = None,
        start_lag: int = 0,
        n_ar_lags: int = 0,
        dummy_periods: list[pd.Timestamp] | None = None,
    ) -> None:
        if not variables:
            raise ValueError("variables must be a non-empty list.")
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got '{method}'")
        if n_lags < 1:
            raise ValueError("n_lags must be >= 1")
        if n_ar_lags < 0:
            raise ValueError("n_ar_lags must be >= 0")

        self.specs = [
            copy.copy(s)
            for s in _resolve_specs(
                variables, method, n_lags, n_pars_weights, estimator, start_lag
            )
        ]

        # Validate and resolve the estimator for each specification.
        for spec in self.specs:
            spec.frequency = spec.frequency.upper()
            if spec.frequency not in ("ME", "QE"):
                raise ValueError(
                    f"VariableSpec frequency must be 'ME' or 'QE', "
                    f"got '{spec.frequency}' for variable '{spec.variable}'"
                )
            if spec.frequency == "QE":
                # Quarterly regressors enter linearly, so skip MIDAS checks.
                spec.estimator = "ols"
                continue
            if spec.method not in _METHODS:
                raise ValueError(
                    f"VariableSpec method must be one of {_METHODS}, "
                    f"got '{spec.method}' for variable '{spec.variable}'"
                )
            if spec.estimator is None:
                spec.estimator = "ols" if spec.method in _OLS_METHODS else "nls"
            elif spec.estimator == "ols" and spec.method not in _OLS_METHODS:
                raise ValueError(
                    f"OLS estimator is only valid for {_OLS_METHODS}, "
                    f"not '{spec.method}' (variable '{spec.variable}')."
                )

        # Reject duplicate variable names.
        var_names = [s.variable for s in self.specs]
        dupes = [v for v in var_names if var_names.count(v) > 1]
        if dupes:
            raise ValueError(f"Duplicate variable names: {sorted(set(dupes))}")

        # Separate monthly MIDAS and quarterly linear specifications.
        self.monthly_specs = [s for s in self.specs if s.frequency == "ME"]
        self.quarterly_specs = [s for s in self.specs if s.frequency == "QE"]

        # Use joint OLS when every monthly variable supports it.
        self._use_ols = all(spec.estimator == "ols" for spec in self.monthly_specs)

        if horizons is not None and (
            not isinstance(horizons, list)
            or not all(isinstance(h, int) for h in horizons)
        ):
            raise ValueError("horizons must be a list of integers.")

        self.horizons = horizons if horizons is not None else [0]
        self.n_ar_lags = n_ar_lags
        self.dummy_periods = dummy_periods

        # Populated by fit()
        self.fits_: dict[int, FittedMultiMidas] = {}
        self.valid_mask_: np.ndarray | None = None
        self.target_: pd.DataFrame | None = None
        self.forecasts_df_: pd.DataFrame | None = None

        # Name reported in the ``spec`` column of ``forecast()``.  A
        # MultiMIDAS forecast is a single joint prediction, so all variables
        # are folded into one label.
        self._spec_name = "+".join(s.variable for s in self.specs)

    # ---------------------------------------------------------------------- #
    #  fit                                                                     #
    # ---------------------------------------------------------------------- #

    def fit(self, target: pd.DataFrame, regressors: pd.DataFrame) -> MultiMIDAS:
        """Fit the multi-regressor MIDAS model.

        Parameters
        ----------
        target : pd.DataFrame
            Quarterly target with at least ``date`` and ``value`` columns.
        regressors : pd.DataFrame
            High-frequency regressors.  Must contain at least ``date``,
            ``variable``, and ``value`` columns.  Rows are filtered per
            variable name to build each regressor's lag matrix.

        Returns
        -------
        self : MultiMIDAS

        Raises
        ------
        ValueError
            If either input is missing required columns, a variable is
            absent, or no valid observations are available.
        """
        if not {"date", "value"}.issubset(target.columns):
            raise ValueError("target must have 'date' and 'value' columns.")
        if not {"date", "variable", "value"}.issubset(regressors.columns):
            raise ValueError(
                "regressors must have 'date', 'variable', and 'value' columns."
            )

        target = target.sort_values("date")
        self.target_ = target.reset_index(drop=True)

        target_dates_np = pd.to_datetime(target["date"]).values
        y = target["value"].to_numpy(dtype=float)
        T = len(y)

        # Build a lag matrix for each monthly MIDAS variable.
        X_list: list[np.ndarray] = []
        for spec in self.monthly_specs:
            var_data = regressors.loc[
                regressors["variable"] == spec.variable, ["date", "value"]
            ].sort_values("date")
            if var_data.empty:
                raise ValueError(f"Variable '{spec.variable}' not found in regressors.")
            X_list.append(
                _build_lag_matrix(target["date"], var_data, spec.n_lags, spec.start_lag)
            )

        # Build a lag matrix for each quarterly linear variable.
        Z_list: list[np.ndarray] = []
        for spec in self.quarterly_specs:
            var_data = regressors.loc[
                regressors["variable"] == spec.variable, ["date", "value"]
            ].sort_values("date")
            if var_data.empty:
                raise ValueError(f"Variable '{spec.variable}' not found in regressors.")
            Z_list.append(
                _build_quarterly_lag_matrix(
                    target["date"], var_data, spec.n_lags, spec.start_lag
                )
            )

        D = None
        if self.dummy_periods is not None:
            D = _build_dummy_matrix(target_dates_np, dummy_quarters=self.dummy_periods)

        Y_ar = _build_ar_lag_matrix(y, self.n_ar_lags)

        # Keep rows with finite lags for every variable.
        valid = np.isfinite(y)
        for X_k in X_list:
            valid &= np.all(np.isfinite(X_k), axis=1)
        for Z_k in Z_list:
            valid &= np.all(np.isfinite(Z_k), axis=1)
        if self.n_ar_lags > 0:
            valid &= np.all(np.isfinite(Y_ar), axis=1)
        self.valid_mask_ = valid

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
        for h in self.horizons:
            if h < 0:
                X_origin = []
                Z_origin = []
                for spec in self.specs:
                    var_data = regressors.loc[
                        regressors["variable"] == spec.variable, ["date", "value"]
                    ]
                    block = _build_horizon_lag_matrix(
                        target_dates_np,
                        var_data,
                        spec.n_lags,
                        spec.start_lag,
                        h,
                        quarterly=spec.frequency == "QE",
                    )
                    if spec.frequency == "QE":
                        Z_origin.append(block)
                    else:
                        X_origin.append(block)
                origin_valid = np.isfinite(y) & np.all(np.isfinite(Y_ar), axis=1)
                for block in X_origin + Z_origin:
                    origin_valid &= np.all(np.isfinite(block), axis=1)
                X_h = [block[origin_valid] for block in X_origin]
                Z_h = [block[origin_valid] for block in Z_origin]
                y_h = y[origin_valid]
                D_h = D[origin_valid] if D is not None else None
                Y_ar_h = Y_ar[origin_valid]
                dates_h = target_dates_np[origin_valid]
            else:
                origin_valid = valid[: T - h] & np.isfinite(y[h:])
                X_h = [X_k[: T - h][origin_valid] for X_k in X_list]
                Z_h = [Z_k[: T - h][origin_valid] for Z_k in Z_list]
                y_h = y[h:][origin_valid]
                D_h = D[h:][origin_valid] if D is not None else None
                Y_ar_h = Y_ar[: T - h][origin_valid]
                dates_h = target_dates_np[h:][origin_valid]
            if not len(y_h):
                raise ValueError(f"No valid observations for horizon {h}.")
            fit = self._fit_single_horizon(y_h, X_h, Z_h, D_h, Y_ar_h)
            fit.dates = dates_h
            self.fits_[h] = fit

        # Store fitted values in long format (date, horizon, value).
        all_dates = []
        all_horizons = []
        all_values = []
        for h in self.horizons:
            fit = self.fits_[h]
            all_dates.extend(fit.dates)
            all_horizons.extend([h] * len(fit.dates))
            all_values.extend(fit.fitted_values)
        self.fits_df_ = pd.DataFrame(
            {"date": all_dates, "horizon": all_horizons, "value": all_values}
        )

        return self

    # ---------------------------------------------------------------------- #
    #  Internal fit helpers                                                  #
    # ---------------------------------------------------------------------- #

    def _fit_single_horizon(
        self,
        y: np.ndarray,
        X_list: list[np.ndarray],
        Z_list: list[np.ndarray],
        D: np.ndarray | None,
        Y_ar: np.ndarray,
    ) -> FittedMultiMidas:
        """Fit one horizon from aligned regressor arrays."""
        if self._use_ols:
            return self._fit_ols(y, X_list, Z_list, D, Y_ar)
        return self._fit_nls(y, X_list, Z_list, D, Y_ar)

    def _fit_ols(
        self,
        y: np.ndarray,
        X_list: list[np.ndarray],
        Z_list: list[np.ndarray],
        D: np.ndarray | None,
        Y_ar: np.ndarray,
    ) -> FittedMultiMidas:
        """Fit all variables for one horizon with OLS."""
        T = len(y)
        cols = [np.ones((T, 1))]
        V_list: list[np.ndarray | None] = []
        for spec, X_k in zip(self.monthly_specs, X_list):
            if spec.method == "unrestricted":
                cols.append(X_k)
                V_list.append(None)
            else:  # almon
                j = np.arange(spec.n_lags, dtype=float)
                V_k = np.column_stack([j**i for i in range(spec.n_pars_weights)])
                cols.append(X_k @ V_k)
                V_list.append(V_k)

        # Add quarterly regressors as plain lag columns.
        cols.extend(Z_list)

        n_d = D.shape[1] if D is not None else 0
        n_ar = Y_ar.shape[1]
        if D is not None:
            cols.append(D)
        if n_ar > 0:
            cols.append(Y_ar)

        A = np.hstack(cols)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)

        alpha = float(coef[0])
        cursor = 1
        var_fits: dict[str, VariableFit] = {}
        for spec, V_k in zip(self.monthly_specs, V_list):
            n_pars = (
                spec.n_lags if spec.method == "unrestricted" else spec.n_pars_weights
            )
            theta = coef[cursor : cursor + n_pars]
            weights = theta.copy() if V_k is None else V_k @ theta
            var_fits[spec.variable] = VariableFit(
                beta=1.0, theta=theta, weights=weights
            )
            cursor += n_pars

        # Store one fitted coefficient for each quarterly lag.
        for spec in self.quarterly_specs:
            delta = coef[cursor : cursor + spec.n_lags]
            var_fits[spec.variable] = VariableFit(
                beta=1.0, theta=delta.copy(), weights=delta.copy()
            )
            cursor += spec.n_lags

        gamma = coef[cursor : cursor + n_d] if n_d > 0 else np.array([])
        cursor += n_d
        phi = coef[cursor:] if n_ar > 0 else np.array([])

        fitted_values = A @ coef
        residuals = y - fitted_values
        return FittedMultiMidas(
            alpha=alpha,
            variable_fits=var_fits,
            gamma=gamma,
            phi=phi,
            A=A,
            fitted_values=fitted_values,
            residuals=residuals,
            nobs=T,
            y=y,
        )

    def _fit_nls(
        self,
        y: np.ndarray,
        X_list: list[np.ndarray],
        Z_list: list[np.ndarray],
        D: np.ndarray | None,
        Y_ar: np.ndarray,
    ) -> FittedMultiMidas:
        """Fit all variables for one horizon with nonlinear least squares."""
        T = len(y)
        n_dummies = D.shape[1] if D is not None else 0
        n_ar = Y_ar.shape[1]

        # Split monthly variables into nonlinear and linear groups.
        nl_specs: list[VariableSpec] = []
        nl_X: list[np.ndarray] = []
        lin_cols: list[np.ndarray] = []
        # Store (variable, basis V or None) for each linear block.
        lin_meta: list[tuple[str, np.ndarray | None]] = []
        for spec, X_k in zip(self.monthly_specs, X_list):
            if spec.method in _OLS_METHODS:
                if spec.method == "unrestricted":
                    lin_cols.append(X_k)
                    lin_meta.append((spec.variable, None))
                else:  # almon
                    j = np.arange(spec.n_lags, dtype=float)
                    V_k = np.column_stack([j**i for i in range(spec.n_pars_weights)])
                    lin_cols.append(X_k @ V_k)
                    lin_meta.append((spec.variable, V_k))
            else:
                nl_specs.append(spec)
                nl_X.append(X_k)

        # Quarterly regressors also enter linearly.
        for spec, Z_k in zip(self.quarterly_specs, Z_list):
            lin_cols.append(Z_k)
            lin_meta.append((spec.variable, None))

        lin_widths = [c.shape[1] for c in lin_cols]
        n_linear = sum(lin_widths)
        Lin = np.hstack(lin_cols) if lin_cols else np.empty((T, 0))

        y_j = jnp.array(y)
        nlX_j = [jnp.array(X_k) for X_k in nl_X]
        Lin_j = jnp.array(Lin)
        D_j = jnp.array(D) if D is not None else None
        Yar_j = jnp.array(Y_ar) if n_ar > 0 else None

        # Record parameter block boundaries for nonlinear variables.
        # Each tuple stores (beta_idx, theta_start, n_theta).
        offsets: list[tuple[int, int, int]] = []
        cursor = 1  # index 0 is alpha
        for spec in nl_specs:
            n_theta = spec.n_pars_weights
            offsets.append((cursor, cursor + 1, n_theta))
            cursor += 1 + n_theta  # beta + theta block

        lin_start = cursor
        gamma_start = lin_start + n_linear
        phi_start = gamma_start + n_dummies

        def residuals_jax(pars: jnp.ndarray) -> jnp.ndarray:
            fitted = pars[0] * jnp.ones(T)
            for (b_idx, th_start, n_theta), spec, X_j in zip(offsets, nl_specs, nlX_j):
                w_k = get_weights(
                    spec.method, pars[th_start : th_start + n_theta], spec.n_lags
                )
                fitted = fitted + pars[b_idx] * (X_j @ w_k)
            if n_linear > 0:
                fitted = fitted + Lin_j @ pars[lin_start:gamma_start]
            if D_j is not None:
                fitted = fitted + D_j @ pars[gamma_start:phi_start]
            if Yar_j is not None:
                fitted = fitted + Yar_j @ pars[phi_start:]
            return y_j - fitted

        jac_jax = jax.jacobian(residuals_jax)

        def residuals_np(pars: np.ndarray) -> np.ndarray:
            r = residuals_jax(jnp.array(pars))
            return np.asarray(jnp.where(jnp.isfinite(r), r, 1e3), dtype=float)

        def jac_np(pars: np.ndarray) -> np.ndarray:
            J = jac_jax(jnp.array(pars))
            return np.asarray(jnp.where(jnp.isfinite(J), J, 0.0), dtype=float)

        # Build the initial parameter vector.
        init: list[float] = [0.0]  # alpha
        for spec in nl_specs:
            init.append(1.0)  # beta_k
            if spec.method == "beta":
                init.extend([1.0, 2.0])
            else:  # exp_almon
                init.extend([0.0] * spec.n_pars_weights)
        init.extend([0.0] * n_linear)  # linear coefficients
        init.extend([0.0] * n_dummies)
        init.extend([0.0] * n_ar)

        result = least_squares(
            residuals_np,
            np.array(init),
            jac=jac_np,
            method="lm",  # Levenberg-Marquardt
        )

        alpha = float(result.x[0])
        var_fits: dict[str, VariableFit] = {}
        for (b_idx, th_start, n_theta), spec in zip(offsets, nl_specs):
            theta_k = np.array(result.x[th_start : th_start + n_theta])
            weights_k = np.asarray(
                get_weights(spec.method, jnp.array(theta_k), spec.n_lags)
            )
            var_fits[spec.variable] = VariableFit(
                beta=float(result.x[b_idx]),
                theta=theta_k,
                weights=weights_k,
            )

        # Extract linear coefficients for monthly and quarterly variables.
        lin_coefs = result.x[lin_start:gamma_start]
        l_cursor = 0
        for (variable, V_k), width in zip(lin_meta, lin_widths):
            coefs = np.array(lin_coefs[l_cursor : l_cursor + width])
            weights = coefs.copy() if V_k is None else V_k @ coefs
            var_fits[variable] = VariableFit(
                beta=1.0, theta=coefs.copy(), weights=weights
            )
            l_cursor += width

        gamma = (
            np.array(result.x[gamma_start:phi_start]) if n_dummies > 0 else np.array([])
        )
        phi = np.array(result.x[phi_start:]) if n_ar > 0 else np.array([])

        # Recompute fitted values from the extracted parameters.
        fitted_values = np.full(T, alpha)
        for spec, X_k in zip(self.monthly_specs, X_list):
            vf = var_fits[spec.variable]
            fitted_values += vf.beta * (X_k @ vf.weights)
        for spec_q, Z_k in zip(self.quarterly_specs, Z_list):
            vf = var_fits[spec_q.variable]
            fitted_values += Z_k @ vf.weights
        if D is not None and len(gamma):
            fitted_values += D @ gamma
        if n_ar > 0 and len(phi):
            fitted_values += Y_ar @ phi
        residuals = y - fitted_values

        return FittedMultiMidas(
            alpha=alpha,
            variable_fits=var_fits,
            gamma=gamma,
            phi=phi,
            A=None,
            fitted_values=fitted_values,
            residuals=residuals,
            nobs=T,
            y=y,
        )

    # ---------------------------------------------------------------------- #
    #  forecast                                                                #
    # ---------------------------------------------------------------------- #

    @legacy_forecast_alias
    def forecast(
        self,
        regressors: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute out-of-sample forecasts for all fitted horizons.

        Parameters
        ----------
        regressors : pd.DataFrame
            High-frequency regressors with ``date``, ``variable``, and
            ``value`` columns.
        Returns
        -------
        pd.DataFrame
            Long-format forecasts with columns ``date``, ``horizon``,
            ``spec``, ``value`` — one row per fitted horizon.  ``spec`` is
            the ``"+"``-joined regressor variable names (a MultiMIDAS
            forecast is a single joint prediction).

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self.fits_:
            raise RuntimeError("Not fitted. Call fit() first.")

        forecast_origin, data_by_variable = _forecast_context(regressors, self.specs)

        # Build one lag row for each monthly MIDAS variable.
        x_rows: dict[str, np.ndarray] = {}
        for spec in self.monthly_specs:
            x_rows[spec.variable] = _build_lag_matrix(
                forecast_origin,
                data_by_variable[spec.variable],
                spec.n_lags,
                spec.start_lag,
            )

        # Build one lag row for each quarterly linear variable.
        z_rows: dict[str, np.ndarray] = {}
        for spec in self.quarterly_specs:
            z_rows[spec.variable] = _build_quarterly_lag_matrix(
                forecast_origin,
                data_by_variable[spec.variable],
                spec.n_lags,
                spec.start_lag,
            )

        # Build the autoregressive predictors.
        y_ar = np.array([])
        if self.n_ar_lags > 0:
            past = self.target_["value"].to_numpy(dtype=float)
            y_ar = (
                past[-self.n_ar_lags :][::-1]
                if len(past) >= self.n_ar_lags
                else np.full(self.n_ar_lags, np.nan)
            )

        rows: list[dict] = []
        base_period = forecast_origin.to_period("Q")
        for h, fit in sorted(self.fits_.items()):
            forecast_date = (base_period + h).end_time.normalize()
            complete = _fit_parameters_are_finite(fit)
            fc_val = float(fit.alpha)

            for spec in self.monthly_specs:
                vf = fit.variable_fits[spec.variable]
                x_row = x_rows[spec.variable]
                if np.any(~np.isfinite(x_row)):
                    complete = False
                else:
                    fc_val += float(vf.beta * (x_row[0] @ vf.weights))

            for spec in self.quarterly_specs:
                vf = fit.variable_fits[spec.variable]
                z_row = z_rows[spec.variable]
                if np.any(~np.isfinite(z_row)):
                    complete = False
                else:
                    fc_val += float(z_row[0] @ vf.weights)

            if len(fit.gamma) > 0 and self.dummy_periods is not None:
                d_row = _build_dummy_matrix(
                    forecast_date, dummy_quarters=self.dummy_periods
                )
                fc_val += float(d_row[0] @ fit.gamma)

            if len(fit.phi) > 0:
                if np.any(~np.isfinite(y_ar)):
                    complete = False
                else:
                    fc_val += float(y_ar @ fit.phi)

            if not complete or not np.isfinite(fc_val):
                fc_val = np.nan

            rows.append(
                {
                    "date": forecast_date,
                    "horizon": h,
                    "spec": self._spec_name,
                    "value": fc_val,
                }
            )

        self.forecasts_df_ = pd.DataFrame(
            rows, columns=["date", "horizon", "spec", "value"]
        )
        return self.forecasts_df_

    # ---------------------------------------------------------------------- #
    #  forecast decomposition                                                  #
    # ---------------------------------------------------------------------- #

    def forecast_decomp(
        self,
        regressors: pd.DataFrame,
    ) -> pd.DataFrame:
        """Additive component decomposition of each out-of-sample forecast.

        Splits every horizon's point forecast into additive components that
        sum back to the forecast value produced by `forecast()`:

            forecast[h] = intercept
                        + sum_k  monthly-MIDAS-block_k
                        + sum_j  quarterly-block_j
                        + dummies + AR-lags

        Each monthly/quarterly regressor block is reported as one component
        named after its variable.  The ``weight`` is ``NaN`` for weighted-lag
        blocks.  Intercept, dummy, and AR components carry their scalar
        coefficient as ``weight``.

        Parameters
        ----------
        regressors : pd.DataFrame
            High-frequency regressors with ``date``, ``variable`` and
            ``value`` columns (same input as `forecast()`).

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
            raise RuntimeError("Not fitted. Call fit() first.")

        forecast_origin, data_by_variable = _forecast_context(regressors, self.specs)

        # Build one lag row per monthly (MIDAS) variable.
        x_rows: dict[str, np.ndarray] = {}
        for spec in self.monthly_specs:
            x_rows[spec.variable] = _build_lag_matrix(
                forecast_origin,
                data_by_variable[spec.variable],
                spec.n_lags,
                spec.start_lag,
            )

        # Build one lag row per quarterly (linear) variable.
        z_rows: dict[str, np.ndarray] = {}
        for spec in self.quarterly_specs:
            z_rows[spec.variable] = _build_quarterly_lag_matrix(
                forecast_origin,
                data_by_variable[spec.variable],
                spec.n_lags,
                spec.start_lag,
            )

        # AR predictors (mirror forecast()).
        y_ar = np.array([])
        if self.n_ar_lags > 0:
            past = self.target_["value"].to_numpy(dtype=float)
            y_ar = (
                past[-self.n_ar_lags :][::-1]
                if len(past) >= self.n_ar_lags
                else np.full(self.n_ar_lags, np.nan)
            )

        rows: list[dict] = []
        base_period = forecast_origin.to_period("Q")
        for h, fit in sorted(self.fits_.items()):
            forecast_date = (base_period + h).end_time.normalize()

            incomplete = any(
                np.any(~np.isfinite(x_rows[spec.variable]))
                for spec in self.monthly_specs
            ) or any(
                np.any(~np.isfinite(z_rows[spec.variable]))
                for spec in self.quarterly_specs
            )
            if len(fit.phi) > 0:
                incomplete = incomplete or np.any(~np.isfinite(y_ar))
            incomplete = incomplete or not _fit_parameters_are_finite(fit)
            if incomplete:
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

            # Monthly MIDAS regressor blocks.
            for spec in self.monthly_specs:
                vf = fit.variable_fits[spec.variable]
                x_row = x_rows[spec.variable]
                if np.all(np.isfinite(x_row)):
                    horizon_rows.append(
                        {
                            "horizon": h,
                            "date": forecast_date,
                            "component": spec.variable,
                            "contribution": float(vf.beta * (x_row[0] @ vf.weights)),
                            "weight": np.nan,
                        }
                    )

            # Quarterly linear regressor blocks.
            for spec in self.quarterly_specs:
                vf = fit.variable_fits[spec.variable]
                z_row = z_rows[spec.variable]
                if np.all(np.isfinite(z_row)):
                    horizon_rows.append(
                        {
                            "horizon": h,
                            "date": forecast_date,
                            "component": spec.variable,
                            "contribution": float(z_row[0] @ vf.weights),
                            "weight": np.nan,
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

            if np.isfinite(sum(row["contribution"] for row in horizon_rows)):
                rows.extend(horizon_rows)

        return pd.DataFrame(
            rows, columns=["horizon", "date", "component", "contribution", "weight"]
        )

    # ---------------------------------------------------------------------- #
    #  summary                                                                 #
    # ---------------------------------------------------------------------- #

    def summary(self, horizon: int = 0) -> str:
        """Print a formatted summary of the fitted model for one horizon and return it.

        Parameters
        ----------
        horizon : int
            Horizon to summarise (default 0).

        Returns
        -------
        str
            The formatted summary text (also printed to stdout).

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        ValueError
            If *horizon* was not fitted.
        """
        if not self.fits_:
            raise RuntimeError("Not fitted. Call fit() first.")
        if horizon not in self.fits_:
            raise ValueError(
                f"horizon={horizon} not fitted. Available: {sorted(self.fits_)}"
            )
        fit = self.fits_[horizon]
        sse = float(fit.residuals @ fit.residuals)
        rmse = float(np.sqrt(sse / fit.nobs))
        est = "OLS" if self._use_ols else "NLS (Levenberg-Marquardt)"

        sep = "=" * 56
        thin = "-" * 56
        lines = [
            sep,
            "       Multi-Regressor MIDAS Results",
            sep,
            f"  Estimator  : {est}",
            f"  Horizon    : {horizon}",
            f"  Obs        : {fit.nobs}",
            thin,
            f"  alpha : {fit.alpha: .6f}",
        ]
        for spec in self.monthly_specs:
            vf = fit.variable_fits[spec.variable]
            lines.append(
                f"\n  [{spec.variable}]  method={spec.method}  n_lags={spec.n_lags}"
            )
            if not self._use_ols:
                lines.append(f"    beta  = {vf.beta: .6f}")
            for i, t in enumerate(vf.theta):
                lines.append(f"    theta[{i}] = {t: .6f}")
            for i, w in enumerate(vf.weights):
                lines.append(f"    w[{i:2d}]    = {w: .6f}")

        for spec in self.quarterly_specs:
            vf = fit.variable_fits[spec.variable]
            lines.append(f"\n  [{spec.variable}]  frequency=QE  n_lags={spec.n_lags}")
            for i, d in enumerate(vf.weights):
                lines.append(f"    delta[{i}] = {d: .6f}")

        for i, g in enumerate(fit.gamma):
            lines.append(f"  gamma[{i}] = {g: .6f}")
        for i, p in enumerate(fit.phi):
            lines.append(f"  phi[{i + 1}]   = {p: .6f}")

        lines.extend([thin, f"  SSE  : {sse: .6f}", f"  RMSE : {rmse: .6f}", sep])
        text = "\n".join(lines)
        print(text)
        return text


def _resolve_specs(
    variables: list[str | VariableSpec],
    method: str,
    n_lags: int,
    n_pars_weights: int,
    estimator: str | None,
    start_lag: int,
) -> list[VariableSpec]:
    """Resolve a mix of strings and VariableSpec objects to a list of VariableSpec."""
    resolved = []
    for v in variables:
        if isinstance(v, str):
            resolved.append(
                VariableSpec(
                    variable=v,
                    method=method,
                    n_lags=n_lags,
                    n_pars_weights=n_pars_weights,
                    estimator=estimator,
                    start_lag=start_lag,
                )
            )
        else:
            resolved.append(v)
    return resolved


@dataclass
class VariableFit:
    """Fitted parameters for one regressor within a `MultiMIDAS` model.

    Attributes
    ----------
    beta : float
        Estimated slope. The model estimates it separately for nonlinear
        methods (``'exp_almon'``, ``'beta'``); it fixes the value at ``1.0`` for the
        linearly-estimated methods (``'almon'``, ``'unrestricted'``) and
        for quarterly regressors.
    theta : np.ndarray
        Weight-shape parameters.
    weights : np.ndarray
        Evaluated lag weights ``w(theta)``.  Normalised for the nonlinear
        methods; unnormalised lag coefficients for ``'almon'`` /
        ``'unrestricted'`` and the linear coefficients for quarterly
        regressors.
    """

    beta: float = 1.0
    theta: np.ndarray = field(default_factory=lambda: np.array([]))
    weights: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class FittedMultiMidas:
    """Fitted result for a single horizon of a `MultiMIDAS` model.

    Attributes
    ----------
    alpha : float
        Estimated intercept.
    variable_fits : dict[str, VariableFit]
        Mapping from variable name to `VariableFit`.
    gamma : np.ndarray
        Dummy coefficients (empty if no dummies).
    phi : np.ndarray
        AR coefficients (empty if ``n_ar_lags == 0``).
    A : np.ndarray | None
        OLS design matrix (``None`` for NLS estimation).
    fitted_values : np.ndarray
        In-sample fitted values.
    residuals : np.ndarray
        In-sample residuals.
    nobs : int
        Number of observations used.
    y : np.ndarray
        Target vector used in estimation.
    dates : np.ndarray | None
        Low-frequency target dates aligned to the fitted sample.
    """

    alpha: float = 0.0
    variable_fits: dict[str, VariableFit] = field(default_factory=dict)
    gamma: np.ndarray = field(default_factory=lambda: np.array([]))
    phi: np.ndarray = field(default_factory=lambda: np.array([]))
    A: np.ndarray | None = None
    fitted_values: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    nobs: int = 0
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    dates: np.ndarray | None = None
