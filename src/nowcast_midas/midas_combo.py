"""
Combination of MIDAS nowcasts.

Pipeline
--------
1.  Accept quarterly target and monthly regressors as DataFrames.
2.  Fit a `MIDAS` model for each
    `MidasSpec`.
3.  Combine fitted values according to a hierarchy of
    `ComboSpec` nodes.

Combination methods
~~~~~~~~~~~~~~~~~~~
- ``'average'``    - simple equal-weight average.
- ``'rmse'``       - inverse-RMSE discounted weighting.
- ``'mse'``        - inverse-MSE discounted weighting.
- ``'mae'``        - inverse-MAE discounted weighting.
- ``'regression'`` - constrained least-squares (weights >= 0, sum to 1).

Direct-forecasting date convention
----------------------------------
Direct forecasting fits ``y[t+h] ~ X[t]`` (one model per horizon ``h``).
When forecasting, the forecast origin is ``T_X``, the latest date with a
finite regressor value. To forecast
``y[T+step]`` where ``T`` is the last observed target, the horizon used is the
gap between the forecast date and the latest regressor: ``h = (T + step) - T_X``.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from ._compat import legacy_forecast_alias
from .combo_weights import (
    _filter_sources,
    fit_average,
    fit_error_based_weights,
    fit_regression_weights,
)
from .midas import MIDAS, FittedMidas
from .multi_midas import FittedMultiMidas, MultiMIDAS
from .ols import OLS, FittedOLS
from .plots.midas_combo import _ComboPlots
from .specs import ComboSpec, MidasSpec, MultiMidasSpec, OLSSpec, VariableSpec
from .utils import _build_lag_matrix

__all__ = ["MidasCombo"]

_VALID_COMBO_METHODS = ("average", "rmse", "mse", "mae", "regression")


class MidasCombo(_ComboPlots):
    """MIDAS forecast combination pipeline.

    Fits individual MIDAS / OLS regressions for the indicators
    referenced by the combination tree and combines them according to
    the supplied hierarchy of `ComboSpec` nodes.

    The pipeline derives MIDAS, OLS, and MultiMIDAS indicator specs from
    ``combo_specs``. It registers each spec object that appears as a source
    and fits the resulting leaves before it fits the combination nodes.

    Parameters
    ----------
    combo_specs : ComboSpec | None
        Root combination node.  ``sources`` may mix variable / combo
        name strings, `MidasSpec` / `OLSSpec` /
        `MultiMidasSpec` instances and nested `ComboSpec`
        instances.  The tree is automatically flattened into dependency
        order.
    horizons : int
        Number of application forecast steps (default 3). Each indicator fits
        the direct horizons required to produce those steps, subject to the
        available target sample.

    Raises
    ------
    TypeError
        If *combo_specs* is not a `ComboSpec`.
    ValueError
        If model names collide or a combination method is invalid.
    """

    def __init__(
        self,
        combo_specs: ComboSpec | None = None,
        horizons: int = 3,
    ) -> None:
        # Flatten the nested ComboSpec tree into dependency order and
        # harvest any embedded indicator specs.
        if combo_specs is not None:
            if not isinstance(combo_specs, ComboSpec):
                raise TypeError(f"Expected ComboSpec, got {type(combo_specs)}")
            self._combo_specs_flat = combo_specs.flatten()
            midas_specs, ols_specs, multi_midas_specs = combo_specs.collect_indicators()
        else:
            self._combo_specs_flat = []
            midas_specs, ols_specs, multi_midas_specs = [], [], []

        for spec in midas_specs:
            if not isinstance(spec, MidasSpec):
                raise TypeError(f"Expected MidasSpec, got {type(spec)}")
        for spec in ols_specs:
            if not isinstance(spec, OLSSpec):
                raise TypeError(f"Expected OLSSpec, got {type(spec)}")
        for spec in multi_midas_specs:
            if not isinstance(spec, MultiMidasSpec):
                raise TypeError(f"Expected MultiMidasSpec, got {type(spec)}")

        # Detect name collisions between MIDAS, OLS and MultiMIDAS.
        midas_names = {s.variable for s in midas_specs}
        ols_names = {s.variable for s in ols_specs}
        multi_names = {s.name for s in multi_midas_specs}
        collisions = (
            (midas_names & ols_names)
            | (midas_names & multi_names)
            | (ols_names & multi_names)
        )
        if collisions:
            raise ValueError(
                f"Variables appear in both midas_specs and ols_specs: {collisions}"
            )

        for spec in self._combo_specs_flat:
            if spec.method not in _VALID_COMBO_METHODS:
                raise ValueError(
                    f"ComboSpec method must be one of {_VALID_COMBO_METHODS}, "
                    f"got '{spec.method}'"
                )

        self.midas_specs = midas_specs
        self.ols_specs = ols_specs
        self.multi_midas_specs = multi_midas_specs
        self.indicator_variables_ = {
            **{spec.variable: [spec.variable] for spec in midas_specs},
            **{spec.variable: [spec.variable] for spec in ols_specs},
            **{
                spec.name: [
                    item.variable if isinstance(item, VariableSpec) else item
                    for item in spec.variables
                ]
                for spec in multi_midas_specs
            },
        }
        self.combo_specs = combo_specs
        self.horizons = horizons

        # Populated by fit() --------------------------------------------------
        self.target_: pd.Series | None = None
        self.midas_models_: dict[str, dict[int, FittedMidas]] | None = {}
        self.ols_models_: dict[str, dict[int, FittedOLS]] | None = {}
        self.multi_midas_models_: dict[str, dict[int, FittedMultiMidas]] | None = {}
        self.fitted_: dict[str, pd.Series] | None = {}
        self.fits_df_: pd.DataFrame = pd.DataFrame()
        self.weights_df_: pd.DataFrame = pd.DataFrame()

        self.combo_weights_: dict[str, dict[str, np.ndarray]] | None = {}

        # Populated by forecast() -----------------------------------------------
        self.forecasts_df_: pd.DataFrame = pd.DataFrame()
        self.fits_and_forecasts_df_: pd.DataFrame = pd.DataFrame()

    def fit(
        self,
        target: pd.DataFrame,
        regressors: pd.DataFrame,
    ) -> MidasCombo:
        """Fit MIDAS models and combinations according to the specifications.

        Parameters
        ----------
        target : pd.DataFrame
            Quarterly target in long format with columns ``date``,
            ``variable``, ``frequency``, ``value``. Must contain exactly one
            ``variable``; every ``frequency`` entry must be ``"QE"``
            (case-insensitive); ``value`` must be NaN-free.
        regressors : pd.DataFrame
            Regressors in long format with the same four columns. MIDAS
            indicators must use ``frequency="ME"`` (monthly); OLS indicators
            must use ``frequency="QE"`` (quarterly); each ``variable`` must
            carry a single frequency. Trailing NaNs (ragged edge) in
            ``value`` are allowed.

        Returns
        -------
        self : MidasCombo

        """
        _validate_pipeline_inputs(
            target,
            regressors,
            self.midas_specs,
            self.ols_specs,
            self._combo_specs_flat,
            self.multi_midas_specs,
        )

        # Store target --------------------------------------------------------
        target_sorted = target.sort_values("date")
        self.regressors_ = regressors.sort_values(["variable", "date"])
        self.target_ = pd.Series(
            target_sorted["value"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(target_sorted["date"])),
        )
        self.indicator_model_horizons_, self.indicator_forecast_horizons_ = (
            self._resolve_indicator_horizons()
        )

        # Reset containers for a fresh fit call
        self.midas_models_ = defaultdict(dict)
        self.ols_models_ = defaultdict(dict)
        self.multi_midas_models_ = defaultdict(dict)
        self.midas_instances_: dict[str, MIDAS] = {}
        self.ols_instances_: dict[str, OLS] = {}
        self.multi_midas_instances_: dict[str, MultiMIDAS] = {}
        self.fitted_ = defaultdict(dict)
        self.combo_weights_ = defaultdict(dict)
        self._combo_fits_list: list[pd.DataFrame] = []
        self.fits_df_: pd.DataFrame | None = None
        self.weights_df_: pd.DataFrame | None = pd.DataFrame(
            columns=["spec", "source", "horizon", "value"]
        )
        self.forecasts_df_ = pd.DataFrame()
        self.fits_and_forecasts_df_ = pd.DataFrame()

        # Stage 1: fit each MIDAS indicator at its required direct horizons
        self._fit_midas_models()

        # Stage 1b: fit each quarterly OLS indicator at its required horizons
        self._fit_ols_models()

        # Stage 1c: fit each MultiMIDAS indicator at its required horizons
        self._fit_multi_midas_models()

        # Stage 2: fit combination hierarchy at every horizon.
        for h in range(self.horizons):
            for spec in self._combo_specs_flat:
                self._fit_combo(spec, horizon=h)

        return self

    @legacy_forecast_alias
    def forecast(self) -> pd.DataFrame:
        """Compute out-of-sample forecasts for 1qa .. horizons.

        Must be called after `fit()`.

        Produces one long-format row per indicator and combination for each
        requested application step. A row is ``NaN`` when its indicator lacks
        the finite data or fitted direct horizon needed for that step.

        Attributes set:
        - ``self.forecasts_df_``: long-format OOS forecasts
          [date, horizon, value, spec]
        - ``self.fits_and_forecasts_df_``: concatenation of
          self.fits_df_ (in-sample) and self.forecasts_df_ (OOS)

        Returns
        -------
        pd.DataFrame
            Long-format OOS forecasts with columns ``date``, ``horizon``,
            ``value``, ``spec``. One row per spec per horizon step.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.target_ is None:
            raise ValueError("SC MIDAS model has not been fitted yet")

        # Trim fitted_ back to in-sample length in case forecast() is re-called.
        last_target = self.target_.index[-1]

        # ------------------------------------------------------------------
        rows: list[dict] = []
        for step in range(self.horizons):
            h = step
            forecast_date = last_target + pd.DateOffset(months=3 * (step + 1))
            oos_vals = self._forecast_models(forecast_date)
            combo_vals = self._forecast_combos(step=step, h=h, oos_vals=oos_vals)
            rows.append(
                {
                    "step": step,
                    **oos_vals,
                    **combo_vals,
                }
            )
        df_long = pd.DataFrame(rows)

        # ------------------------------------------------------------------
        # Step 3: produce long-format forecasts only (no wide pivot).
        # ------------------------------------------------------------------
        value_cols = [c for c in df_long.columns if c not in ("step", "frequency")]

        # Melt to long format
        forecasts_df_ = df_long.melt(
            id_vars=["step"],
            value_vars=value_cols,
            var_name="spec",
            value_name="value",
        ).sort_values(["step", "spec"])

        # Add date (step=0 → y[T+1], step=1 → y[T+2], etc.)
        forecasts_df_["date"] = last_target + forecasts_df_["step"].apply(
            lambda r: pd.DateOffset(months=3 * (r + 1))
        )

        # Rename step to horizon and use the canonical column order.
        self.forecasts_df_ = forecasts_df_.rename(columns={"step": "horizon"})[
            ["date", "horizon", "spec", "value"]
        ]

        # ------------------------------------------------------------------
        # Step 4: combine fits_df_ and forecasts_df_ into a single long df.
        # ------------------------------------------------------------------
        if self.fits_df_ is not None and len(self.fits_df_) > 0:
            # Both have [date, horizon, value, spec] columns
            self.fits_and_forecasts_df_ = pd.concat(
                [self.fits_df_, self.forecasts_df_], ignore_index=True
            )
        else:
            self.fits_and_forecasts_df_ = self.forecasts_df_.copy()

        return self.forecasts_df_

    # ---------------------------------------------------------------------- #
    #  Forecast helpers                                                      #
    # ---------------------------------------------------------------------- #

    def _resolve_indicator_horizons(
        self,
    ) -> tuple[dict[str, list[int]], dict[str, list[int | None]]]:
        """Resolve ragged-edge horizons for each indicator model.

        For each indicator, determine the direct model horizons needed for
        the requested forecast steps, based on the latest finite regressor
        release shared by that indicator's variables.
        """
        last_target_period = self.target_.index[-1].to_period("Q")
        forecast_horizons: dict[str, list[int | None]] = {}
        model_horizons: dict[str, list[int]] = {}
        for name, variables in self.indicator_variables_.items():
            latest_period = self._latest_period(variables)
            required: list[int | None] = []
            for step in range(self.horizons):
                if latest_period is None:
                    required.append(None)
                    continue
                target_period = last_target_period + step + 1
                horizon = target_period.ordinal - latest_period.ordinal
                required.append(horizon if horizon >= 0 else None)
            forecast_horizons[name] = required
            finite = [horizon for horizon in required if horizon is not None]
            max_horizon = min(
                max(finite, default=0),
                max(len(self.target_) - 1, 0),
            )
            model_horizons[name] = list(range(max_horizon + 1))

        return model_horizons, forecast_horizons

    def _latest_period(
        self,
        variables: list[str],
        regressors: pd.DataFrame | None = None,
    ) -> pd.Period | None:
        """Return the latest finite quarter shared by the given variables."""
        source = regressors if regressors is not None else self.regressors_
        latest_dates = []
        for variable in variables:
            data = source.loc[source["variable"] == variable]
            finite_dates = data.loc[np.isfinite(data["value"]), "date"]
            if len(finite_dates):
                latest_dates.append(pd.Timestamp(finite_dates.max()))
        if not latest_dates:
            return None
        return max(latest_dates).to_period("Q")

    def _indicator_horizon(
        self,
        name: str,
        forecast_date: pd.Timestamp,
        regressors: pd.DataFrame | None = None,
    ) -> int | None:
        """Return an indicator's direct horizon for a common target date."""
        latest_period = self._latest_period(self.indicator_variables_[name], regressors)
        if latest_period is None:
            return None
        horizon = (
            pd.Timestamp(forecast_date).to_period("Q").ordinal - latest_period.ordinal
        )
        return horizon if horizon >= 0 else None

    def _source_horizon(self, source: str, step: int) -> int | None:
        """Return the direct horizon for a source at an application step."""
        combo_names = {spec.name for spec in self._combo_specs_flat}
        if source in combo_names:
            return step
        return self.indicator_forecast_horizons_.get(source, [None] * self.horizons)[
            step
        ]

    def _source_fitted(self, source: str, step: int) -> pd.Series:
        """Return a source's fitted values for one combination step."""
        horizon = self._source_horizon(source, step)
        if horizon is None:
            return pd.Series(np.nan, index=self.target_.index)
        fitted = self.fitted_.get(source, {}).get(horizon)
        if fitted is None:
            return pd.Series(np.nan, index=self.target_.index)
        return pd.Series(fitted, index=self.target_.index)

    def _forecast_models(
        self,
        forecast_date: pd.Timestamp,
        regressors: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Forecast every indicator at one common target date.

        Each indicator selects the direct forecast horizon implied by its own
        latest finite regressor release.  This allows leaves with different
        publication lags to forecast the same target quarter.
        """
        out: dict[str, float] = {name: np.nan for name in self.indicator_variables_}
        source = regressors if regressors is not None else self.regressors_

        # Single-variable models (MIDAS / OLS)
        single_instances: dict[str, MIDAS | OLS] = {
            **getattr(self, "midas_instances_", {}),
            **getattr(self, "ols_instances_", {}),
        }
        for variable, mdl in single_instances.items():
            reg = source.loc[
                source["variable"] == variable,
                ["date", "value"],
            ].sort_values("date")
            horizon = self._indicator_horizon(variable, forecast_date, source)
            if horizon is None or horizon not in mdl.fits_:
                out[variable] = np.nan
                continue
            if isinstance(mdl, MIDAS):
                finite_dates = reg.loc[np.isfinite(reg["value"]), "date"]
                if finite_dates.empty:
                    out[variable] = np.nan
                    continue
                regressor_date = pd.Timestamp(finite_dates.max())
                lag_row = _build_lag_matrix(
                    regressor_date, reg, mdl.n_lags, mdl.start_lag
                )
                if not np.all(np.isfinite(lag_row)):
                    out[variable] = np.nan
                    continue
            fc = mdl.forecast(reg)
            matching = fc.loc[
                fc["horizon"] == horizon,
                "value",
            ]
            out[variable] = float(matching.iloc[0]) if len(matching) else np.nan

        # Multi-variable models (MultiMIDAS)
        for name, mdl in getattr(self, "multi_midas_instances_", {}).items():
            # MultiMIDAS.forecast() expects all variables in one DF
            var_names = [s.variable for s in mdl.specs]
            reg = source.loc[
                source["variable"].isin(var_names),
                ["date", "variable", "value"],
            ].sort_values(["variable", "date"])
            horizon = self._indicator_horizon(name, forecast_date, source)
            if horizon is None or horizon not in mdl.fits_:
                out[name] = np.nan
                continue
            fc = mdl.forecast(reg)
            matching = fc.loc[
                fc["horizon"] == horizon,
                "value",
            ]
            out[name] = float(matching.iloc[0]) if len(matching) else np.nan

        return out

    def _forecast_combos(
        self,
        step: int,
        h: int,
        oos_vals: dict[str, float],
    ) -> dict[str, float]:
        """Combine indicator forecasts using the OOS weight row.

        The weight-fitting functions append one extra weight row (index ``T``) per
        horizon, estimated from the full in-sample history, for
        out-of-sample use. Every application step/horizon reads that same
        row; the row already varies by horizon because each horizon has
        its own ``combo_weights_[name][h]`` array. Falls back to the last
        finite in-sample weight if that slot is NaN (e.g. ``method='average'``,
        whose weight arrays have no such extra row).
        """
        T = len(self.target_)
        w_idx = T
        out: dict[str, float] = {}
        # ``_combo_specs_flat`` is in dependency order (leaves first),
        # so a combo whose source is another combo can read that inner
        # combo's already-computed OOS value from ``available`` below.
        available: dict[str, float] = dict(oos_vals)
        for cspec in self._combo_specs_flat:
            combo_w = self.combo_weights_.get(cspec.name, {}).get(h)
            if combo_w is None:
                out[cspec.name] = np.nan
                available[cspec.name] = np.nan
                continue
            source_values = {
                src: available.get(src, np.nan) for src in cspec.source_names
            }
            weights = self._available_source_weights(
                cspec.source_names, combo_w, w_idx, source_values
            )
            if not weights:
                out[cspec.name] = np.nan
                available[cspec.name] = np.nan
                continue
            combo_val = 0.0
            for src, weight in weights.items():
                combo_val += weight * source_values[src]
            out[cspec.name] = combo_val
            available[cspec.name] = combo_val
        return out

    # ---------------------------------------------------------------------- #
    #  Forecast decomposition                                                #
    # ---------------------------------------------------------------------- #

    def forecast_decomp(
        self,
        spec_name: str | None = None,
        regressors: pd.DataFrame | None = None,
        aggregate: bool = False,
    ) -> pd.DataFrame:
        """Additive component decomposition of a combo's OOS forecast.

        The combined forecast of a `ComboSpec` is a weighted sum of
        its source forecasts, and each source is itself a (possibly nested)
        combination of individual MIDAS / OLS / MultiMIDAS indicator models.
        This routine flattens that hierarchy down to the indicator models,
        computes each indicator's *effective* combination weight in the
        target combo, and pushes that weight through the indicator's own
        `forecast_decomp()` so every component sums back to the combined
        forecast produced by `forecast()`:

            forecast[h] = sum_models  w_eff_model
                          * sum_components  contribution_{model, component}

        Components are labelled ``"{model}::{component}"``.  An indicator's
        block whose ``weight`` is ``NaN`` (e.g. a MIDAS weighted-lag block)
        keeps ``NaN``; otherwise the reported ``weight`` is the effective
        combination weight times the indicator's scalar coefficient.

        Parameters
        ----------
        spec_name : str | None
            Name of the combo (or indicator model) to decompose.
            Defaults to the root `ComboSpec` passed at construction.
        regressors : pd.DataFrame | None
            Long-format regressors to use for the decomposition. If None,
            the regressors stored at fit time (``self.regressors_``) are used.
            Pass a different DataFrame to compute counterfactual decompositions
            (e.g. old model evaluated on new data for revision attribution).
        aggregate : bool
            If ``False`` (default), emit one row per indicator sub-component,
            labelled ``"{model}::{component}"`` (the disaggregate view). If
            ``True``, collapse each indicator down to a single row per model
            named ``"{model}"``, with ``contribution = w_eff * model_forecast``
            and ``weight = w_eff``. Both views individually sum to the same
            combined forecast at each horizon; they are not meant to be
            concatenated together.

        Returns
        -------
        pd.DataFrame
            Long-format decomposition with columns ``horizon``, ``date``,
            ``component``, ``contribution`` and ``weight``.  Contributions
            sum to the combined forecast value at every horizon.

        Raises
        ------
        ValueError
            If the model has not been fitted or the requested specification
            cannot be resolved.
        """
        if self.target_ is None:
            raise ValueError("SC MIDAS model has not been fitted yet")

        if spec_name is None:
            if self.combo_specs is None:
                raise ValueError(
                    "spec_name is required when no ComboSpec was provided."
                )
            spec_name = self.combo_specs.name

        combo_names = {s.name for s in self._combo_specs_flat}
        T = len(self.target_)
        last_target = self.target_.index[-1]

        rows: list[dict] = []
        for step in range(self.horizons):
            h = step
            forecast_date = last_target + pd.DateOffset(months=3 * (h + 1))
            oos_vals = self._forecast_models(forecast_date, regressors=regressors)
            combo_vals = self._forecast_combos(step=step, h=h, oos_vals=oos_vals)
            available = {**oos_vals, **combo_vals}

            if spec_name in combo_names:
                eff_weights = self._effective_indicator_weights(
                    spec_name, h, T, available
                )
            else:
                # An indicator model decomposes with effective weight 1.
                eff_weights = {spec_name: 1.0}

            for model_name, w_eff in eff_weights.items():
                fval = oos_vals.get(model_name, np.nan)
                if w_eff == 0.0 or not np.isfinite(fval):
                    continue
                comp_rows = self._model_decomp_rows(
                    model_name, forecast_date, regressors=regressors
                )
                if aggregate:
                    # Sum sub-component contributions computed from the supplied data
                    # rather than reusing the stale fit-time `fval`, so
                    # counterfactual `regressors` passed by the caller (e.g.
                    # for revision attribution) are respected here too.
                    total_contrib = sum(contrib for _, contrib, _ in comp_rows)
                    rows.append(
                        {
                            "horizon": h,
                            "date": forecast_date,
                            "component": model_name,
                            "contribution": w_eff * total_contrib,
                            "weight": w_eff,
                        }
                    )
                    continue
                for comp, contrib, cw in comp_rows:
                    rows.append(
                        {
                            "horizon": h,
                            "date": forecast_date,
                            "component": f"{model_name}::{comp}",
                            "contribution": w_eff * contrib,
                            "weight": w_eff * cw if np.isfinite(cw) else np.nan,
                        }
                    )

        return pd.DataFrame(
            rows, columns=["horizon", "date", "component", "contribution", "weight"]
        )

    @staticmethod
    def _resolve_oos_weight(w_arr: np.ndarray, w_idx: int) -> float:
        """Resolve the applicable OOS weight (mirrors `_forecast_combos()`)."""
        if w_idx < len(w_arr) and np.isfinite(w_arr[w_idx]):
            return float(w_arr[w_idx])
        finite = w_arr[np.isfinite(w_arr)]
        return float(finite[-1]) if len(finite) else 0.0

    def _available_source_weights(
        self,
        source_names: list[str],
        combo_weights: dict[str, np.ndarray],
        w_idx: int,
        available: dict[str, float],
    ) -> dict[str, float]:
        """Return normalised weights for sources with finite forecasts."""
        values = np.asarray(
            [available.get(source, np.nan) for source in source_names], dtype=float
        )
        available_mask = np.isfinite(values)
        if not available_mask.any():
            return {}

        raw_weights = np.asarray(
            [
                self._resolve_oos_weight(combo_weights.get(source, np.array([])), w_idx)
                for source in source_names
            ],
            dtype=float,
        )
        raw_weights = np.where(available_mask, raw_weights, 0.0)
        if raw_weights.sum() <= 0:
            return {}
        weights = raw_weights / raw_weights.sum()
        return {
            source: float(weight)
            for source, weight, is_available in zip(
                source_names, weights, available_mask
            )
            if is_available
        }

    def _effective_indicator_weights(
        self,
        combo_name: str,
        h: int,
        w_idx: int,
        available: dict[str, float],
    ) -> dict[str, float]:
        """Reduce a (possibly nested) combo to effective indicator weights.

        Returns a mapping ``{indicator_model_name: effective_weight}`` such
        that the combo forecast equals ``sum_model w_eff * model_forecast``.
        """
        spec = next(s for s in self._combo_specs_flat if s.name == combo_name)
        combo_w = self.combo_weights_.get(combo_name, {}).get(h)
        out: dict[str, float] = defaultdict(float)
        if combo_w is None:
            return out
        combo_names = {s.name for s in self._combo_specs_flat}
        weights = self._available_source_weights(
            spec.source_names, combo_w, w_idx, available
        )
        for src, w in weights.items():
            if src in combo_names:
                for indicator, indicator_weight in self._effective_indicator_weights(
                    src, h, w_idx, available
                ).items():
                    out[indicator] += w * indicator_weight
            else:
                out[src] += w
        return dict(out)

    def _model_decomp_rows(
        self,
        name: str,
        forecast_date: pd.Timestamp,
        regressors: pd.DataFrame | None = None,
    ) -> list[tuple[str, float, float]]:
        """Return ``(component, contribution, weight)`` rows of an indicator model.

        Parameters
        ----------
        name : str
            Name of the indicator model (MIDAS, OLS, or MultiMIDAS) to decompose.
        forecast_date : pd.Timestamp
            Common target date to extract.
        regressors : pd.DataFrame | None
            Long-format regressors to use. Falls back to ``self.regressors_``
            when None.

        Returns
        -------
        list[tuple[str, float, float]]
            Component, contribution, and scalar weight rows.
        """
        src = regressors if regressors is not None else self.regressors_
        if name in getattr(self, "midas_instances_", {}):
            mdl = self.midas_instances_[name]
            reg = src.loc[src["variable"] == name, ["date", "value"]].sort_values(
                "date"
            )
            decomp = mdl.forecast_decomp(reg, regressor_name=name)
        elif name in getattr(self, "ols_instances_", {}):
            mdl = self.ols_instances_[name]
            reg = src.loc[src["variable"] == name, ["date", "value"]].sort_values(
                "date"
            )
            decomp = mdl.forecast_decomp(reg, regressor_name=name)
        elif name in getattr(self, "multi_midas_instances_", {}):
            mdl = self.multi_midas_instances_[name]
            var_names = [s.variable for s in mdl.specs]
            reg = src.loc[
                src["variable"].isin(var_names),
                ["date", "variable", "value"],
            ].sort_values(["variable", "date"])
            decomp = mdl.forecast_decomp(reg)
        else:
            return []

        forecast_period = pd.Timestamp(forecast_date).to_period("Q")
        decomp = decomp.loc[
            pd.to_datetime(decomp["date"]).dt.to_period("Q") == forecast_period
        ]
        return list(
            zip(
                decomp["component"],
                decomp["contribution"],
                decomp["weight"],
            )
        )

    def summary(self, horizon: int = 0) -> str:
        """Print a formatted summary of all models and combinations and return it.

        Parameters
        ----------
        horizon : int
            The forecast horizon to summarise.

        Returns
        -------
        str
            The formatted summary text (also printed to stdout).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """

        # check that the model has been fitted first
        if self.target_ is None or self.fitted_ is None:
            raise ValueError("SC MIDAS model has not been fitted yet")

        sep = "=" * 60
        thin = "-" * 60
        lines = [
            sep,
            f"       MIDAS Combination Pipeline Results - Horizon {horizon}",
            sep,
        ]

        # ---- Individual MIDAS models ----
        lines.append("\n  Individual MIDAS Models:")
        lines.append(thin)
        for spec in self.midas_specs:
            model = self.midas_models_.get(spec.variable, {}).get(horizon)
            if model is not None:
                rmse = float(np.sqrt(model.residuals @ model.residuals / model.nobs))
                lines.append(
                    f"    {spec.variable:20s}  method={spec.method:12s}  "
                    f"n_lags={spec.n_lags}  RMSE={rmse:.4f}"
                )
            else:
                lines.append(f"    {spec.variable:20s}  (not fitted)")

        # ---- Combinations ----
        lines.append("\n  Combinations:")
        lines.append(thin)
        for spec in self._combo_specs_flat:
            combo_vals = self.fitted_.get(spec.name, {}).get(horizon)
            if combo_vals is None:
                lines.append(f"    {spec.name:20s}  (not fitted for horizon={horizon})")
                continue
            target = self.target_
            # combo_vals may be longer than target if OOS rows were appended;
            # align by index so only in-sample dates are compared.
            combo_vals = combo_vals.reindex(target.index)
            residuals = target - combo_vals
            valid = residuals.notna()
            rmse = (
                float(np.sqrt((residuals[valid] ** 2).mean()))
                if valid.any()
                else np.nan
            )

            sources_str = ", ".join(spec.source_names)
            lines.append(
                f"    {spec.name:20s}  method={spec.method:12s}  "
                f"sources=[{sources_str}]  RMSE={rmse:.4f}"
            )

            if (
                spec.name in self.combo_weights_
                and horizon in self.combo_weights_[spec.name]
            ):
                lines.append("      Latest weights:")
                for src, w in self.combo_weights_[spec.name][horizon].items():
                    finite = w[np.isfinite(w)]
                    latest = float(finite[-1]) if len(finite) else np.nan
                    lines.append(f"        {src}: {latest:.4f}")

        lines.append(sep)
        text = "\n".join(lines)
        print(text)
        return text

    # ---------------------------------------------------------------------- #
    #  Stage 1: MIDAS models                                                 #
    # ---------------------------------------------------------------------- #

    def _fit_midas_models(self) -> None:
        """Fit individual MIDAS models for each spec across all horizons."""
        T = len(self.target_)
        if not hasattr(self, "midas_instances_"):
            self.midas_instances_ = {}

        midas_dfs_list = []
        for spec in self.midas_specs:
            horizons = self.indicator_model_horizons_[spec.variable]
            var_data = self.regressors_.loc[
                self.regressors_["variable"] == spec.variable
            ].sort_values("date")

            # Prepare 2-column DataFrames expected by MIDAS.fit()
            target_df = pd.DataFrame(
                {"date": self.target_.index, "value": self.target_.to_numpy()}
            )
            regressor_df = var_data[["date", "value"]].copy()

            model = MIDAS(
                method=spec.method,
                n_lags=spec.n_lags,
                n_pars_weights=spec.n_pars_weights,
                estimator=spec.estimator,
                horizons=horizons,
                start_lag=spec.start_lag,
                n_ar_lags=spec.n_ar_lags,
                dummy_periods=spec.dummy_periods,
            )

            try:
                model.fit(target_df, regressor_df)

                fits_h0 = model.fits_df_.loc[model.fits_df_["horizon"] == horizons[0]]

                if (
                    spec.minimum_sample_size is not None
                    and len(fits_h0["value"]) < spec.minimum_sample_size
                ):
                    raise ValueError()

                # Collect model.fits_df_ with spec column for accumulation
                midas_df = model.fits_df_.copy()
                midas_df["spec"] = spec.variable
                midas_dfs_list.append(midas_df)
            except ValueError:
                warnings.warn(
                    f"Insufficient data for '{spec.variable}'",
                    RuntimeWarning,
                    stacklevel=2,
                )
                for h in horizons:
                    self.fitted_[spec.variable][h] = pd.Series(
                        np.full(T, np.nan), index=self.target_.index
                    )
                continue

            # Map fitted values back into the full T-length target array
            # (NaN where lags were missing). fit.fitted_values is a Series
            # with the exact target dates it covers, so .loc alignment is direct.
            self.midas_instances_[spec.variable] = model

            for h, fit in model.fits_.items():
                self.midas_models_[spec.variable][h] = fit

                s = pd.Series(np.full(T, np.nan), index=self.target_.index)
                s.loc[fit.fitted_values.index] = fit.fitted_values
                self.fitted_[spec.variable][h] = s

        # Concat all MIDAS fits DataFrames once at end
        if midas_dfs_list:
            self.fits_df_ = pd.concat(midas_dfs_list, ignore_index=True)

    def _fit_ols_models(self) -> None:
        """Fit a quarterly `OLS` model per `OLSSpec`.

        Fitted values are mapped back into the full T-length target
        array (NaN where lags / dummies were missing) and stored under
        ``self.fitted_[spec.variable][h]`` so that `ComboSpec`
        nodes can consume them transparently.
        """
        T = len(self.target_)
        target_df = pd.DataFrame(
            {"date": self.target_.index, "value": self.target_.to_numpy()}
        )
        if not hasattr(self, "ols_instances_"):
            self.ols_instances_ = {}

        ols_dfs_list = []
        for spec in self.ols_specs:
            horizons = self.indicator_model_horizons_[spec.variable]
            regressor_df = self.regressors_.loc[
                self.regressors_["variable"] == spec.variable,
                ["date", "value"],
            ].sort_values("date")

            model = OLS(
                n_lags=spec.n_lags,
                start_lag=spec.start_lag,
                horizons=horizons,
                n_ar_lags=spec.n_ar_lags,
                dummy_periods=spec.dummy_periods,
            )

            try:
                model.fit(target_df, regressor_df)

                fits_h0 = model.fits_df_.loc[model.fits_df_["horizon"] == horizons[0]]

                if (
                    spec.minimum_sample_size is not None
                    and len(fits_h0["value"]) < spec.minimum_sample_size
                ):
                    raise ValueError()
                    # This follows the EViews implementation, but the check must
                    # also cover other horizons.

                # Collect model.fits_df_ with spec column for accumulation
                ols_df = model.fits_df_.copy()
                ols_df["spec"] = spec.variable
                ols_dfs_list.append(ols_df)
            except ValueError:
                warnings.warn(
                    f"Insufficient data for OLSSpec '{spec.variable}'",
                    RuntimeWarning,
                    stacklevel=2,
                )
                for h in horizons:
                    self.fitted_[spec.variable][h] = pd.Series(
                        np.full(T, np.nan), index=self.target_.index
                    )
                continue

            # Map fitted values back into the full T-length target array.
            self.ols_instances_[spec.variable] = model

            for h, fit in model.fits_.items():
                self.ols_models_[spec.variable][h] = fit

                s = pd.Series(np.full(T, np.nan), index=self.target_.index)
                s.loc[fit.fitted_values.index] = fit.fitted_values
                self.fitted_[spec.variable][h] = s

        # Concat all OLS fits DataFrames and append to existing self.fits_df_
        if ols_dfs_list:
            ols_fits_df = pd.concat(ols_dfs_list, ignore_index=True)
            if self.fits_df_ is not None:
                self.fits_df_ = pd.concat(
                    [self.fits_df_, ols_fits_df], ignore_index=True
                )
            else:
                self.fits_df_ = ols_fits_df

    def _fit_multi_midas_models(self) -> None:
        """Fit `MultiMIDAS` models per
        `MultiMidasSpec`.

        Fitted values are mapped back into the full T-length target
        array (NaN where lags were missing) and stored under
        ``self.fitted_[spec.name][h]``.
        """
        T = len(self.target_)
        target_df = pd.DataFrame(
            {"date": self.target_.index, "value": self.target_.to_numpy()}
        )

        for spec in self.multi_midas_specs:
            horizons = self.indicator_model_horizons_[spec.name]
            # Collect all variable names referenced by this spec
            var_names = [
                v.variable if isinstance(v, VariableSpec) else v for v in spec.variables
            ]
            regressor_df = self.regressors_.loc[
                self.regressors_["variable"].isin(var_names),
                ["date", "variable", "value"],
            ].sort_values(["variable", "date"])

            model = MultiMIDAS(
                variables=spec.variables,
                method=spec.method,
                n_lags=spec.n_lags,
                n_pars_weights=spec.n_pars_weights,
                estimator=spec.estimator,
                horizons=horizons,
                start_lag=spec.start_lag,
                n_ar_lags=spec.n_ar_lags,
                dummy_periods=spec.dummy_periods,
            )

            try:
                model.fit(target_df, regressor_df)

                fits_h0 = model.fits_df_.loc[model.fits_df_["horizon"] == horizons[0]]

                if (
                    spec.minimum_sample_size is not None
                    and len(fits_h0["value"]) < spec.minimum_sample_size
                ):
                    raise ValueError()
                    # This follows the EViews implementation, but the check must
                    # also cover other horizons.

            except ValueError:
                warnings.warn(
                    f"Insufficient data for MultiMidasSpec '{spec.name}'",
                    RuntimeWarning,
                    stacklevel=2,
                )
                for h in horizons:
                    self.fitted_[spec.name][h] = np.full(T, np.nan)
                continue

            valid_indices = np.where(model.valid_mask_)[0]
            self.multi_midas_instances_[spec.name] = model

            for h, fit in model.fits_.items():
                self.multi_midas_models_[spec.name][h] = fit

                fitted = np.full(T, np.nan)
                n_fit = len(fit.fitted_values)
                target_indices = valid_indices[:n_fit] + h
                fitted[target_indices] = fit.fitted_values
                self.fitted_[spec.name][h] = fitted

    def _fit_combo(self, spec: ComboSpec, horizon: int) -> None:
        source_names = list(spec.source_names)
        source_df = pd.DataFrame(
            {src: self._source_fitted(src, horizon) for src in source_names},
            index=self.target_.index,
        )
        source_df = _filter_sources(source_df, spec.minimum_sample_size)

        if spec.method == "average":
            combo, weights = fit_average(source_df)
        elif spec.method == "regression":
            combo, weights = fit_regression_weights(
                self.target_,
                source_df,
                method=spec.estimator,
                window=spec.window,
                dummy_periods=spec.dummy_periods,
                minimum_sample_size=spec.minimum_sample_size,
            )
        else:
            combo, weights = fit_error_based_weights(
                self.target_,
                source_df,
                method=spec.method,
                window=spec.window,
                discount_rate=spec.discount_rate,
                dummy_periods=spec.dummy_periods,
            )

        combo_series = pd.Series(combo, index=source_df.index)
        self.fitted_[spec.name][horizon] = combo_series.reindex(self.target_.index)
        complete_weights = {
            source: np.zeros(len(self.target_), dtype=float) for source in source_names
        }
        complete_weights.update(weights)
        self.combo_weights_[spec.name][horizon] = complete_weights

        # Accumulate combo fits for batch concat at end of fit()
        combo_fits_df = pd.DataFrame(
            {
                "spec": spec.name,
                "date": combo_series.index,
                "horizon": horizon,
                "value": combo_series.values,
            }
        )
        self.fits_df_ = pd.concat([self.fits_df_, combo_fits_df], ignore_index=True)

        # Save weights for each source in long format
        weights_dfs = []
        for source, weight_arr in weights.items():
            weights_df = pd.DataFrame(
                {
                    "spec": spec.name,
                    "source": source,
                    "horizon": horizon,
                    "value": weight_arr,
                }
            )
            weights_dfs.append(weights_df)
            # Add dates after the combination fit loop provides the final index.

        if weights_dfs:
            combined_weights_df = pd.concat(weights_dfs, ignore_index=True)
            self.weights_df_ = pd.concat(
                [self.weights_df_, combined_weights_df], ignore_index=True
            )


# ======================================================================== #
#  Module-level helpers                                                     #
# ======================================================================== #


def _validate_pipeline_inputs(
    target: pd.DataFrame,
    regressors: pd.DataFrame,
    midas_specs: list[MidasSpec],
    ols_specs: list[OLSSpec],
    combo_specs: list[ComboSpec],
    multi_midas_specs: list[MultiMidasSpec] | None = None,
) -> None:
    """Raise informative errors if the pipeline inputs are malformed."""
    for name, df in [("target", target), ("regressors", regressors)]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"{name} must be a pandas DataFrame, got {type(df).__name__}."
            )

    required = {"date", "variable", "frequency", "value"}
    for name, df in [("target", target), ("regressors", regressors)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} DataFrame missing columns: {missing}")

    if not all(f.upper() == "QE" for f in target["frequency"].unique()):
        raise ValueError("target frequency must be 'QE'")

    if len(target["variable"].unique()) != 1:
        raise ValueError("target must contain exactly one variable")

    # target can't have NaNs
    if target["value"].isna().any():
        raise ValueError("target 'value' column contains NaNs")

    # Per-variable frequency check: MIDAS variables must be ME, OLS QE.
    var_freq = (
        regressors.groupby("variable")["frequency"]
        .agg(lambda s: s.str.upper().unique())
        .to_dict()
    )

    available_vars = set(regressors["variable"].unique())
    for spec in midas_specs:
        if spec.variable not in available_vars:
            raise ValueError(
                f"MidasSpec variable '{spec.variable}' not found in "
                f"regressors. Available: {available_vars}"
            )
        freqs = var_freq[spec.variable]
        if not (len(freqs) == 1 and freqs[0] == "ME"):
            raise ValueError(
                f"MidasSpec variable '{spec.variable}' must have "
                f"frequency 'ME', got {list(freqs)}"
            )
    for spec in ols_specs:
        if spec.variable not in available_vars:
            raise ValueError(
                f"OLSSpec variable '{spec.variable}' not found in "
                f"regressors. Available: {available_vars}"
            )
        freqs = var_freq[spec.variable]
        if not (len(freqs) == 1 and freqs[0] == "QE"):
            raise ValueError(
                f"OLSSpec variable '{spec.variable}' must have "
                f"frequency 'QE', got {list(freqs)}"
            )

    if multi_midas_specs is None:
        multi_midas_specs = []

    for spec in multi_midas_specs:
        for v in spec.variables:
            var_name = v.variable if isinstance(v, VariableSpec) else v
            if var_name not in available_vars:
                raise ValueError(
                    f"MultiMidasSpec '{spec.name}' variable '{var_name}' "
                    f"not found in regressors. Available: {available_vars}"
                )
            # Check frequency matches what the VariableSpec declares
            expected_freq = v.frequency.upper() if isinstance(v, VariableSpec) else "ME"
            actual_freqs = var_freq[var_name]
            if not (len(actual_freqs) == 1 and actual_freqs[0] == expected_freq):
                raise ValueError(
                    f"MultiMidasSpec '{spec.name}' variable '{var_name}' "
                    f"expected frequency '{expected_freq}', "
                    f"got {list(actual_freqs)} in regressors."
                )

    indicator_names = (
        {s.variable for s in midas_specs}
        | {s.variable for s in ols_specs}
        | {s.name for s in multi_midas_specs}
    )
    combo_names = {s.name for s in combo_specs}
    overlap = indicator_names & combo_names
    if overlap:
        raise ValueError(
            f"Name collision between indicator specs and ComboSpec: {overlap}"
        )

    all_names = indicator_names | combo_names
    for spec in combo_specs:
        for src in spec.source_names:
            if src not in all_names:
                raise ValueError(
                    f"ComboSpec '{spec.name}' references unknown source "
                    f"'{src}'. Known names: {all_names}"
                )
