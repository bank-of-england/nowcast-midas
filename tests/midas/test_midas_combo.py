"""Tests for :class:`~nowcast_midas.midas_combo.MidasCombo`.

Coverage:

- ``test_midascombo_exact_recovery``: generate data from the true DGP
  of the indicator model (matching method, single regressor, near-zero
  noise) and check that ``MidasCombo`` recovers fitted values and the
  OOS forecast essentially to machine precision.
- ``test_midascombo_ragged_edge``: ragged-edge data availability.
- ``test_ols_spec_recovers_coefficients``: OLS-path coefficient recovery.
- ``test_ols_spec_in_combo_and_forecast``: OLS source inside ComboSpec.
- ``test_ols_spec_wrong_frequency_raises``: input-validation.
- independent vintage tests: leaf horizons, parameter recovery, forecasts,
  nested arithmetic, and forecast decomposition.
"""

import numpy as np
import pandas as pd
import pytest

from nowcast_midas.midas import MIDAS
from nowcast_midas.midas_combo import MidasCombo
from nowcast_midas.multi_midas import MultiMIDAS
from nowcast_midas.ols import OLS
from nowcast_midas.specs import ComboSpec, MidasSpec, MultiMidasSpec, OLSSpec
from nowcast_midas.utils import sample_combo_data
from tests.midas.sample_midas import sample_midas
from tests.midas.sample_midas_combo import sample_midas_combo, sample_mixed_model_combo
from tests.midas.sample_multi_midas import sample_multi_midas

# ============================================================================
# Exact-recovery: with the true DGP of the model and tiny noise, fitted
# values and OOS forecasts should match the truth to machine precision.
# ============================================================================


@pytest.mark.parametrize(
    ("kind", "h"),
    [("midas", 0), ("ols", 0), ("ols", 1), ("ols", 2)],
)
def test_midascombo_exact_recovery(kind, h):
    """If we generate data with the indicator model's true DGP and
    near-zero noise, ``MidasCombo`` must recover both the in-sample
    fitted values and the OOS forecast essentially exactly.

    A single indicator is used per case so the DGP equals the model's
    own functional form (no omitted-variable bias).  Method and
    estimator are matched to the DGP so the linear / non-linear LS
    problem has a unique zero-residual solution.
    """
    if kind == "midas":
        sample = sample_midas_combo()
        midas_specs = [
            MidasSpec(
                variable="fast",
                method="unrestricted",
                n_lags=3,
                estimator="ols",
            )
        ]
        ols_specs = []
        var = "fast"
        target_fit = sample.target_train
        regressors = sample.regressors.loc[sample.regressors["variable"] == var].copy()
        y_truth = sample.leaves[var].forecast
        horizons = 1
    else:
        monthly, quarterly = [], ["x_q"]
        method = "almon"
        midas_specs = []
        ols_specs = [OLSSpec(variable="x_q", n_lags=1)]
        var = "x_q"
        target, regressors, _ = sample_combo_data(
            n_quarters=80 + h + 1,
            n_lags=6,
            noise=1e-10,
            seed=11,
            horizon=h,
            method=method,
            monthly_vars=monthly,
            quarterly_vars=quarterly,
            outlier_date=None,
        )
        n_fit = len(target) - (h + 1)
        target_fit = target.iloc[:n_fit].copy()
        y_truth = float(target["value"].iloc[n_fit + h])
        last_x_date = target_fit["date"].iloc[-1] + pd.offsets.QuarterEnd(1)
        regressors = regressors[regressors["date"] <= last_x_date].copy()
        horizons = h + 1

    # Combine all specs into ComboSpec sources
    combo = ComboSpec(name="combo", sources=midas_specs + ols_specs, method="average")
    mc = MidasCombo(combo_specs=combo, horizons=horizons)
    mc.fit(target_fit, regressors)
    mc.forecast()

    # ---- In-sample fit: residuals should be O(noise) ----------------
    fit = mc.midas_models_[var][h] if kind == "midas" else mc.ols_models_[var][h]
    max_resid = float(np.max(np.abs(fit.residuals)))
    assert max_resid < 1e-6, (
        f"[{kind}, h={h}] In-sample residuals not near-zero: max={max_resid:.3e}"
    )

    # ---- Verify fits_df_ contains fitted values for this spec ----
    assert mc.fits_df_ is not None
    fits_for_var = mc.fits_df_[mc.fits_df_["spec"] == var]
    assert len(fits_for_var) > 0, f"No fits_df_ rows for {var}"
    fits_h = fits_for_var[fits_for_var["horizon"] == h]
    assert len(fits_h) > 0, f"No fits_df_ rows for {var} at horizon {h}"

    # ---- OOS forecast: should match held-out truth in forecasts_df_ ----
    fc_row = mc.forecasts_df_[
        (mc.forecasts_df_["spec"] == var) & (mc.forecasts_df_["horizon"] == h)
    ]
    assert len(fc_row) > 0, f"No forecast row for {var} horizon {h}"
    fc = float(fc_row["value"].iloc[0])
    np.testing.assert_allclose(
        fc,
        y_truth,
        atol=1e-6,
        err_msg=(
            f"[{kind}, h={h}] OOS forecast {fc:.6g} does not match held-out "
            f"truth {y_truth:.6g}."
        ),
    )


# ============================================================================
# Nested ComboSpec: an outer combo whose source is an inner combo must
# include the inner combo's OOS forecast (not silently treat it as 0).
# ============================================================================


def test_midascombo_nested_combo_oos_uses_inner_combo():
    """Outer combo over [inner_combo, x_q] with ``average`` method.

    Expected OOS:
        inner_oos = mean(PMI_oos, IP_oos)
        outer_oos = mean(inner_oos, x_q_oos)
    """
    rng = np.random.default_rng(0)
    n_q = 40
    dates_q = pd.date_range("2010-03-31", periods=n_q, freq="QE")
    dates_m = pd.date_range(
        dates_q[0] - pd.DateOffset(months=6), end=dates_q[-1], freq="ME"
    )

    target = pd.DataFrame(
        {
            "date": dates_q,
            "variable": "y",
            "frequency": "QE",
            "value": rng.standard_normal(n_q),
        }
    )
    regressors = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates_m,
                    "variable": "PMI",
                    "frequency": "ME",
                    "value": rng.standard_normal(len(dates_m)),
                }
            ),
            pd.DataFrame(
                {
                    "date": dates_m,
                    "variable": "IP",
                    "frequency": "ME",
                    "value": rng.standard_normal(len(dates_m)),
                }
            ),
            pd.DataFrame(
                {
                    "date": dates_q,
                    "variable": "x_q",
                    "frequency": "QE",
                    "value": rng.standard_normal(n_q),
                }
            ),
        ],
        ignore_index=True,
    )

    midas_pmi = MidasSpec(variable="PMI", method="almon", n_lags=3)
    midas_ip = MidasSpec(variable="IP", method="almon", n_lags=3)
    ols_xq = OLSSpec(variable="x_q", n_lags=1)

    inner = ComboSpec(name="inner", sources=[midas_pmi, midas_ip], method="average")
    outer = ComboSpec(name="outer", sources=[inner, ols_xq], method="average")

    mc = MidasCombo(combo_specs=outer, horizons=1)
    mc.fit(target, regressors)
    mc.forecast()

    pmi_oos = float(
        mc.forecasts_df_[
            (mc.forecasts_df_["spec"] == "PMI") & (mc.forecasts_df_["horizon"] == 0)
        ]["value"].iloc[0]
    )
    ip_oos = float(
        mc.forecasts_df_[
            (mc.forecasts_df_["spec"] == "IP") & (mc.forecasts_df_["horizon"] == 0)
        ]["value"].iloc[0]
    )
    xq_oos = float(
        mc.forecasts_df_[
            (mc.forecasts_df_["spec"] == "x_q") & (mc.forecasts_df_["horizon"] == 0)
        ]["value"].iloc[0]
    )
    inner_oos = float(
        mc.forecasts_df_[
            (mc.forecasts_df_["spec"] == "inner") & (mc.forecasts_df_["horizon"] == 0)
        ]["value"].iloc[0]
    )
    outer_oos = float(
        mc.forecasts_df_[
            (mc.forecasts_df_["spec"] == "outer") & (mc.forecasts_df_["horizon"] == 0)
        ]["value"].iloc[0]
    )

    np.testing.assert_allclose(inner_oos, 0.5 * (pmi_oos + ip_oos))
    # If the inner combo's OOS value weren't fed into the outer combo,
    # outer_oos would equal 0.5 * (0 + xq_oos) = 0.5 * xq_oos.
    np.testing.assert_allclose(outer_oos, 0.5 * (inner_oos + xq_oos))


# ============================================================================
# Ragged-edge data availability
# ============================================================================


def test_midascombo_ragged_edge():
    """Three monthly regressors with different end-of-sample lengths
    (Feb vs Jan beyond the last target quarter); pipeline must fit all
    individual models, build the average combo, and produce finite
    forecasts for all four series.
    """
    rng = np.random.default_rng(42)
    n_quarters = 20

    target_dates = pd.date_range("2019-03-31", periods=n_quarters, freq="QE")
    target = pd.DataFrame(
        {
            "date": target_dates,
            "variable": "y",
            "frequency": "QE",
            "value": np.cumsum(rng.standard_normal(n_quarters)) + 100,
        }
    )

    def _reg(name, end, base):
        dates = pd.date_range("2019-01-01", end, freq="ME")
        return pd.DataFrame(
            {
                "date": dates,
                "variable": name,
                "frequency": "ME",
                "value": np.cumsum(rng.standard_normal(len(dates))) + base,
            }
        )

    regressors = pd.concat(
        [
            _reg("x1", "2024-02-29", 50),  # 2 months beyond
            _reg("x2", "2024-01-31", 60),  # 1 month beyond
            _reg("x3", "2024-01-31", 70),  # 1 month beyond
        ],
        ignore_index=True,
    )

    midas_specs = [
        MidasSpec(variable=v, method="almon", n_lags=6) for v in ("x1", "x2", "x3")
    ]
    combo = ComboSpec(name="combo_all", sources=midas_specs, method="average")

    mc = MidasCombo(combo_specs=combo, horizons=3)
    mc.fit(target, regressors)
    forecasts = mc.forecast()

    for var in ("x1", "x2", "x3"):
        for h in range(3):
            assert h in mc.midas_models_[var]

    for h in range(3):
        weights = mc.combo_weights_["combo_all"][h]
        assert set(weights) == {"x1", "x2", "x3"}
        for w in weights.values():
            finite = w[np.isfinite(w)]
            if len(finite):
                np.testing.assert_allclose(finite[-1], 1 / 3, rtol=1e-5)

    # forecasts is now a long-format DataFrame
    assert isinstance(forecasts, pd.DataFrame)
    assert "horizon" in forecasts.columns
    assert forecasts["horizon"].max() == 2  # horizons 0, 1, 2 (3 total)

    for var in ("x1", "x2", "x3", "combo_all"):
        # Check forecasts_df_ has rows for each spec
        spec_forecasts = mc.forecasts_df_[mc.forecasts_df_["spec"] == var]
        assert len(spec_forecasts) > 0, f"No forecasts for {var} in forecasts_df_"
        # Should have finite values
        assert spec_forecasts["value"].notna().any()


def test_midascombo_uses_leaf_specific_availability_horizons():
    sample = sample_midas(method="unrestricted")
    vintage = sample.vintages[2]
    target = vintage.target_train.copy()
    target["variable"] = "y"
    target["frequency"] = "QE"

    fast = vintage.regressors.copy()
    fast["variable"] = "fast"
    fast["frequency"] = "ME"
    slow = fast.loc[fast["date"] <= pd.Timestamp("2023-12-31")].copy()
    slow["variable"] = "slow"
    regressors = pd.concat([fast, slow], ignore_index=True)

    combo = ComboSpec(
        name="combo",
        sources=[
            MidasSpec("fast", method="unrestricted", n_lags=5),
            MidasSpec("slow", method="unrestricted", n_lags=5),
        ],
    )
    model = MidasCombo(combo_specs=combo, horizons=1)
    model.fit(target, regressors)
    model.forecast()

    fast_forecast = model.forecasts_df_.loc[
        (model.forecasts_df_["spec"] == "fast") & (model.forecasts_df_["horizon"] == 0),
        "value",
    ].iloc[0]
    slow_forecast = model.forecasts_df_.loc[
        (model.forecasts_df_["spec"] == "slow") & (model.forecasts_df_["horizon"] == 0),
        "value",
    ].iloc[0]
    expected = sample.truth.target_value

    np.testing.assert_allclose(fast_forecast, expected, atol=2e-5)
    np.testing.assert_allclose(slow_forecast, expected, atol=2e-5)


# ============================================================================
# OLSSpec path
# ============================================================================


def _make_mixed_freq_data(n_obs: int = 60, seed: int = 0):
    """Quarterly target, one monthly MIDAS regressor, one quarterly OLS regressor."""
    rng = np.random.default_rng(seed)
    dates_q = pd.date_range("2000-03-31", periods=n_obs, freq="QE")
    dates_m = pd.date_range(
        dates_q[0] - pd.DateOffset(months=6), end=dates_q[-1], freq="ME"
    )

    x_q = rng.standard_normal(n_obs)
    y = 1.0 + 2.0 * x_q + 0.1 * rng.standard_normal(n_obs)
    x_m = rng.standard_normal(len(dates_m))

    target = pd.DataFrame(
        {"date": dates_q, "variable": "y", "frequency": "QE", "value": y}
    )
    reg_m = pd.DataFrame(
        {"date": dates_m, "variable": "x_m", "frequency": "ME", "value": x_m}
    )
    reg_q = pd.DataFrame(
        {"date": dates_q, "variable": "x_q", "frequency": "QE", "value": x_q}
    )
    regressors = pd.concat([reg_m, reg_q], ignore_index=True)
    return target, regressors, x_q


def test_ols_spec_recovers_coefficients():
    """OLSSpec should recover the DGP intercept and slope from y = 1 + 2*x + eps."""
    target, regressors, _ = _make_mixed_freq_data(n_obs=200, seed=7)
    midas_spec = MidasSpec(variable="x_m", method="almon", n_lags=3)
    ols_spec = OLSSpec(variable="x_q", n_lags=1)
    combo = ComboSpec(name="combo", sources=[midas_spec, ols_spec], method="average")

    mc = MidasCombo(combo_specs=combo, horizons=1)
    mc.fit(target, regressors)

    fit = mc.ols_models_["x_q"][0]
    np.testing.assert_allclose(fit.intercept, 1.0, atol=0.05)
    np.testing.assert_allclose(fit.coef, [2.0], atol=0.05)

    fitted = mc.fitted_["x_q"][0]
    assert len(fitted) == len(target)
    assert np.isfinite(fitted).all()

    # Check fits_df_ contains OLS fits
    fits_x_q = mc.fits_df_[mc.fits_df_["spec"] == "x_q"]
    assert len(fits_x_q) > 0, "No fits for x_q in fits_df_"


def test_ols_spec_in_combo_and_forecast():
    """OLSSpec works as a source for ComboSpec and produces OOS forecasts."""
    target, regressors, _ = _make_mixed_freq_data(n_obs=80, seed=3)
    midas_spec = MidasSpec(variable="x_m", method="almon", n_lags=3)
    ols_spec = OLSSpec(variable="x_q", n_lags=1)
    combo = ComboSpec(
        name="combo_avg", sources=[midas_spec, ols_spec], method="average"
    )

    mc = MidasCombo(combo_specs=combo, horizons=2)
    mc.fit(target, regressors)
    forecasts = mc.forecast()

    # forecasts is long-format; ``forecast`` is the deprecated alias of ``value``.
    assert isinstance(forecasts, pd.DataFrame)
    assert list(forecasts.columns) == ["date", "horizon", "spec", "value", "forecast"]
    assert (forecasts["forecast"] == forecasts["value"]).all()

    # Check all specs are present
    specs_in_forecasts = set(forecasts["spec"].unique())
    assert specs_in_forecasts == {"x_m", "x_q", "combo_avg"}

    # Use forecasts_df_ for the key check
    x_m_fc = float(
        mc.forecasts_df_[
            (mc.forecasts_df_["spec"] == "x_m") & (mc.forecasts_df_["horizon"] == 0)
        ]["value"].iloc[0]
    )
    x_q_fc = float(
        mc.forecasts_df_[
            (mc.forecasts_df_["spec"] == "x_q") & (mc.forecasts_df_["horizon"] == 0)
        ]["value"].iloc[0]
    )
    combo_fc = float(
        mc.forecasts_df_[
            (mc.forecasts_df_["spec"] == "combo_avg")
            & (mc.forecasts_df_["horizon"] == 0)
        ]["value"].iloc[0]
    )

    expected = 0.5 * (x_m_fc + x_q_fc)
    np.testing.assert_allclose(combo_fc, expected)


def test_ols_spec_wrong_frequency_raises():
    """OLSSpec referencing an ME variable must raise."""
    target, regressors, _ = _make_mixed_freq_data(n_obs=40, seed=1)
    ols_spec = OLSSpec(variable="x_m", n_lags=1)  # x_m is ME, not QE
    combo = ComboSpec(name="combo", sources=[ols_spec], method="average")
    mc = MidasCombo(combo_specs=combo, horizons=1)
    try:
        mc.fit(target, regressors)
    except ValueError as exc:
        assert "frequency" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for ME variable in OLSSpec")


def test_residuals_from_fits_df_exact_recovery():
    """Residuals computed from fits_df_ and observations match fit.residuals exactly."""
    target, regressors, _ = _make_mixed_freq_data(n_obs=100, seed=42)
    midas_spec = MidasSpec(variable="x_m", method="almon", n_lags=3)
    ols_spec = OLSSpec(variable="x_q", n_lags=1)
    combo = ComboSpec(name="combo", sources=[midas_spec, ols_spec], method="average")

    mc = MidasCombo(combo_specs=combo, horizons=2)
    mc.fit(target, regressors)

    # Extract target values as a Series
    target_vals = pd.Series(
        target["value"].values, index=pd.DatetimeIndex(target["date"])
    )

    # For each spec and horizon, verify residuals can be replicated from fits_df_
    for spec in [midas_spec, ols_spec]:
        var = spec.variable
        for h in range(mc.horizons):
            # Get fitted values from fits_df_
            fits_rows = mc.fits_df_[
                (mc.fits_df_["spec"] == var) & (mc.fits_df_["horizon"] == h)
            ]
            if len(fits_rows) == 0:
                continue

            fitted_dates = pd.to_datetime(fits_rows["date"])
            fitted_vals = fits_rows["value"].values

            # Get observed values aligned with fitted dates
            obs_vals = target_vals.loc[fitted_dates].values

            # Compute residuals from fits_df_ and observations
            residuals_from_fits_df = obs_vals - fitted_vals

            # Get residuals from fit object
            if var in mc.midas_models_ and h in mc.midas_models_[var]:
                fit = mc.midas_models_[var][h]
            elif var in mc.ols_models_ and h in mc.ols_models_[var]:
                fit = mc.ols_models_[var][h]
            else:
                continue

            # Compare: residuals should match to machine precision
            np.testing.assert_allclose(
                residuals_from_fits_df,
                fit.residuals,
                rtol=1e-14,
                atol=1e-14,
                err_msg=(
                    f"Residuals from fits_df_ do not match fit.residuals "
                    f"for {var} at horizon {h}"
                ),
            )


# ============================================================================
# Forecast decomposition: contributions must sum to the combo forecast.
# ============================================================================


def _combo_decomp_fixture(seed: int = 7):
    """Build a nested MidasCombo (outer over [inner_combo, x_q]) fitted on
    random data. Returns the fitted ``MidasCombo``."""
    rng = np.random.default_rng(seed)
    n_q = 48
    dates_q = pd.date_range("2010-03-31", periods=n_q, freq="QE")
    dates_m = pd.date_range(
        dates_q[0] - pd.DateOffset(months=6), end=dates_q[-1], freq="ME"
    )

    target = pd.DataFrame(
        {
            "date": dates_q,
            "variable": "y",
            "frequency": "QE",
            "value": rng.standard_normal(n_q),
        }
    )
    regressors = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates_m,
                    "variable": "PMI",
                    "frequency": "ME",
                    "value": rng.standard_normal(len(dates_m)),
                }
            ),
            pd.DataFrame(
                {
                    "date": dates_m,
                    "variable": "IP",
                    "frequency": "ME",
                    "value": rng.standard_normal(len(dates_m)),
                }
            ),
            pd.DataFrame(
                {
                    "date": dates_q,
                    "variable": "x_q",
                    "frequency": "QE",
                    "value": rng.standard_normal(n_q),
                }
            ),
        ],
        ignore_index=True,
    )

    midas_pmi = MidasSpec(variable="PMI", method="almon", n_lags=3)
    midas_ip = MidasSpec(variable="IP", method="almon", n_lags=3)
    ols_xq = OLSSpec(variable="x_q", n_lags=1)

    inner = ComboSpec(name="inner", sources=[midas_pmi, midas_ip], method="average")
    outer = ComboSpec(name="outer", sources=[inner, ols_xq], method="average")

    mc = MidasCombo(combo_specs=outer, horizons=3)
    mc.fit(target, regressors)
    mc.forecast()
    return mc


class TestMidasComboForecastDecomp:
    """Combo decomposition contributions must sum to the combo forecast."""

    def test_columns_and_sum_to_root_forecast(self):
        mc = _combo_decomp_fixture()
        decomp = mc.forecast_decomp()  # defaults to the root combo "outer"

        assert list(decomp.columns) == [
            "horizon",
            "date",
            "component",
            "contribution",
            "weight",
        ]

        for h in range(mc.horizons):
            outer_oos = float(
                mc.forecasts_df_[
                    (mc.forecasts_df_["spec"] == "outer")
                    & (mc.forecasts_df_["horizon"] == h)
                ]["value"].iloc[0]
            )
            s = decomp.loc[decomp["horizon"] == h, "contribution"].sum()
            np.testing.assert_allclose(s, outer_oos, atol=1e-9)

    def test_components_reference_underlying_models(self):
        mc = _combo_decomp_fixture()
        decomp = mc.forecast_decomp("outer")
        components = set(decomp["component"])
        # Every leaf model surfaces through the "model::component" labels.
        prefixes = {c.split("::")[0] for c in components}
        assert {"PMI", "IP", "x_q"}.issubset(prefixes)

    def test_inner_combo_decomp_sums(self):
        mc = _combo_decomp_fixture()
        decomp = mc.forecast_decomp("inner")
        for h in range(mc.horizons):
            inner_oos = float(
                mc.forecasts_df_[
                    (mc.forecasts_df_["spec"] == "inner")
                    & (mc.forecasts_df_["horizon"] == h)
                ]["value"].iloc[0]
            )
            s = decomp.loc[decomp["horizon"] == h, "contribution"].sum()
            np.testing.assert_allclose(s, inner_oos, atol=1e-9)

    def test_leaf_model_decomp_sums(self):
        mc = _combo_decomp_fixture()
        decomp = mc.forecast_decomp("x_q")
        for h in range(mc.horizons):
            xq_oos = float(
                mc.forecasts_df_[
                    (mc.forecasts_df_["spec"] == "x_q")
                    & (mc.forecasts_df_["horizon"] == h)
                ]["value"].iloc[0]
            )
            s = decomp.loc[decomp["horizon"] == h, "contribution"].sum()
            np.testing.assert_allclose(s, xq_oos, atol=1e-9)


def _combo_vintage_model(method="average"):
    sample = sample_midas_combo()
    specs = [
        MidasSpec(variable=name, method="unrestricted", n_lags=3)
        for name in sample.leaves
    ]
    combo = ComboSpec(name="combo", sources=specs, method=method)
    model = MidasCombo(combo_specs=combo, horizons=1)
    model.fit(sample.target_train, sample.regressors)
    return sample, model


def test_each_component_parameters_match_independent_truth():
    sample, model = _combo_vintage_model()

    for name, truth in sample.leaves.items():
        fit = model.midas_models_[name][truth.horizon]
        np.testing.assert_allclose(fit.alpha, sample.alpha, atol=2e-5)
        np.testing.assert_allclose(
            fit.beta * fit.weights,
            truth.coefficients,
            atol=2e-5,
        )


def test_each_component_uses_availability_horizon():
    sample, model = _combo_vintage_model()

    assert sample.leaves["fast"].latest_regressor_date == pd.Timestamp("2024-03-31")
    assert sample.leaves["slow"].latest_regressor_date == pd.Timestamp("2023-12-31")
    assert sample.leaves["fast"].horizon == 0
    assert sample.leaves["slow"].horizon == 1
    assert set(model.midas_models_["slow"]) >= {0, 1}


def test_each_component_held_out_forecast_matches_independent_truth():
    sample, model = _combo_vintage_model()
    forecasts = model.forecast()

    for name, truth in sample.leaves.items():
        value = forecasts.loc[
            (forecasts["spec"] == name) & (forecasts["horizon"] == 0), "value"
        ].iloc[0]
        np.testing.assert_allclose(value, truth.forecast, atol=2e-5)


def test_average_weights_and_forecast_match_independent_truth():
    sample, model = _combo_vintage_model()
    forecasts = model.forecast()
    combo_forecast = forecasts.loc[
        (forecasts["spec"] == "combo") & (forecasts["horizon"] == 0), "value"
    ].iloc[0]
    expected = np.mean([truth.forecast for truth in sample.leaves.values()])

    np.testing.assert_allclose(combo_forecast, expected, atol=2e-5)
    for source in sample.leaves:
        weights = model.combo_weights_["combo"][0][source]
        np.testing.assert_allclose(weights[-1], 0.5)


@pytest.mark.parametrize("vintage_index", range(4))
def test_mixed_estimators_recover_each_nowcast_vintage(vintage_index):
    sample = sample_mixed_model_combo()
    vintage = sample.vintages[vintage_index]
    truth = sample.truth
    expected_horizons = [
        {"midas": 1, "multi": 1, "quarterly": 2},
        {"midas": 0, "multi": 0, "quarterly": 1},
        {"midas": 0, "multi": 0, "quarterly": 1},
        {"midas": 0, "multi": 0, "quarterly": 1},
    ]
    assert vintage.horizons == expected_horizons[vintage_index]

    midas_spec = MidasSpec("midas", method="unrestricted", n_lags=3)
    multi_spec = MultiMidasSpec(
        "multi",
        variables=["multi_fast", "multi_slow"],
        method="unrestricted",
        n_lags=3,
    )
    ols_spec = OLSSpec("quarterly", n_lags=1)
    combo = ComboSpec(
        "average",
        sources=[midas_spec, multi_spec, ols_spec],
        method="average",
    )
    model = MidasCombo(combo_specs=combo, horizons=1)
    model.fit(vintage.target_train, vintage.regressors)
    forecasts = model.forecast()

    assert model.leaf_forecast_horizons_ == {
        name: [horizon] for name, horizon in vintage.horizons.items()
    }
    target = vintage.target_train[["date", "value"]]

    midas_regressors = vintage.regressors.loc[
        vintage.regressors["variable"] == "midas", ["date", "value"]
    ]
    midas_horizon = vintage.horizons["midas"]
    standalone_midas = MIDAS(
        method="unrestricted", n_lags=3, horizons=[midas_horizon]
    ).fit(target, midas_regressors)
    combo_midas = model.midas_models_["midas"][midas_horizon]
    direct_midas = standalone_midas.fits_[midas_horizon]
    np.testing.assert_allclose(combo_midas.alpha, direct_midas.alpha, atol=1e-12)
    np.testing.assert_allclose(combo_midas.weights, direct_midas.weights, atol=1e-12)

    multi_regressors = vintage.regressors.loc[
        vintage.regressors["variable"].isin(["multi_fast", "multi_slow"]),
        ["date", "variable", "value"],
    ]
    multi_horizon = vintage.horizons["multi"]
    standalone_multi = MultiMIDAS(
        ["multi_fast", "multi_slow"],
        method="unrestricted",
        n_lags=3,
        horizons=[multi_horizon],
    ).fit(target, multi_regressors)
    combo_multi = model.multi_midas_models_["multi"][multi_horizon]
    direct_multi = standalone_multi.fits_[multi_horizon]
    np.testing.assert_allclose(combo_multi.alpha, direct_multi.alpha, atol=1e-12)
    for variable in ("multi_fast", "multi_slow"):
        np.testing.assert_allclose(
            combo_multi.variable_fits[variable].weights,
            direct_multi.variable_fits[variable].weights,
            atol=1e-12,
        )

    ols_regressors = vintage.regressors.loc[
        vintage.regressors["variable"] == "quarterly", ["date", "value"]
    ]
    ols_horizon = vintage.horizons["quarterly"]
    standalone_ols = OLS(n_lags=1, horizons=[ols_horizon]).fit(target, ols_regressors)
    combo_ols = model.ols_models_["quarterly"][ols_horizon]
    direct_ols = standalone_ols.fits_[ols_horizon]
    np.testing.assert_allclose(combo_ols.intercept, direct_ols.intercept, atol=1e-12)
    np.testing.assert_allclose(combo_ols.coef, direct_ols.coef, atol=1e-12)

    expected_scale = truth.persistence
    np.testing.assert_allclose(
        combo_midas.weights,
        expected_scale**midas_horizon * truth.weights,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        combo_multi.variable_fits["multi_fast"].weights,
        expected_scale**multi_horizon * truth.weights,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        combo_multi.variable_fits["multi_slow"].weights,
        np.zeros_like(truth.weights),
        atol=2e-10,
    )
    np.testing.assert_allclose(
        combo_ols.coef, [expected_scale**ols_horizon], atol=2e-10
    )

    leaf_values = forecasts.loc[
        forecasts["spec"].isin(["midas", "multi", "quarterly"]), "value"
    ]
    np.testing.assert_allclose(leaf_values, truth.target_value, atol=2e-10)
    combo_value = forecasts.loc[forecasts["spec"] == "average", "value"].iloc[0]
    np.testing.assert_allclose(combo_value, leaf_values.mean(), atol=1e-12)
    for source in ("midas", "multi", "quarterly"):
        weights = model.combo_weights_["average"][0][source]
        np.testing.assert_allclose(weights[np.isfinite(weights)][-1], 1 / 3)


def test_weighted_combination_uses_leaf_release_horizons():
    sample, model = _combo_vintage_model(method="mse")
    forecasts = model.forecast()
    weights = {
        source: model.combo_weights_["combo"][0][source][-1] for source in sample.leaves
    }
    combo_forecast = forecasts.loc[
        (forecasts["spec"] == "combo") & (forecasts["horizon"] == 0), "value"
    ].iloc[0]
    expected = sum(
        weights[source] * truth.forecast for source, truth in sample.leaves.items()
    )

    assert sample.leaves["fast"].horizon == 0
    assert sample.leaves["slow"].horizon == 1
    np.testing.assert_allclose(sum(weights.values()), 1.0)
    np.testing.assert_allclose(combo_forecast, expected, atol=2e-5)


def test_weighted_combination_fits_from_leaf_specific_histories():
    _sample, model = _combo_vintage_model(method="mse")
    slow_h0 = pd.Series(np.full(len(model.target_), 1e6), index=model.target_.index)
    model.fitted_["slow"][0] = slow_h0
    model._fit_combo(model.combo_specs, horizon=0)

    weights = model.combo_weights_["combo"][0]
    np.testing.assert_allclose(weights["fast"][-1], 0.5)
    np.testing.assert_allclose(weights["slow"][-1], 0.5)


def test_nested_combination_matches_independent_truth():
    sample = sample_midas_combo()
    specs = [
        MidasSpec(variable=name, method="unrestricted", n_lags=3)
        for name in sample.leaves
    ]
    inner = ComboSpec(name="inner", sources=specs, method="average")
    outer = ComboSpec(name="outer", sources=[inner, specs[0]], method="average")
    model = MidasCombo(combo_specs=outer, horizons=1)
    model.fit(sample.target_train, sample.regressors)
    forecasts = model.forecast()

    fast_forecast = sample.leaves["fast"].forecast
    slow_forecast = sample.leaves["slow"].forecast
    inner_expected = (fast_forecast + slow_forecast) / 2
    outer_expected = (inner_expected + fast_forecast) / 2
    inner_forecast = forecasts.loc[
        (forecasts["spec"] == "inner") & (forecasts["horizon"] == 0), "value"
    ].iloc[0]
    outer_forecast = forecasts.loc[
        (forecasts["spec"] == "outer") & (forecasts["horizon"] == 0), "value"
    ].iloc[0]

    np.testing.assert_allclose(inner_forecast, inner_expected, atol=2e-5)
    np.testing.assert_allclose(outer_forecast, outer_expected, atol=2e-5)


def test_combo_decomposition_sums_to_independent_forecast():
    sample, model = _combo_vintage_model()
    forecasts = model.forecast()
    decomposition = model.forecast_decomp()
    combo_forecast = forecasts.loc[
        (forecasts["spec"] == "combo") & (forecasts["horizon"] == 0), "value"
    ].iloc[0]

    assert set(pd.to_datetime(decomposition["date"])) == {sample.target_date}
    np.testing.assert_allclose(
        decomposition["contribution"].sum(), combo_forecast, atol=1e-9
    )


def test_multi_midas_leaf_matches_independent_vintage_truth():
    sample = sample_multi_midas()
    vintage = sample.vintages[-1]
    target = vintage.target_train.copy()
    target["variable"] = "y"
    target["frequency"] = "QE"
    frequencies = {spec.variable: spec.frequency for spec in sample.truth.variables}
    regressors = vintage.regressors.copy()
    regressors["frequency"] = regressors["variable"].map(frequencies)

    multi = MultiMidasSpec(
        name="joint",
        variables=list(sample.truth.variables),
        method="unrestricted",
        n_lags=2,
    )
    combo = ComboSpec(name="combo", sources=[multi], method="average")
    model = MidasCombo(combo_specs=combo, horizons=1)
    model.fit(target, regressors)
    forecasts = model.forecast()

    fit = model.multi_midas_models_["joint"][vintage.horizon]
    np.testing.assert_allclose(fit.alpha, sample.truth.alpha, atol=2e-5)
    for variable in ("survey", "activity"):
        variable_fit = fit.variable_fits[variable]
        expected = sample.truth.variables_truth[variable].effective_coefficients[
            vintage.horizon
        ]
        np.testing.assert_allclose(
            variable_fit.beta * variable_fit.weights,
            expected,
            atol=2e-5,
        )

    value = forecasts.loc[
        (forecasts["spec"] == "joint") & (forecasts["horizon"] == 0), "value"
    ].iloc[0]
    np.testing.assert_allclose(value, vintage.independent_forecast, atol=2e-5)


def test_filtered_sources_are_undefined_during_forecast():
    target, regressors, _ = sample_combo_data(
        n_quarters=40,
        n_lags=6,
        noise=0.0,
        seed=17,
        monthly_vars=["x1", "x2", "x3"],
        quarterly_vars=[],
        outlier_date=None,
    )
    inner = ComboSpec(
        name="inner",
        sources=[MidasSpec("x1"), MidasSpec("x2")],
        minimum_sample_size=100,
    )
    outer = ComboSpec(name="outer", sources=[inner, MidasSpec("x3")])
    model = MidasCombo(combo_specs=outer, horizons=1)
    model.fit(target, regressors)
    forecasts = model.forecast()

    inner_value = forecasts.loc[
        (forecasts["spec"] == "inner") & (forecasts["horizon"] == 0), "value"
    ].iloc[0]
    outer_value = forecasts.loc[
        (forecasts["spec"] == "outer") & (forecasts["horizon"] == 0), "value"
    ].iloc[0]
    x3_value = forecasts.loc[
        (forecasts["spec"] == "x3") & (forecasts["horizon"] == 0), "value"
    ].iloc[0]

    assert np.isnan(inner_value)
    np.testing.assert_allclose(outer_value, x3_value)
    decomposition = model.forecast_decomp("outer")
    np.testing.assert_allclose(decomposition["contribution"].sum(), outer_value)


def test_counterfactual_decomposition_uses_supplied_regressors():
    sample, model = _combo_vintage_model()
    counterfactual = sample.regressors.copy()
    counterfactual.loc[counterfactual["variable"] == "fast", "value"] *= 2

    decomposition = model.forecast_decomp(
        "fast", regressors=counterfactual, aggregate=True
    )
    expected = model._forecast_models(sample.target_date, regressors=counterfactual)[
        "fast"
    ]

    assert len(decomposition) == 1
    np.testing.assert_allclose(decomposition.iloc[0]["contribution"], expected)


def test_incomplete_leaf_lags_return_nan_forecasts():
    sample, model = _combo_vintage_model()
    incomplete = sample.regressors.copy()
    incomplete.loc[
        (incomplete["variable"] == "fast")
        & (incomplete["date"] == pd.Timestamp("2024-01-31")),
        "value",
    ] = np.nan

    forecasts = model._forecast_models(sample.target_date, regressors=incomplete)

    assert np.isnan(forecasts["fast"])


def test_unavailable_long_horizon_does_not_discard_shorter_leaf_models():
    target_dates = pd.date_range("2020-03-31", periods=4, freq="QE")
    regressor_dates = pd.date_range("2019-01-31", "2021-03-31", freq="ME")
    target = pd.DataFrame(
        {
            "date": target_dates,
            "variable": "y",
            "frequency": "QE",
            "value": np.arange(len(target_dates), dtype=float),
        }
    )
    regressors = pd.DataFrame(
        {
            "date": regressor_dates,
            "variable": "x",
            "frequency": "ME",
            "value": np.arange(len(regressor_dates), dtype=float),
        }
    )
    model = MidasCombo(
        ComboSpec("combo", [MidasSpec("x", n_lags=3)], minimum_sample_size=1),
        horizons=5,
    )

    model.fit(target, regressors)

    assert set(model.midas_models_["x"]) == {0, 1, 2, 3}
    assert len(model.forecast()) == 10


def test_all_nonfinite_leaf_is_available_as_nan_for_every_step():
    target_dates = pd.date_range("2020-03-31", periods=8, freq="QE")
    regressor_dates = pd.date_range("2018-01-31", "2020-12-31", freq="ME")
    target = pd.DataFrame(
        {
            "date": target_dates,
            "variable": "y",
            "frequency": "QE",
            "value": np.arange(len(target_dates), dtype=float),
        }
    )
    regressors = pd.DataFrame(
        {
            "date": regressor_dates,
            "variable": "x",
            "frequency": "ME",
            "value": np.nan,
        }
    )
    model = MidasCombo(
        ComboSpec("combo", [MidasSpec("x", n_lags=3)], minimum_sample_size=1),
        horizons=2,
    )

    model.fit(target, regressors)
    forecasts = model.forecast()

    assert set(forecasts["spec"]) == {"x", "combo"}
    assert forecasts["value"].isna().all()


def test_failed_leaf_remains_in_public_forecast_table():
    target, regressors, _ = sample_combo_data(
        n_quarters=40,
        n_lags=6,
        noise=0.0,
        seed=17,
        monthly_vars=["good", "bad"],
        quarterly_vars=[],
        outlier_date=None,
    )
    regressors.loc[regressors["variable"] == "bad", "value"] = np.nan
    combo = ComboSpec(
        name="combo",
        sources=[MidasSpec("good"), MidasSpec("bad")],
        minimum_sample_size=1,
    )
    model = MidasCombo(combo_specs=combo, horizons=1)

    model.fit(target, regressors)
    forecasts = model.forecast()

    assert set(forecasts["spec"]) == {"good", "bad", "combo"}
    assert forecasts.loc[forecasts["spec"] == "bad", "value"].isna().all()


def test_refit_clears_previous_forecasts():
    target, regressors, _ = sample_combo_data(
        n_quarters=20,
        n_lags=6,
        noise=0.0,
        seed=19,
        monthly_vars=["x"],
        quarterly_vars=[],
        outlier_date=None,
    )
    model = MidasCombo(
        ComboSpec("combo", [MidasSpec("x")], minimum_sample_size=1),
        horizons=1,
    )
    model.fit(target, regressors).forecast()
    old_forecast = model.forecasts_df_.copy()

    model.fit(target.iloc[:-1], regressors)

    assert model.forecasts_df_.empty
    assert model.fits_and_forecasts_df_.empty
    assert not old_forecast.empty
