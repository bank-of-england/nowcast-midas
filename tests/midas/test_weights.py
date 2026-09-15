"""Tests for combo_weights functions and the MidasCombo pipeline."""

import numpy as np
import pandas as pd
import pytest

from nowcast_midas.combo_weights import (
    _filter_sources,
    clipped_ols,
    constrained_least_squares,
    fit_average,
    fit_error_based_weights,
    fit_regression_weights,
)
from nowcast_midas.midas_combo import MidasCombo
from nowcast_midas.specs import ComboSpec, MidasSpec

# ======================================================================
#  Fixtures
# ======================================================================

N_QUARTERS = 40
N_MONTHS = N_QUARTERS * 3 + 6


@pytest.fixture()
def target() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-03-31", periods=N_QUARTERS, freq="QE")
    return pd.DataFrame(
        {
            "date": dates,
            "variable": "GDP",
            "frequency": "QE",
            "value": np.cumsum(rng.standard_normal(N_QUARTERS)),
        }
    )


@pytest.fixture()
def regressors() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2014-10-31", periods=N_MONTHS, freq="ME")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "variable": name,
                    "frequency": "ME",
                    "value": rng.standard_normal(N_MONTHS),
                }
            )
            for name in ("PMI", "IP")
        ],
        ignore_index=True,
    )


# ======================================================================
#  combo_weights – unit tests
# ======================================================================


class TestFitAverage:
    def test_equal_weights_and_nan_handling(self):
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        combined, weights = fit_average(pd.DataFrame({"a": a, "b": b}))
        np.testing.assert_allclose(combined, [2.0, 2.0, 2.0])
        # Weights are per-observation: where a is NaN, its weight is 0
        np.testing.assert_allclose(weights["a"], [0.5, 0.0, 0.5])
        np.testing.assert_allclose(weights["b"], [0.5, 1.0, 0.5])


class TestFilterSources:
    def test_removes_sources_below_minimum_sample_size(self):
        source_fitted = pd.DataFrame(
            {
                "kept": [1.0, 2.0, np.nan, 4.0],
                "removed": [1.0, np.nan, np.nan, np.nan],
            }
        )

        filtered = _filter_sources(source_fitted, minimum_sample_size=3)

        assert list(filtered.columns) == ["kept"]

    def test_combo_spec_defaults_to_ten(self):
        spec = ComboSpec(name="combo")

        assert spec.minimum_sample_size == 10


class TestErrorBasedWeights:
    def test_better_model_gets_higher_weight(self):
        target = pd.Series(np.arange(8, dtype=float))
        perfect = target.to_numpy().copy()
        biased = target.to_numpy() + 2.0
        _, weights = fit_error_based_weights(
            target,
            pd.DataFrame({"perfect": perfect, "biased": biased}),
            method="rmse",
            window=4,
            discount_rate=1.0,
        )
        assert weights["perfect"][-1] > weights["biased"][-1]

    @pytest.mark.parametrize("method", ["rmse", "mse", "mae"])
    def test_all_methods_produce_valid_output(self, method):
        target = pd.Series(np.arange(10, dtype=float))
        combined, weights = fit_error_based_weights(
            target,
            pd.DataFrame({"x": target + 0.1, "y": target - 0.2}),
            method=method,
            window=4,
            discount_rate=0.95,
        )
        assert combined.shape == (10,)
        assert set(weights) == {"x", "y"}

    def test_weights_are_in_sample_only(self):
        target = pd.Series([1.0, 2.0])
        _, weights = fit_error_based_weights(
            target,
            pd.DataFrame({"a": [1.1, 2.1], "b": [1.2, 2.2]}),
            method="rmse",
            window=4,
            discount_rate=1.0,
        )
        assert len(weights["a"]) == len(target) + 1
        assert weights["a"][len(target) - 1] == weights["b"][len(target) - 1]

    def test_drops_residual_rows_with_any_missing_source(self):
        target = pd.Series([0.0, 1.0, 2.0, 3.0])
        source_fitted = pd.DataFrame(
            {
                "a": [1.0, np.nan, 5.0, 4.0],
                "b": [2.0, 101.0, 4.0, 5.0],
            }
        )

        _, weights = fit_error_based_weights(
            target,
            source_fitted,
            method="mse",
            window=3,
            discount_rate=1.0,
        )

        np.testing.assert_allclose(weights["a"][3], 0.5)
        np.testing.assert_allclose(weights["b"][3], 0.5)

    def test_zeroes_and_renormalises_unavailable_current_sources(self):
        target = pd.Series([0.0, 1.0, 2.0])
        source_fitted = pd.DataFrame(
            {
                "a": [0.0, 1.0, np.nan],
                "b": [10.0, 11.0, 12.0],
            }
        )

        combined, weights = fit_error_based_weights(
            target,
            source_fitted,
            method="mse",
            window=2,
            discount_rate=1.0,
        )

        assert weights["a"][2] == 0.0
        assert weights["b"][2] == 1.0
        assert combined[-1] == 12.0

    def test_window_selects_latest_complete_rows(self):
        target = pd.Series(np.zeros(6))
        source_fitted = pd.DataFrame(
            {
                "a": [1.0, 1.0, 1.0, np.nan, 1.0, 1.0],
                "b": [1.0, 1.0, 1.0, 1.0, 2.0, 1.0],
            }
        )

        _, weights = fit_error_based_weights(
            target,
            source_fitted,
            method="mse",
            window=3,
            discount_rate=1.0,
        )

        np.testing.assert_allclose(
            [weights["a"][5], weights["b"][5]],
            [2 / 3, 1 / 3],
        )

    def test_finite_window_uses_available_rows_during_warmup(self):
        target = pd.Series(np.arange(4, dtype=float))
        source_fitted = pd.DataFrame(
            {
                "perfect": target,
                "biased": target + 1.0,
            }
        )

        _, weights = fit_error_based_weights(
            target,
            source_fitted,
            method="mse",
            window=3,
            discount_rate=1.0,
        )

        assert weights["perfect"][1] == weights["biased"][1] == 0.5
        assert weights["perfect"][2] == weights["biased"][2] == 0.5

    def test_regression_waits_for_minimum_common_sample(self):
        target = pd.Series([0.5, 1.0, 1.5, 2.0])
        source_fitted = pd.DataFrame(
            {
                "a": [0.0, 1.0, 2.0, 3.0],
                "b": [2.0, 1.0, 0.0, -1.0],
            }
        )

        _, weights = fit_regression_weights(
            target,
            source_fitted,
            method="constrained_ls",
            window=3,
            minimum_sample_size=3,
        )

        np.testing.assert_allclose(
            [weights["a"][1], weights["a"][2]],
            [0.5, 0.5],
        )
        assert weights["a"][3] > weights["b"][3]


class TestFitWeightsRegression:
    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(99)
        target = pd.Series(rng.standard_normal(20))
        source_df = pd.DataFrame(
            {
                "a": target + rng.normal(0, 0.1, 20),
                "b": target + rng.normal(0, 0.5, 20),
            }
        )
        _, weights = fit_regression_weights(
            target,
            source_df,
            method="constrained_ls",
            window=10,
        )
        for t in range(20):
            w_a, w_b = weights["a"][t], weights["b"][t]
            if np.isfinite(w_a):
                np.testing.assert_allclose(w_a + w_b, 1.0, atol=1e-6)


class TestConstrainedLeastSquares:
    def test_weights_non_negative_and_sum_to_one(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 3))
        y = X @ [0.5, 0.3, 0.2] + rng.normal(0, 0.1, 30)
        w = constrained_least_squares(X, y)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
        assert (w >= -1e-10).all()

    def test_empty_input_returns_nan(self):
        w = constrained_least_squares(np.zeros((0, 2)), np.zeros(0))
        assert np.isnan(w).all()


class TestClippedOls:
    def test_weights_non_negative_and_sum_to_one(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 3))
        y = X @ [0.5, 0.3, 0.2] + rng.normal(0, 0.1, 30)
        w = clipped_ols(X, y)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
        assert (w >= -1e-10).all()

    def test_empty_input_returns_nan(self):
        w = clipped_ols(np.zeros((0, 2)), np.zeros(0))
        assert np.isnan(w).all()


# ======================================================================
#  MidasCombo – validation
# ======================================================================


class TestMidasComboValidation:
    def test_missing_target_column_raises(self, regressors):
        bad_target = pd.DataFrame({"date": [1], "value": [1]})
        midas_pmi = MidasSpec("PMI")
        model = MidasCombo(
            combo_specs=ComboSpec("c", [midas_pmi]),
        )
        with pytest.raises(ValueError, match="missing columns"):
            model.fit(bad_target, regressors)

    def test_non_dataframe_input_raises_typeerror(self, regressors):
        midas_pmi = MidasSpec("PMI")
        model = MidasCombo(combo_specs=ComboSpec("c", [midas_pmi]))
        with pytest.raises(TypeError, match="target must be a pandas DataFrame"):
            model.fit({"date": [1], "value": [1]}, regressors)

    def test_unknown_variable_raises(self, target, regressors):
        midas_missing = MidasSpec("MISSING")
        model = MidasCombo(
            combo_specs=ComboSpec("c", [midas_missing]),
        )
        with pytest.raises(ValueError, match="not found in regressors"):
            model.fit(target, regressors)

    def test_invalid_combo_method_raises(self):
        with pytest.raises(ValueError, match="ComboSpec method"):
            midas_pmi = MidasSpec("PMI")
            MidasCombo(
                combo_specs=ComboSpec("c", [midas_pmi], method="bad"),
            )

    def test_allow_missing_indicators_argument_was_removed(self):
        with pytest.raises(TypeError, match="allow_missing_indicators"):
            ComboSpec(name="c", allow_missing_indicators=True)

    def test_unresolved_source_raises(self, target, regressors):
        midas_pmi = MidasSpec("PMI")
        model = MidasCombo(
            combo_specs=ComboSpec("c", [midas_pmi, "GHOST"]),
        )
        with pytest.raises(ValueError, match="unknown source"):
            model.fit(target, regressors)


# ======================================================================
#  MidasCombo – fit
# ======================================================================


class TestMidasComboFit:
    def test_average_pipeline(self, target, regressors):
        midas_pmi = MidasSpec("PMI")
        midas_ip = MidasSpec("IP")
        model = MidasCombo(
            combo_specs=ComboSpec("avg", [midas_pmi, midas_ip], method="average"),
        )
        model.fit(target, regressors)

        assert "PMI" in model.fitted_
        assert "IP" in model.fitted_
        assert "avg" in model.fitted_
        # fitted_ is now dict[str, dict[int, ndarray]] — check horizon 0
        assert model.fitted_["avg"][0].shape == (len(target),)

    def test_hierarchical_combo(self, target, regressors):
        midas_pmi = MidasSpec("PMI")
        midas_ip = MidasSpec("IP")
        soft = ComboSpec("s1", [midas_pmi, midas_ip], method="average")
        model = MidasCombo(
            combo_specs=ComboSpec(
                "s2",
                [soft, midas_pmi],
                method="regression",
                window=10,
            ),
        )
        model.fit(target, regressors)

        assert "s1" in model.fitted_
        assert "s2" in model.fitted_
        assert "s2" in model.combo_weights_

    def test_summary_contains_model_info(self, target, regressors):
        midas_pmi = MidasSpec("PMI")
        model = MidasCombo(
            combo_specs=ComboSpec("c", [midas_pmi], method="average"),
        )
        model.fit(target, regressors)
        text = model.summary()
        assert "PMI" in text
        assert "RMSE" in text
