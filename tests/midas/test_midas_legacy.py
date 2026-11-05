"""Tests for the MIDAS class."""

from typing import ClassVar

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend for testing
import pytest

from nowcast_midas.midas import MIDAS, FittedMidas
from nowcast_midas.temporal_weights import almon, beta, exp_almon
from nowcast_midas.utils import _build_lag_matrix, sample_data

RNG = np.random.default_rng(42)
T, K = 1_000, 6
NOISE = 0.01
TRUE_W = np.asarray(exp_almon(np.array([-0.5, -0.1]), K))


def _simulate(
    true_weights: np.ndarray, alpha: float = 2.0, beta_: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (y, X) from known weights with little noise."""
    X = RNG.standard_normal((T, K))
    y = alpha + beta_ * (X @ true_weights) + NOISE * RNG.standard_normal(T)
    return y, X


class TestBuildLagMatrixRaggedEdge:
    """Verify that build_lag_matrix applies uniform ragged-edge alignment."""

    def _make_regressors(self, end_date: str):
        dates_m = pd.date_range("2019-01-31", end=end_date, freq="ME")
        vals = np.arange(len(dates_m), dtype=float)
        return pd.DataFrame({"date": dates_m, "value": vals})

    @pytest.mark.parametrize(
        "end_month,expected_mwq",
        [("2024-10-31", 1), ("2024-11-30", 2), ("2024-12-31", 3)],
    )
    def test_anchor_aligns_to_month_within_quarter(self, end_month, expected_mwq):
        """Each quarter's lag0 uses the Nth month (N=month_within_quarter)."""
        regressors = self._make_regressors(end_month)
        targets = pd.date_range("2023-03-31", periods=8, freq="QE")
        X = _build_lag_matrix(targets, regressors, n_lags=6)

        # Consecutive quarters should differ by exactly 3 in lag0 index
        diffs = np.diff(X[:, 0])
        np.testing.assert_array_equal(diffs, 3.0)

    def test_fit_forecast_consistency_ragged(self):
        """Lag row from training target date == lag row from forecast info_date.

        During fit(), target dates are quarter-ends.  During forecast(),
        the target is info_date (= max regressor date, often mid-quarter).
        Both must produce the same lag row.
        """
        regressors = self._make_regressors("2024-11-30")  # month_within_quarter=2
        quarter_end = pd.DatetimeIndex(["2024-12-31"])
        info_date = pd.Timestamp("2024-11-30")

        x_train = _build_lag_matrix(quarter_end, regressors, n_lags=6)
        x_fc = _build_lag_matrix(info_date, regressors, n_lags=6)

        np.testing.assert_array_equal(x_train[0], x_fc[0])

    def test_full_quarter_uses_all_three_months(self):
        """month_within_quarter=3: lag0 is the quarter-end month itself."""
        regressors = self._make_regressors("2024-12-31")
        target = pd.DatetimeIndex(["2024-12-31"])
        X = _build_lag_matrix(target, regressors, n_lags=3)

        # Dec=idx71, Nov=idx70, Oct=idx69 for data starting 2019-01
        dec_idx = len(regressors) - 1
        assert X[0, 0] == dec_idx
        assert X[0, 1] == dec_idx - 1
        assert X[0, 2] == dec_idx - 2

    def test_ragged_skips_unavailable_months(self):
        """month_within_quarter=2: lag0 is month 2 (Nov for Q4), not Dec."""
        regressors = self._make_regressors("2024-11-30")
        target = pd.DatetimeIndex(["2024-12-31"])
        X = _build_lag_matrix(target, regressors, n_lags=3)

        # Nov is last available = index len-1
        nov_idx = len(regressors) - 1
        assert X[0, 0] == nov_idx
        assert X[0, 1] == nov_idx - 1
        assert X[0, 2] == nov_idx - 2


class TestMIDASInit:
    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method must be"):
            MIDAS(method="invalid")

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="n_lags must be"):
            MIDAS(n_lags=0)

    def test_ols_not_allowed_for_nls_methods(self):
        with pytest.raises(ValueError, match="OLS only valid"):
            MIDAS(method="exp_almon", estimator="ols")

    def test_defaults(self):
        m = MIDAS()
        assert m.method == "almon"
        assert m.n_lags == 6
        assert m.estimator == "ols"


class TestMIDASFit:
    def test_shape_mismatch_raises(self):
        y, X = _simulate(TRUE_W)
        with pytest.raises(ValueError, match="cols"):
            MIDAS(n_lags=10)._fit_single_horizon(y, X)

    def test_row_mismatch_raises(self):
        y, X = _simulate(TRUE_W)
        with pytest.raises(ValueError, match="row counts"):
            MIDAS(n_lags=K)._fit_single_horizon(y[:5], X)

    @pytest.mark.parametrize("method", ["almon", "unrestricted"])
    def test_fit_populates_attrs(self, method):
        y, X = _simulate(TRUE_W)
        fit = MIDAS(method=method, n_lags=K)._fit_single_horizon(y, X)
        assert isinstance(fit, FittedMidas)
        assert fit.nobs == T
        assert fit.residuals.shape == (T,)
        assert fit.fitted_values.shape == (T,)

    def test_plot_fit(self):
        """Plot uses the fits_ dict directly."""
        y, X = _simulate(TRUE_W)
        m = MIDAS(method="almon", n_lags=K)
        m.fits_[0] = m._fit_single_horizon(y, X)
        fig, ax = m.plot_fit(horizon=0)
        assert fig is not None
        assert ax is not None
        # Actual and fitted series are both drawn.
        assert len(ax.lines) >= 2
        assert ax.get_legend() is not None


class TestMIDASWeightRecovery:
    """Check that each method recovers known weights from simulated data."""

    ATOL = 0.02  # tolerance for weight recovery

    def test_unrestricted(self):
        true_w = np.array([0.4, 0.25, 0.15, 0.10, 0.07, 0.03])
        y, X = _simulate(true_w)
        fit = MIDAS(method="unrestricted", n_lags=K)._fit_single_horizon(y, X)
        np.testing.assert_allclose(fit.weights, true_w, atol=self.ATOL)

    def test_exp_almon(self):
        theta = np.array([-0.5, -0.1])
        true_w = np.asarray(exp_almon(theta, K))
        y, X = _simulate(true_w)
        fit = MIDAS(method="exp_almon", n_lags=K)._fit_single_horizon(y, X)
        np.testing.assert_allclose(fit.weights, true_w, atol=self.ATOL)

    def test_almon(self):
        theta = np.array([1.0, -0.2])
        true_w = np.asarray(almon(theta, K))
        y, X = _simulate(true_w)
        fit = MIDAS(method="almon", n_lags=K)._fit_single_horizon(y, X)
        np.testing.assert_allclose(fit.weights, true_w, atol=self.ATOL)

    def test_beta(self):
        theta = np.array([2.0, 5.0])
        true_w = np.asarray(beta(theta, K))
        y, X = _simulate(true_w)
        fit = MIDAS(method="beta", n_lags=K)._fit_single_horizon(y, X)
        np.testing.assert_allclose(fit.weights, true_w, atol=self.ATOL)

    def test_plot_weights(self):
        y, X = _simulate(TRUE_W)
        m = MIDAS(method="almon", n_lags=K)
        m.fits_[0] = m._fit_single_horizon(y, X)
        fig, ax = m.plot_weights(horizon=0)
        assert fig is not None
        assert ax is not None
        # One drawn artist per lag weight (line or bar).
        assert len(ax.lines) + len(ax.patches) >= 1
        assert ax.get_xlabel() or ax.get_ylabel() or ax.get_title()


class TestMIDASForecast:
    def test_forecast_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Not fitted"):
            MIDAS(n_lags=K).forecast(
                pd.DataFrame(
                    {
                        "date": pd.date_range("2020-01-31", periods=12, freq="ME"),
                        "value": np.ones(12),
                    }
                )
            )

    def test_plot_forecast(self):
        """plot_forecast requires forecast() to have been called."""
        y, X = _simulate(TRUE_W)
        y, X = y[:40], X[:40]
        m = MIDAS(method="almon", n_lags=K)
        fit = m._fit_single_horizon(y, X)
        dates = pd.date_range("2018-03-31", periods=fit.nobs, freq="QE").values
        fit.dates = dates
        fit.fitted_values = pd.Series(fit.fitted_values, index=pd.DatetimeIndex(dates))
        m.fits_[0] = fit
        # Manually set a forecasts_df_ so plot_forecast works
        m.forecasts_df_ = pd.DataFrame(
            {
                "horizon": [0],
                "date": [pd.Timestamp(dates[-1]) + pd.offsets.QuarterEnd(1)],
                "value": [1.0],
            }
        )
        fig, ax = m.plot_forecast(horizon=0)
        assert fig is not None
        assert ax is not None
        assert len(ax.lines) >= 1
        assert ax.get_legend() is not None

    def test_forecast_accuracy_h0(self):
        """Noiseless h=0: out-of-sample forecast ≈ held-out actual.

        DGP: y[t] = alpha + beta * X[t] @ w.  Hold out last obs.
        Pass regressors up to the held-out date so h=0 can nowcast it.
        """
        target, regressors = sample_data(
            n_obs=200,
            n_lags=K,
            noise=0.0,
            seed=7,
            horizon=0,
            method="exp_almon",
            theta_true=[-0.5, -0.1],
        )
        target_train = target.iloc[:-1]
        actual = target.iloc[-1]["value"]
        held_out_date = target.iloc[-1]["date"]

        m = MIDAS(method="exp_almon", n_lags=K, horizons=[0]).fit(
            target_train, regressors
        )
        # Pass regressors up to the held-out target date so the
        # information set covers it (h=0 needs contemporaneous X).
        regressors_fc = regressors[regressors["date"] <= held_out_date]
        fc = m.forecast(regressors_fc)
        predicted = fc.loc[fc["horizon"] == 0, "value"].iloc[0]
        np.testing.assert_allclose(predicted, actual, atol=0.05)

    def test_forecast_accuracy_h4(self):
        """Direct h=4 forecasting must use X_T to forecast y_{T+4}.

        Convention: ``y_{t+h} = f(X_t)``, i.e. to forecast ``y_{T+H}`` the
        model uses the regressor at the latest available date ``T`` (here
        ``X_T``), and the forecast is *dated* ``T + H`` quarters.

        With a noiseless DGP ``y[t+4] = alpha + beta * X[t] @ w`` and the
        latest available X at quarter ``T``, this test pins down that:

        * the forecast equals ``f(X_T)`` reconstructed from the anchor lag
          row and the fitted weights,
        * that value equals the held-out actual ``y_{T+4}``, and
        * the forecast is dated exactly at the held-out target quarter
          ``T + 4`` (not ``T + 5``).
        """
        H = 4
        target, regressors = sample_data(
            n_obs=200,
            n_lags=K,
            noise=0.0,
            seed=7,
            horizon=H,
            method="exp_almon",
            theta_true=[-0.5, -0.1],
        )
        target_train = target.iloc[:-H]
        actual = target.iloc[-1]["value"]
        actual_date = target.iloc[-1]["date"]
        last_train_date = target_train.iloc[-1]["date"]

        m = MIDAS(method="exp_almon", n_lags=K, horizons=[H]).fit(
            target_train, regressors
        )
        # Pass regressors only up to the last training date, so the latest
        # available X is X_T at ``last_train_date`` (= origin T).
        regressors_fc = regressors[regressors["date"] <= last_train_date]
        fc = m.forecast(regressors_fc)
        row = fc.loc[fc["horizon"] == H].iloc[0]
        predicted = row["value"]

        # The model must anchor on the latest available X (X_T), not one
        # quarter further forward.
        x_anchor = _build_lag_matrix(last_train_date, regressors_fc, n_lags=K)
        fit = m.fits_[H]
        expected = float(fit.alpha) + float(x_anchor[0] @ (fit.beta * fit.weights))

        # Forecast == f(X_T) and == held-out actual y_{T+H}.
        np.testing.assert_allclose(predicted, expected, atol=1e-8)
        np.testing.assert_allclose(predicted, actual, atol=0.05)

        # The forecast must be dated at the held-out target quarter T + H,
        # confirming X_T is used to forecast y_{T+H} (no one-quarter drift).
        assert pd.Timestamp(row["date"]) == pd.Timestamp(actual_date)

    def test_forecast_accuracy_h0_ragged(self):
        """Ragged-edge h=0: monthly data ends mid-quarter (2 months).

        DGP is built from the ragged lag matrix so the noiseless
        forecast should still match the held-out actual exactly.
        """
        from nowcast_midas.temporal_weights import get_weights

        n_obs, alpha, beta_ = 200, 2.0, 1.0
        theta = [-0.5, -0.1]
        true_w = np.asarray(get_weights("almon", np.array(theta), K))
        rng = np.random.default_rng(7)

        dates_q = pd.date_range("2000-03-31", periods=n_obs, freq="QE")
        # End monthly data 1 month before last quarter-end → ragged
        dates_m = pd.date_range(
            dates_q[0] - pd.DateOffset(months=K),
            end=dates_q[-1] - pd.DateOffset(months=1),
            freq="ME",
        )
        monthly_vals = rng.standard_normal(len(dates_m))
        regressors = pd.DataFrame({"date": dates_m, "value": monthly_vals})

        X = _build_lag_matrix(dates_q, regressors, K)
        valid = ~np.any(np.isnan(X), axis=1)
        X = X[valid]
        dates_q_valid = dates_q[valid]
        y = alpha + beta_ * (X @ true_w)

        target = pd.DataFrame({"date": dates_q_valid, "value": y})
        target_train = target.iloc[:-1]
        actual = target.iloc[-1]["value"]
        held_out_date = target.iloc[-1]["date"]

        m = MIDAS(method="almon", n_lags=K, horizons=[0]).fit(target_train, regressors)
        regressors_fc = regressors[regressors["date"] <= held_out_date]
        fc = m.forecast(regressors_fc)
        predicted = fc.loc[fc["horizon"] == 0, "value"].iloc[0]
        np.testing.assert_allclose(predicted, actual, atol=0.05)

    def test_forecast_accuracy_h4_ragged(self):
        """Ragged-edge h=4: monthly data ends mid-quarter (2 months).

        DGP is built from the ragged lag matrix so the noiseless
        forecast should still match the held-out actual exactly.
        """
        from nowcast_midas.temporal_weights import get_weights

        H = 4
        n_obs, alpha, beta_ = 200, 2.0, 1.0
        theta = [-0.5, -0.1]
        true_w = np.asarray(get_weights("almon", np.array(theta), K))
        rng = np.random.default_rng(7)

        dates_q = pd.date_range("2000-03-31", periods=n_obs, freq="QE")
        # End monthly data 1 month before last quarter-end → ragged
        dates_m = pd.date_range(
            dates_q[0] - pd.DateOffset(months=K),
            end=dates_q[-1] - pd.DateOffset(months=1),
            freq="ME",
        )
        monthly_vals = rng.standard_normal(len(dates_m))
        regressors = pd.DataFrame({"date": dates_m, "value": monthly_vals})

        X = _build_lag_matrix(dates_q, regressors, K)
        valid = ~np.any(np.isnan(X), axis=1)
        X = X[valid]
        dates_q_valid = dates_q[valid]
        T = len(y := np.empty(X.shape[0]))
        y[:H] = alpha + rng.standard_normal(H)
        y[H:] = alpha + beta_ * (X[: T - H] @ true_w)

        target = pd.DataFrame({"date": dates_q_valid, "value": y})
        target_train = target.iloc[:-H]
        actual = target.iloc[-1]["value"]
        last_train_date = target_train.iloc[-1]["date"]

        m = MIDAS(method="almon", n_lags=K, horizons=[H]).fit(target_train, regressors)
        # Trim to month 2 of the quarter (matching the ragged pattern)
        # so that month_within_quarter stays consistent with training.
        ragged_cutoff = last_train_date - pd.DateOffset(months=1)
        regressors_fc = regressors[regressors["date"] <= ragged_cutoff]
        fc = m.forecast(regressors_fc)
        predicted = fc.loc[fc["horizon"] == H, "value"].iloc[0]
        np.testing.assert_allclose(predicted, actual, atol=0.05)


class TestMIDASDataFrameFit:
    """Test the public DataFrame-based fit() interface."""

    def _make_dataframes(
        self, true_weights=None, alpha=2.0, beta_=1.0, n_quarters=80, n_lags=K
    ):
        """Create target and regressor DataFrames using sample_data."""
        # Use sample_data which generates synthetic data efficiently
        target, regressors = sample_data(
            n_obs=n_quarters, n_lags=n_lags, alpha=alpha, beta_=beta_, seed=99
        )
        return target, regressors

    def test_fit_from_dataframes(self):
        target, regressors = self._make_dataframes()
        m = MIDAS(method="exp_almon", n_lags=K, horizons=[0, 1, 2]).fit(
            target, regressors
        )
        assert set(m.fits_.keys()) == {0, 1, 2}
        for fit in m.fits_.values():
            assert isinstance(fit, FittedMidas)
            assert fit.nobs > 0
            assert fit.fitted_values.shape[0] == fit.nobs

    def test_forecast_from_dataframes(self):
        target, regressors = self._make_dataframes()
        m = MIDAS(method="almon", n_lags=K, horizons=[0, 1, 2]).fit(target, regressors)
        fc = m.forecast(regressors)
        assert list(fc.columns) == ["date", "horizon", "spec", "value", "forecast"]
        assert (fc["forecast"] == fc["value"]).all()  # deprecated alias
        assert len(fc) == 3  # one row per horizon

    def test_forecast_adds_gamma_and_ar_when_x_missing(self):
        m = MIDAS(
            method="almon",
            n_lags=3,
            horizons=[0],
            n_ar_lags=2,
            dummy_periods=[pd.Timestamp("2020-03-31")],
        )

        # Minimal fitted state required by forecast().
        fit = FittedMidas(
            alpha=1.0,
            beta=1.0,
            theta=np.array([0.0, 0.0]),
            weights=np.array([0.2, 0.3, 0.5]),
            nobs=1,
        )
        fit.gamma = np.array([2.0])
        fit.phi = np.array([0.5, -0.25])
        m.fits_[0] = fit

        # Provide enough target history so AR predictors are available.
        m.target_ = pd.DataFrame(
            {
                "date": pd.to_datetime(["2019-06-30", "2019-09-30", "2019-12-31"]),
                "value": [5.0, 10.0, 20.0],
            }
        )

        # Use regressors with date after training period so x_row contains NaN.
        regressors_fc = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-03-31"]),
                "value": [1.23],
            }
        )
        fc = m.forecast(regressors_fc)

        predicted = float(fc.loc[fc["horizon"] == 0, "value"].iloc[0])
        # x contribution is skipped; forecast keeps alpha + gamma + AR.
        expected = 1.0 + 2.0 + (20.0 * 0.5 + 10.0 * -0.25)
        assert predicted == pytest.approx(expected)

    def test_missing_columns_raises(self):
        df_bad = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df_ok = pd.DataFrame(
            {"date": pd.date_range("2020", periods=2), "value": [1, 2]}
        )
        with pytest.raises(ValueError):
            MIDAS().fit(df_bad, df_ok)
        with pytest.raises(ValueError):
            MIDAS().fit(df_ok, df_bad)

    def test_target_nan_raises(self):
        target = pd.DataFrame(
            {
                "date": pd.date_range("2020-03-31", periods=3, freq="QE"),
                "value": [1.0, np.nan, 3.0],
            }
        )
        regressors = pd.DataFrame(
            {
                "date": pd.date_range("2019-01-31", periods=20, freq="ME"),
                "value": np.ones(20),
            }
        )
        with pytest.raises(ValueError, match="NaN"):
            MIDAS(n_lags=3).fit(target, regressors)


class TestMIDASFitRecovery:
    """Test weight recovery through the public .fit() DataFrame interface."""

    ATOL = 0.05
    THETA: ClassVar[list[float]] = [-0.5, -0.1]

    def test_fit_recovers_weights_h0(self):
        """fit() with horizon=0 recovers known exp_almon weights."""
        target, regressors = sample_data(
            n_obs=500,
            n_lags=K,
            noise=0.01,
            seed=42,
            horizon=0,
            method="exp_almon",
            theta_true=self.THETA,
        )
        m = MIDAS(method="exp_almon", n_lags=K, horizons=[0]).fit(target, regressors)
        true_w = np.asarray(exp_almon(np.array(self.THETA), K))
        np.testing.assert_allclose(m.fits_[0].weights, true_w, atol=self.ATOL)

    def test_fit_recovers_alpha_and_beta_h0(self):
        """fit() recovers intercept and slope at h=0."""
        target, regressors = sample_data(
            n_obs=500,
            n_lags=K,
            alpha=2.0,
            beta_=1.0,
            noise=0.01,
            seed=42,
            horizon=0,
            method="exp_almon",
            theta_true=self.THETA,
        )
        m = MIDAS(method="exp_almon", n_lags=K, horizons=[0]).fit(target, regressors)
        assert abs(m.fits_[0].alpha - 2.0) < 0.1
        assert abs(m.fits_[0].beta - 1.0) < 0.1

    def test_fit_recovers_weights_h1(self):
        """fit() with horizons=[0,1] recovers weights at h=1 when DGP has lead=1."""
        target, regressors = sample_data(
            n_obs=500,
            n_lags=K,
            noise=0.01,
            seed=42,
            horizon=1,
            method="exp_almon",
            theta_true=self.THETA,
        )
        m = MIDAS(method="exp_almon", n_lags=K, horizons=[0, 1]).fit(target, regressors)
        true_w = np.asarray(exp_almon(np.array(self.THETA), K))
        # h=1 should recover the known weights
        np.testing.assert_allclose(m.fits_[1].weights, true_w, atol=self.ATOL)

    def test_fit_h0_does_not_match_h1_dgp(self):
        """When DGP has lead=1, h=0 fit should NOT recover the true weights."""
        target, regressors = sample_data(
            n_obs=500,
            n_lags=K,
            noise=0.01,
            seed=42,
            horizon=1,
            method="exp_almon",
            theta_true=self.THETA,
        )
        m = MIDAS(method="exp_almon", n_lags=K, horizons=[0, 1]).fit(target, regressors)
        true_w = np.asarray(exp_almon(np.array(self.THETA), K))
        # h=0 should NOT recover the weights (wrong horizon)
        assert np.max(np.abs(m.fits_[0].weights - true_w)) > 0.1

    @pytest.mark.parametrize(
        "method, theta",
        [
            ("exp_almon", [-0.5, -0.1]),
            ("almon", [1.0, -0.2]),
            ("beta", [2.0, 5.0]),
        ],
    )
    def test_fit_recovers_weights_various_methods(self, method, theta):
        """fit() recovers weights for different weighting schemes."""
        estimator = "ols" if method == "almon" else "nls"
        target, regressors = sample_data(
            n_obs=500,
            n_lags=K,
            noise=0.01,
            seed=42,
            method=method,
            theta_true=theta,
        )
        m = MIDAS(method=method, n_lags=K, horizons=[0], estimator=estimator).fit(
            target, regressors
        )
        from nowcast_midas.temporal_weights import get_weights

        true_w = np.asarray(get_weights(method, np.array(theta), K))
        np.testing.assert_allclose(m.fits_[0].weights, true_w, atol=self.ATOL)


class TestMIDASDummy:
    """Test estimation with dummy variables absorbing outliers."""

    # Covid-era dummy quarters: 2020Q2 through 2021Q3
    DUMMY_PERIODS = pd.date_range("2020-06-30", "2021-09-30", freq="QE")
    OUTLIER_MAGNITUDE = 100.0

    def _make_data_with_outliers(
        self, method="exp_almon", n_obs=200, seed=99, theta_true=None
    ):
        """Generate clean DGP then inject large outliers at dummy dates."""
        if theta_true is None:
            theta_true = [-0.5, -0.1]
        target, regressors = sample_data(
            n_obs=n_obs,
            n_lags=K,
            noise=0.01,
            seed=seed,
            horizon=0,
            method=method,
            theta_true=theta_true,
        )
        # Inject outliers at the dummy periods
        mask = target["date"].isin(self.DUMMY_PERIODS)
        target.loc[mask, "value"] += self.OUTLIER_MAGNITUDE
        return target, regressors

    # True DGP parameters used by sample_data defaults
    TRUE_ALPHA = 2.0
    TRUE_BETA = 1.0
    ATOL = 0.05  # tight tolerance (noise=0.01)

    def test_dummy_recovers_dgp_ols(self):
        """OLS with dummies recovers alpha, weights, and gamma almost exactly."""
        from nowcast_midas.temporal_weights import get_weights

        theta = [1.0, -0.2]
        target, regressors = self._make_data_with_outliers(
            method="almon", theta_true=theta
        )
        true_w = np.asarray(get_weights("almon", np.array(theta), K))

        m = MIDAS(
            method="almon",
            n_lags=K,
            horizons=[0],
            dummy_periods=list(self.DUMMY_PERIODS),
        ).fit(target, regressors)
        fit = m.fits_[0]

        # Recover intercept
        assert abs(fit.alpha - self.TRUE_ALPHA) < self.ATOL, (
            f"alpha: expected {self.TRUE_ALPHA}, got {fit.alpha}"
        )
        # Recover weights
        np.testing.assert_allclose(
            fit.weights, true_w, atol=self.ATOL, err_msg="OLS weights not recovered"
        )
        # Recover dummy coefficients (each ≈ OUTLIER_MAGNITUDE)
        assert len(fit.gamma) == len(self.DUMMY_PERIODS)
        np.testing.assert_allclose(
            fit.gamma,
            self.OUTLIER_MAGNITUDE,
            atol=self.ATOL,
            err_msg="OLS gamma not recovered",
        )

    def test_dummy_recovers_dgp_nls(self):
        """NLS with dummies recovers alpha, beta, weights, and gamma."""
        from nowcast_midas.temporal_weights import get_weights

        theta = [-0.5, -0.1]
        target, regressors = self._make_data_with_outliers(
            method="exp_almon", theta_true=theta
        )

        true_w = np.asarray(get_weights("exp_almon", np.array(theta), K))

        m = MIDAS(
            method="exp_almon",
            n_lags=K,
            horizons=[0],
            dummy_periods=list(self.DUMMY_PERIODS),
        ).fit(target, regressors)
        fit = m.fits_[0]

        # Recover intercept
        assert abs(fit.alpha - self.TRUE_ALPHA) < self.ATOL, (
            f"alpha: expected {self.TRUE_ALPHA}, got {fit.alpha}"
        )
        # Recover slope
        assert abs(fit.beta - self.TRUE_BETA) < self.ATOL, (
            f"beta: expected {self.TRUE_BETA}, got {fit.beta}"
        )
        # Recover weights
        np.testing.assert_allclose(
            fit.weights, true_w, atol=self.ATOL, err_msg="NLS weights not recovered"
        )
        # Recover dummy coefficients
        assert len(fit.gamma) == len(self.DUMMY_PERIODS)
        np.testing.assert_allclose(
            fit.gamma,
            self.OUTLIER_MAGNITUDE,
            atol=self.ATOL,
            err_msg="NLS gamma not recovered",
        )

    def test_without_dummies_weights_distorted(self):
        """Without dummies, outliers distort estimated weights."""

        theta = [1.0, -0.2]
        target, regressors = self._make_data_with_outliers(
            method="almon", theta_true=theta
        )

        m = MIDAS(method="almon", n_lags=K, horizons=[0]).fit(target, regressors)
        # Without dummies, alpha is pulled away from the true value
        assert abs(m.fits_[0].alpha - self.TRUE_ALPHA) > self.ATOL, (
            "Outliers should distort intercept when no dummies used"
        )

    def test_forecast_excludes_dummy_outside_period(self):
        """Forecast for a non-dummy quarter should not include dummy effect."""
        target, regressors = self._make_data_with_outliers(
            method="almon", theta_true=[1.0, -0.2]
        )

        m = MIDAS(
            method="almon",
            n_lags=K,
            horizons=[0],
            dummy_periods=list(self.DUMMY_PERIODS),
        ).fit(target, regressors)

        fc = m.forecast(regressors)
        predicted = fc.loc[fc["horizon"] == 0, "value"].iloc[0]
        # Forecast date is outside dummy period; should be close to true alpha
        assert abs(predicted - self.TRUE_ALPHA) < 2.0, (
            f"Forecast looks contaminated by dummy: {predicted}"
        )


# ============================================================
# AR(p) lags of the dependent variable
# ============================================================
class TestMIDASARLags:
    """Exact-recovery and forecast tests for the n_ar_lags feature."""

    @pytest.mark.parametrize("horizon", [0, 3])
    def test_ols_almon_exact_recovery(self, horizon):
        """OLS-MIDAS (almon) with zero noise must recover alpha, weights, phi."""
        n_lags = 4
        theta_true = np.array([1.0, -0.2])
        phi_true = np.array([0.3, -0.1])
        alpha_true = 2.0
        beta_true = 1.0

        target, regressors = sample_data(
            n_obs=120,
            n_lags=n_lags,
            alpha=alpha_true,
            beta_=beta_true,
            noise=0.0,
            seed=0,
            horizon=horizon,
            method="almon",
            theta_true=theta_true,
            n_ar_lags=len(phi_true),
            phi_true=phi_true,
        )

        model = MIDAS(
            method="almon",
            n_lags=n_lags,
            n_pars_weights=len(theta_true),
            estimator="ols",
            horizons=[horizon],
            n_ar_lags=len(phi_true),
        )
        model.fit(target, regressors)
        fit = model.fits_[horizon]

        true_weights = almon(np.array(theta_true), n_lags)
        true_weights = np.asarray(true_weights)

        assert fit.alpha == pytest.approx(alpha_true, abs=1e-8)
        np.testing.assert_allclose(
            fit.beta * fit.weights, beta_true * true_weights, atol=1e-8
        )
        np.testing.assert_allclose(fit.phi, phi_true, atol=1e-8)

    @pytest.mark.parametrize("horizon", [0, 3])
    def test_forecast_matches_dgp(self, horizon):
        """The OOS forecast must reproduce the actual full-DGP value."""
        n_lags = 4
        target_full, regressors = sample_data(
            n_obs=80,
            n_lags=n_lags,
            noise=0.0,
            seed=1,
            horizon=horizon,
            method="almon",
            theta_true=np.array([1.0, -0.2]),
            n_ar_lags=2,
            phi_true=np.array([0.3, -0.1]),
        )

        # Train on all but the last (horizon + 1) target points.
        train = target_full.iloc[: len(target_full) - (horizon + 1)].copy()
        model = MIDAS(
            method="almon",
            n_lags=n_lags,
            n_pars_weights=2,
            estimator="ols",
            horizons=[horizon],
            n_ar_lags=2,
        )
        model.fit(train, regressors)

        # Use regressors max date at the first quarter past training so that
        # forecast date lands on the held-out actual.
        last_train_q = pd.Timestamp(train["date"].iloc[-1])
        info_date = last_train_q + pd.offsets.QuarterEnd(1)

        # Filter regressors to have max date = info_date
        regressors_fc = regressors.loc[regressors["date"] <= info_date].copy()
        fc = model.forecast(regressors_fc)
        predicted = float(fc.loc[fc["horizon"] == horizon, "value"].iloc[0])

        actual_date = info_date + pd.DateOffset(months=3 * horizon)
        actual = float(
            target_full.loc[target_full["date"] == actual_date, "value"].iloc[0]
        )
        assert predicted == pytest.approx(actual, abs=1e-6)


class TestMIDASForecastDecomp:
    """Forecast decomposition: contributions must sum to the forecast value."""

    def test_columns_and_sum_to_forecast(self):
        """Plain almon MIDAS: per-horizon contributions sum to the forecast."""
        target, regressors = sample_data(n_obs=80, n_lags=K, seed=99)
        m = MIDAS(method="almon", n_lags=K, horizons=[0, 1, 2]).fit(target, regressors)

        fc = m.forecast(regressors)
        decomp = m.forecast_decomp(regressors, regressor_name="x_m")

        assert list(decomp.columns) == [
            "horizon",
            "date",
            "component",
            "contribution",
            "weight",
        ]
        for h in [0, 1, 2]:
            s = decomp.loc[decomp["horizon"] == h, "contribution"].sum()
            f = fc.loc[fc["horizon"] == h, "value"].iloc[0]
            np.testing.assert_allclose(s, f, atol=1e-9)

        # intercept carries weight 1.0; the MIDAS block weight is NaN.
        intercept = decomp.loc[decomp["component"] == "intercept"]
        assert (intercept["weight"] == 1.0).all()
        block = decomp.loc[decomp["component"] == "x_m"]
        assert block["weight"].isna().all()

    def test_sum_to_forecast_with_ar_lags(self):
        """AR lags appear as components and the sum still matches forecast()."""
        target, regressors = sample_data(
            n_obs=120,
            n_lags=4,
            noise=0.0,
            seed=1,
            method="almon",
            theta_true=np.array([1.0, -0.2]),
            n_ar_lags=2,
            phi_true=np.array([0.3, -0.1]),
        )
        m = MIDAS(method="almon", n_lags=4, horizons=[0, 1], n_ar_lags=2).fit(
            target, regressors
        )

        fc = m.forecast(regressors)
        decomp = m.forecast_decomp(regressors, regressor_name="x_m")

        components = set(decomp["component"])
        assert {"ar_lag1", "ar_lag2"}.issubset(components)
        for h in [0, 1]:
            s = decomp.loc[decomp["horizon"] == h, "contribution"].sum()
            f = fc.loc[fc["horizon"] == h, "value"].iloc[0]
            np.testing.assert_allclose(s, f, atol=1e-8)

    def test_dummy_and_ar_components_when_x_missing(self):
        """Manual fitted state with an active dummy + AR lags; sum holds."""
        forecast_date = pd.Timestamp("2020-03-31")
        m = MIDAS(
            method="almon",
            n_lags=3,
            horizons=[0],
            n_ar_lags=2,
            dummy_periods=[forecast_date],
        )

        fit = FittedMidas(
            alpha=1.0,
            beta=1.0,
            theta=np.array([0.0, 0.0]),
            weights=np.array([0.2, 0.3, 0.5]),
            nobs=1,
        )
        fit.gamma = np.array([2.0])
        fit.phi = np.array([0.5, -0.25])
        m.fits_[0] = fit
        m.target_ = pd.DataFrame(
            {
                "date": pd.to_datetime(["2019-06-30", "2019-09-30", "2019-12-31"]),
                "value": [5.0, 10.0, 20.0],
            }
        )

        # X has too few monthly lags so the lag row is NaN (block skipped).
        regressors_fc = pd.DataFrame(
            {"date": pd.to_datetime(["2020-03-31"]), "value": [1.23]}
        )

        fc = m.forecast(regressors_fc)
        decomp = m.forecast_decomp(regressors_fc)

        components = set(decomp["component"])
        assert "x_m" not in components and "X" not in components
        assert {"intercept", "dummy_0", "ar_lag1", "ar_lag2"}.issubset(components)

        s = decomp.loc[decomp["horizon"] == 0, "contribution"].sum()
        f = float(fc.loc[fc["horizon"] == 0, "value"].iloc[0])
        np.testing.assert_allclose(s, f, atol=1e-12)
