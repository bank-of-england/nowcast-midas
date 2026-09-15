from nowcast_midas import __all__ as root_all
from nowcast_midas.combo_weights import __all__ as combo_weights_all
from nowcast_midas.midas import __all__ as midas_all
from nowcast_midas.midas_combo import __all__ as midas_combo_all
from nowcast_midas.multi_midas import __all__ as multi_midas_all
from nowcast_midas.ols import __all__ as ols_all
from nowcast_midas.plots.midas import __all__ as midas_plots_all
from nowcast_midas.plots.midas_combo import __all__ as combo_plots_all
from nowcast_midas.plots.style import __all__ as style_all
from nowcast_midas.specs import __all__ as specs_all
from nowcast_midas.temporal_weights import __all__ as temporal_weights_all
from nowcast_midas.utils import __all__ as utils_all


def test_root_exports_only_user_facing_types():
    assert set(root_all) == {
        "MIDAS",
        "OLS",
        "MidasCombo",
        "MultiMIDAS",
        "FittedMidas",
        "FittedOLS",
        "FittedMultiMidas",
        "VariableFit",
        "VariableSpec",
        "MidasSpec",
        "MultiMidasSpec",
        "OLSSpec",
        "ComboSpec",
    }


def test_supported_submodule_exports_are_explicit():
    assert set(midas_all) == {"MIDAS", "FittedMidas"}
    assert set(midas_combo_all) == {"MidasCombo"}
    assert set(multi_midas_all) == {
        "MultiMIDAS",
        "VariableFit",
        "FittedMultiMidas",
    }
    assert set(ols_all) == {"OLS", "FittedOLS"}
    assert set(specs_all) == {
        "VariableSpec",
        "MidasSpec",
        "MultiMidasSpec",
        "OLSSpec",
        "ComboSpec",
    }
    assert set(temporal_weights_all) == {
        "exp_almon",
        "beta",
        "almon",
        "unrestricted",
        "get_weights",
    }
    assert set(combo_weights_all) == {
        "fit_average",
        "fit_error_based_weights",
        "fit_regression_weights",
        "clipped_ols",
        "constrained_least_squares",
    }
    assert set(utils_all) == {"sample_data", "sample_combo_data"}


def test_plotting_modules_have_no_wildcard_exports():
    assert midas_plots_all == []
    assert combo_plots_all == []
    assert style_all == []
