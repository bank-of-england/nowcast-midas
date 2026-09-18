"""Direct tests for the specification dataclasses in :mod:`nowcast_midas.specs`.

These classes were previously only exercised indirectly through the
`~nowcast_midas.midas_combo.MidasCombo` pipeline tests.  Here we check
construction defaults, ``__post_init__`` validation and error messages,
and the combination-tree helpers (`ComboSpec.flatten`,
`ComboSpec.collect_indicators`, `ComboSpec.source_names`).
"""

import pandas as pd
import pytest

from nowcast_midas.specs import (
    ComboSpec,
    MidasSpec,
    MultiMidasSpec,
    OLSSpec,
    VariableSpec,
)

# --------------------------------------------------------------------------- #
#  Indicator specs                                                            #
# --------------------------------------------------------------------------- #


def test_midas_spec_defaults():
    spec = MidasSpec("PMI")
    assert spec.variable == "PMI"
    assert spec.method == "almon"
    assert spec.n_lags == 3
    assert spec.estimator is None
    assert spec.dummy_periods is None


def test_ols_spec_defaults():
    spec = OLSSpec("UNEMP")
    assert spec.variable == "UNEMP"
    assert spec.n_lags == 1
    assert spec.n_ar_lags == 0


def test_variable_spec_frequency_default_is_monthly():
    assert VariableSpec("IP").frequency == "ME"
    assert VariableSpec("GDP", frequency="QE").frequency == "QE"


@pytest.mark.parametrize("cls", [MidasSpec, MultiMidasSpec, OLSSpec])
def test_dummy_periods_coerced_to_quarter_end(cls):
    """A mid-month quarter-end date is snapped to the quarter-end timestamp."""
    kwargs = {"dummy_periods": ["2020-06-15"]}
    spec = cls("v", **kwargs) if cls is not MultiMidasSpec else cls(name="v", **kwargs)
    assert spec.dummy_periods == [pd.Timestamp("2020-06-30")]


@pytest.mark.parametrize("cls", [MidasSpec, MultiMidasSpec, OLSSpec])
def test_dummy_periods_non_quarter_end_month_raises(cls):
    kwargs = {"dummy_periods": ["2020-05-31"]}
    with pytest.raises(ValueError, match="quarter-end"):
        cls("v", **kwargs) if cls is not MultiMidasSpec else cls(name="v", **kwargs)


@pytest.mark.parametrize("cls", [MidasSpec, MultiMidasSpec, OLSSpec])
def test_dummy_periods_unparseable_raises(cls):
    kwargs = {"dummy_periods": ["not-a-date"]}
    with pytest.raises(ValueError, match="could not be parsed"):
        cls("v", **kwargs) if cls is not MultiMidasSpec else cls(name="v", **kwargs)


# --------------------------------------------------------------------------- #
#  ComboSpec validation                                                       #
# --------------------------------------------------------------------------- #


def test_combo_spec_defaults():
    spec = ComboSpec("c")
    assert spec.method == "average"
    assert spec.window is None
    assert spec.minimum_sample_size == 10
    assert spec.estimator == "constrained_ls"


def test_combo_spec_minimum_sample_size_type_and_range():
    with pytest.raises(TypeError, match="must be an int"):
        ComboSpec("c", minimum_sample_size="10")
    with pytest.raises(ValueError, match=">= 1"):
        ComboSpec("c", minimum_sample_size=0)


def test_combo_spec_estimator_only_for_regression():
    with pytest.raises(ValueError, match="only applicable to method='regression'"):
        ComboSpec("c", method="average", estimator="clipped_ols")


def test_combo_spec_unknown_estimator_raises():
    with pytest.raises(ValueError, match="constrained_ls.*clipped_ols"):
        ComboSpec("c", method="regression", estimator="ridge")


def test_combo_spec_dummy_periods_not_allowed_for_average():
    with pytest.raises(ValueError, match="not applicable to method='average'"):
        ComboSpec("c", method="average", dummy_periods=["2020-06-30"])


def test_combo_spec_dummy_periods_coerced_for_non_average():
    spec = ComboSpec("c", method="mse", dummy_periods=["2020-06-15"])
    assert spec.dummy_periods == [pd.Timestamp("2020-06-30")]


# --------------------------------------------------------------------------- #
#  Combination-tree helpers                                                   #
# --------------------------------------------------------------------------- #


def _combo_of_combo() -> ComboSpec:
    leaf_a = ComboSpec(
        "leaf_a",
        sources=[MidasSpec("m1"), MidasSpec("m2")],
        method="mse",
        window=8,
    )
    leaf_b = ComboSpec(
        "leaf_b",
        sources=[MidasSpec("m3"), OLSSpec("q1")],
        method="rmse",
    )
    return ComboSpec("root", sources=[leaf_a, leaf_b], method="regression")


def test_source_names_resolves_specs_to_identifiers():
    root = _combo_of_combo()
    assert root.source_names == ["leaf_a", "leaf_b"]
    assert root.sources[0].source_names == ["m1", "m2"]


def test_flatten_returns_leaves_before_root():
    root = _combo_of_combo()
    assert [c.name for c in root.flatten()] == ["leaf_a", "leaf_b", "root"]


def test_flatten_deduplicates_shared_child():
    shared = ComboSpec("shared", sources=[MidasSpec("m1")], method="mse")
    root = ComboSpec(
        "root",
        sources=[
            ComboSpec("mid1", sources=[shared], method="mse"),
            ComboSpec("mid2", sources=[shared], method="mse"),
        ],
        method="regression",
    )
    names = [c.name for c in root.flatten()]
    assert names.count("shared") == 1
    assert names[-1] == "root"


def test_collect_indicators_walks_whole_tree():
    root = _combo_of_combo()
    midas, ols, multi = root.collect_indicators()
    assert sorted(m.variable for m in midas) == ["m1", "m2", "m3"]
    assert [o.variable for o in ols] == ["q1"]
    assert multi == []


def test_collect_indicators_deduplicates_by_name():
    dup = ComboSpec(
        "root",
        sources=[MidasSpec("m1", n_lags=3), MidasSpec("m1", n_lags=6)],
        method="mse",
    )
    midas, _, _ = dup.collect_indicators()
    assert len(midas) == 1
    assert midas[0].n_lags == 3  # first occurrence wins


def test_collect_indicators_propagates_minimum_sample_size_to_children():
    root = ComboSpec(
        "root",
        sources=[MidasSpec("m1"), OLSSpec("q1")],
        method="mse",
        minimum_sample_size=15,
    )
    root.collect_indicators()
    assert all(s.minimum_sample_size == 15 for s in root.sources)
