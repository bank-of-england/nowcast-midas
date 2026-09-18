"""Specification and result dataclasses for the MIDAS pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

__all__ = [
    "ComboSpec",
    "MidasSpec",
    "MultiMidasSpec",
    "OLSSpec",
    "VariableSpec",
]


@dataclass
class VariableSpec:
    """Per-variable specification for a `MultiMIDAS` model.

    Parameters
    ----------
    variable : str
        Name of the regressor variable (must match a ``variable`` value
        in the regressors DataFrame).
    method : str
        MIDAS weighting scheme: ``'exp_almon'``, ``'beta'``, ``'almon'``,
        or ``'unrestricted'`` (default ``'almon'``).  Ignored when
        ``frequency='QE'`` (quarterly regressors always enter linearly).
    n_lags : int
        Number of lags to include (default 3).  Monthly lags for
        ``frequency='ME'``, quarterly lags for ``frequency='QE'``.
    n_pars_weights : int
        Weight-shape parameters for polynomial schemes (default 2).
        Ignored for quarterly regressors.
    estimator : str or None
        ``'ols'`` or ``'nls'``.  ``None`` chooses automatically based on
        *method* (``'ols'`` for ``almon``/``unrestricted``, ``'nls'``
        otherwise).  Ignored for quarterly regressors.
    start_lag : int
        Index of the first lag to include (default 0).
    frequency : str
        Sampling frequency of the regressor: ``'ME'`` for monthly
        (default) or ``'QE'`` for quarterly.  Quarterly regressors
        bypass MIDAS weighting and enter the model linearly.
    Attributes
    ----------
    variable : str
        Name of the regressor variable.
    method : str
        MIDAS weighting scheme.
    n_lags : int
        Number of lags to include.
    n_pars_weights : int
        Number of weight-shape parameters.
    estimator : str | None
        Estimator override, or ``None`` for automatic selection.
    start_lag : int
        Index of the first lag to include.
    frequency : str
        Sampling frequency of the regressor.
    """

    variable: str
    method: str = "almon"
    n_lags: int = 3
    n_pars_weights: int = 2
    estimator: str | None = None
    start_lag: int = 0
    frequency: str = "ME"


@dataclass
class MidasSpec:
    """Specification for a single MIDAS indicator model.

    Parameters
    ----------
    variable : str
        Name of the monthly regressor variable (must match a ``variable``
        value in the regressors DataFrame).
    method : str
        MIDAS weighting scheme: ``'exp_almon'``, ``'beta'``, ``'almon'``,
        or ``'unrestricted'`` (default ``'almon'``).
    n_lags : int
        Number of monthly lags to include (default 3).
    n_pars_weights : int
        Weight-shape parameters for polynomial schemes (default 2).
    estimator : str or None
        ``'ols'`` or ``'nls'``.  Defaults based on *method*.
    start_lag : int
        Starting lag for the MIDAS regression (default 0).
    n_ar_lags : int
        Number of own (target) lags to include as AR regressors
        (default 0, i.e. no AR augmentation).
    dummy_periods : list of int or None
        If not None, list of dates of quarters to include as dummy variables.
    minimum_sample_size : int or None
        Minimum number of fitted quarterly observations required before this
        model is considered valid for forecasting. Default is None (no requirement).
    Attributes
    ----------
    variable : str
        Name of the monthly regressor variable.
    method : str
        MIDAS weighting scheme.
    n_lags : int
        Number of monthly lags to include.
    n_pars_weights : int
        Number of weight-shape parameters.
    estimator : str | None
        Estimator override, or ``None`` for automatic selection.
    start_lag : int
        Index of the first lag to include.
    n_ar_lags : int
        Number of autoregressive lags.
    dummy_periods : list[pd.Timestamp] | None
        Quarter-end dates to include as dummy variables.
    minimum_sample_size : int | None
        Minimum number of fitted observations required for forecasting.
    """

    variable: str
    method: str = "almon"
    n_lags: int = 3
    n_pars_weights: int = 2
    estimator: str | None = None
    start_lag: int = 0
    n_ar_lags: int = 0
    dummy_periods: list[pd.Timestamp] | None = None
    minimum_sample_size: int | None = None

    def __post_init__(self) -> None:
        if self.dummy_periods is not None:
            coerced = []
            for p in self.dummy_periods:
                try:
                    ts = pd.Timestamp(p) + pd.offsets.MonthEnd(0)
                except (TypeError, ValueError, OverflowError):
                    raise ValueError(
                        f"dummy_periods entry {p!r} could not be parsed as a date."
                    )
                if ts.month not in (3, 6, 9, 12):
                    raise ValueError(
                        f"dummy_periods entry {p!r} does not fall in a quarter-end "
                        f"month (March, June, September, December)."
                    )
                coerced.append(ts)
            self.dummy_periods = coerced


@dataclass
class MultiMidasSpec:
    """Specification for a multi-regressor MIDAS model inside a
    `MidasCombo` pipeline.

    The fitted values produced by the underlying
    `MultiMIDAS` model are exposed to
    `ComboSpec` nodes under `name`.

    Parameters
    ----------
    name : str
        Unique identifier for this model within the pipeline (used as
        the source name in `ComboSpec`).
    variables : list of str or VariableSpec
        Regressors to include.  Plain strings use the shared defaults;
        `VariableSpec` instances override per-variable settings.
    method : str
        Shared weighting scheme for variables given as plain strings
        (default ``'almon'``).
    n_lags : int
        Shared number of monthly lags (default 3).
    n_pars_weights : int
        Shared weight-shape parameters (default 2).
    estimator : str or None
        Shared estimator override (default ``None`` = auto).
    start_lag : int
        Shared starting lag index (default 0).
    n_ar_lags : int
        Number of AR lags of the target (default 0).
    dummy_periods : list of pd.Timestamp or None
        Outlier-dummy quarters (default ``None``).
    minimum_sample_size : int or None
        Minimum number of fitted quarterly observations required before this
        model is considered valid for forecasting. Default is None (no requirement).
    Attributes
    ----------
    name : str
        Unique identifier for this model.
    variables : list[str | VariableSpec]
        Regressors to include.
    method : str
        Shared MIDAS weighting scheme.
    n_lags : int
        Shared number of lags.
    n_pars_weights : int
        Shared number of weight-shape parameters.
    estimator : str | None
        Shared estimator override.
    start_lag : int
        Shared starting lag index.
    n_ar_lags : int
        Number of autoregressive lags.
    dummy_periods : list[pd.Timestamp] | None
        Outlier-dummy quarters.
    minimum_sample_size : int | None
        Minimum number of fitted observations required for forecasting.
    """

    name: str
    variables: list[str | VariableSpec] = field(default_factory=list)
    method: str = "almon"
    n_lags: int = 3
    n_pars_weights: int = 2
    estimator: str | None = None
    start_lag: int = 0
    n_ar_lags: int = 0
    dummy_periods: list[pd.Timestamp] | None = None
    minimum_sample_size: int | None = None

    def __post_init__(self) -> None:
        if self.dummy_periods is not None:
            coerced = []
            for p in self.dummy_periods:
                try:
                    ts = pd.Timestamp(p) + pd.offsets.MonthEnd(0)
                except (TypeError, ValueError, OverflowError):
                    raise ValueError(
                        f"dummy_periods entry {p!r} could not be parsed as a date."
                    )
                if ts.month not in (3, 6, 9, 12):
                    raise ValueError(
                        f"dummy_periods entry {p!r} does not fall in a quarter-end "
                        f"month (March, June, September, December)."
                    )
                coerced.append(ts)
            self.dummy_periods = coerced


@dataclass
class OLSSpec:
    """Specification for a quarterly OLS regressor (no MIDAS weighting).

    Used for low-frequency indicators that share the target's frequency
    (quarterly) and therefore need plain OLS rather than mixed-frequency
    weighting.  Fitted values are made available to `ComboSpec`
    nodes under ``variable`` exactly like a `MidasSpec`.

    Parameters
    ----------
    variable : str
        Name of the quarterly regressor variable (must match a
        ``variable`` value in the regressors DataFrame with
        ``frequency='QE'``).
    n_lags : int
        Number of quarterly lags to include (default 1, i.e. only the
        contemporaneous value ``x_t``).
    start_lag : int
        Index of the first quarterly lag to include (default 0).
    n_ar_lags : int
        Number of own (target) lags to include as AR regressors
        (default 0).
    dummy_periods : list of pd.Timestamp or None
        Optional outlier-dummy quarters.
    minimum_sample_size : int or None
        Minimum number of fitted quarterly observations required before this
        model is considered valid for forecasting. Default is None (no requirement).
    Attributes
    ----------
    variable : str
        Name of the quarterly regressor variable.
    n_lags : int
        Number of quarterly lags to include.
    start_lag : int
        Index of the first quarterly lag to include.
    n_ar_lags : int
        Number of autoregressive lags.
    dummy_periods : list[pd.Timestamp] | None
        Optional outlier-dummy quarters.
    minimum_sample_size : int | None
        Minimum number of fitted observations required for forecasting.
    """

    variable: str
    n_lags: int = 1
    start_lag: int = 0
    n_ar_lags: int = 0
    dummy_periods: list[pd.Timestamp] | None = None
    minimum_sample_size: int | None = None

    def __post_init__(self) -> None:
        if self.dummy_periods is not None:
            coerced = []
            for p in self.dummy_periods:
                try:
                    ts = pd.Timestamp(p) + pd.offsets.MonthEnd(0)
                except (TypeError, ValueError, OverflowError):
                    raise ValueError(
                        f"dummy_periods entry {p!r} could not be parsed as a date."
                    )
                if ts.month not in (3, 6, 9, 12):
                    raise ValueError(
                        f"dummy_periods entry {p!r} does not fall in a quarter-end "
                        f"month (March, June, September, December)."
                    )
                coerced.append(ts)
            self.dummy_periods = coerced


@dataclass
class ComboSpec:
    """Specification for a forecast combination node.

    Parameters
    ----------
    name : str
        Unique name for this combination.
    sources : list of str, MidasSpec, OLSSpec, MultiMidasSpec or ComboSpec
        Sources to combine.  Each entry may be:

                * a string — the variable or combination name declared elsewhere in
                    the pipeline,
                * a `MidasSpec`, `OLSSpec`, or `MultiMidasSpec` —
                    an indicator model that the pipeline registers automatically,
                * a nested `ComboSpec` — a combination of combinations.
    method : str
        Combination method: ``'average'``, ``'rmse'``, ``'mse'``,
        ``'mae'``, or ``'regression'`` (default ``'average'``).
    window : int or None
        Rolling window for error / regression estimation.
        Use ``None`` for an expanding window covering the full sample.
        Default is ``None``.
    minimum_sample_size : int
        Minimum number of finite fitted values required for a source to enter
        this combination. Sources below this threshold are removed before
        rows with missing sources are filtered. Default is 10. The same
        argument on an indicator spec controls its model fit requirement.
    discount_rate : float
        Exponential discount rate for error weighting.
        Default is 1.0 (no discounting).
    estimator : str
        Weight estimation method when ``method='regression'``.
        Either ``'constrained_ls'`` (default) or ``'clipped_ols'``.
        Both methods return non-negative weights that sum to one.
    dummy_periods : list of str or pd.Timestamp or None
        Quarters to exclude from weight estimation. Not applicable to
        ``method='average'``. Rows whose date matches an entry in
        ``dummy_periods`` are masked out of the error/regression
        estimation.  Default is None (no exclusions).
    Attributes
    ----------
    name : str
        Unique name for this combination.
    sources : list[str | MidasSpec | OLSSpec | MultiMidasSpec | ComboSpec]
        Sources to combine.
    method : str
        Combination method.
    window : int | None
        Rolling estimation window, or ``None`` for an expanding window.
    minimum_sample_size : int
        Minimum finite sample size for a source.
    discount_rate : float
        Exponential discount rate for error weighting.
    estimator : str
        Regression weight estimator.
    dummy_periods : list[pd.Timestamp] | None
        Quarters excluded from weight estimation.
    """

    name: str
    sources: list[str | MidasSpec | OLSSpec | MultiMidasSpec | ComboSpec] = field(
        default_factory=list
    )
    method: str = "average"
    window: int | None = None
    minimum_sample_size: int = 10
    discount_rate: float = 1.0
    estimator: str = "constrained_ls"
    dummy_periods: list[pd.Timestamp] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.minimum_sample_size, int):
            raise TypeError("minimum_sample_size must be an int.")
        if self.minimum_sample_size < 1:
            raise ValueError("minimum_sample_size must be >= 1.")

        if self.method != "regression" and self.estimator != "constrained_ls":
            raise ValueError("estimator is only applicable to method='regression'.")
        if self.estimator not in ("constrained_ls", "clipped_ols"):
            raise ValueError(
                f"estimator must be 'constrained_ls' or 'clipped_ols', "
                f"got {self.estimator!r}"
            )
        if self.dummy_periods is not None:
            if self.method == "average":
                raise ValueError("dummy_periods is not applicable to method='average'.")
            coerced = []
            for p in self.dummy_periods:
                try:
                    ts = pd.Timestamp(p) + pd.offsets.MonthEnd(0)
                except (TypeError, ValueError, OverflowError):
                    raise ValueError(
                        f"dummy_periods entry {p!r} could not be parsed as a date."
                    )
                if ts.month not in (3, 6, 9, 12):
                    raise ValueError(
                        f"dummy_periods entry {p!r} does not fall in a quarter-end "
                        f"month (March, June, September, December)."
                    )
                coerced.append(ts)
            self.dummy_periods = coerced

    @property
    def source_names(self) -> list[str]:
        """Return source names, resolving spec objects to their identifier."""
        out: list[str] = []
        for s in self.sources:
            if isinstance(s, str):
                out.append(s)
            elif isinstance(s, (ComboSpec, MultiMidasSpec)):
                out.append(s.name)
            else:  # MidasSpec / OLSSpec
                out.append(s.variable)
        return out

    def flatten(self) -> list[ComboSpec]:
        """Return all ComboSpec nodes in dependency order (leaves first)."""
        result: list[ComboSpec] = []
        seen: set[str] = set()
        for src in self.sources:
            if isinstance(src, ComboSpec) and src.name not in seen:
                for child in src.flatten():
                    if child.name not in seen:
                        result.append(child)
                        seen.add(child.name)
        if self.name not in seen:
            result.append(self)
            seen.add(self.name)
        return result

    def collect_indicators(
        self,
    ) -> tuple[list[MidasSpec], list[OLSSpec], list[MultiMidasSpec]]:
        """Return all `MidasSpec`, `OLSSpec` and
        `MultiMidasSpec` instances referenced anywhere in the
        combination tree, deduplicated by variable / name (first
        occurrence wins).
        """
        midas: dict[str, MidasSpec] = {}
        ols: dict[str, OLSSpec] = {}
        multi: dict[str, MultiMidasSpec] = {}

        def walk(node: ComboSpec) -> None:
            for src in node.sources:
                if isinstance(src, ComboSpec):
                    walk(src)
                elif isinstance(src, MidasSpec):
                    midas.setdefault(src.variable, src)
                elif isinstance(src, OLSSpec):
                    ols.setdefault(src.variable, src)
                elif isinstance(src, MultiMidasSpec):
                    multi.setdefault(src.name, src)
                # strings are external references — resolved by the
                # caller against the explicit *_specs arguments.

        walk(self)

        # Pass minimum_sample_size from root to direct descendants only.
        for src in self.sources:
            if isinstance(src, (MidasSpec, OLSSpec, MultiMidasSpec)):
                src.minimum_sample_size = self.minimum_sample_size

        return list(midas.values()), list(ols.values()), list(multi.values())
