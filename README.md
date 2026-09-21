# Staggered-Combination MIDAS
Mixed-frequency nowcasting and short-horizon forecasting of quarterly targets (e.g. GDP) from monthly indicators, with forecast combination.

Experimental Python implementation of [Moreira (2025)](https://www.bankofengland.co.uk/macro-technical-paper/2025/nowcasting-gdp-at-the-bank-of-england-a-staggered-combination-midas-approach)

Read the [user manual](docs/index.md) for the full guide.

## Features
* MIDAS regression with Almon, exponential-Almon, Beta and unrestricted lag polynomials.
* Quarterly OLS counterpart (`OLS` / `OLSSpec`) for hard quarterly
  indicators, sharing the same forecasting and dummy interface.
* Direct multi-horizon forecasting: one model per horizon `h`, fit and stored together.
* Outlier dummies on the target (`dummy_periods`) absorbed at the target frequency.
* End-to-end combination pipeline `MidasCombo`:
    * Average, inverse-error (`mae`, `mse`, `rmse`) and regression
      weights across indicators.
    * Rolling-window and discounted-error variants.
    * Combinations of combinations (a `ComboSpec` can reference other `ComboSpec`s as sources).
* NLS estimation of non-linear weight schemes (Almon-exp, Beta).

## Project Structure

    ├── src/nowcast_midas/    # Source code
    ├── docs/            # Zensical documentation site
    ├── examples/        # Example scripts
    ├── tests/midas/     # Unit tests
    └── ...

## Installation

### From PyPi:
```bash
pip install nowcast-midas
```

### Dev version:
```bash
git clone https://github.com/bank-of-england/nowcast-midas.git
cd nowcast-midas
pip install -e .                      # runtime
pip install -e ".[dev]"               # + test / lint / docs tooling
pip install -e ".[realtime]"          # + real-time (vintage) analysis stack
```

Python ≥ 3.10.

## Quick start
```python
from nowcast_midas import ComboSpec, MidasCombo, MidasSpec, OLSSpec
from nowcast_midas.utils import sample_combo_data

# Simulated mixed-frequency data: three monthly series, one quarterly
# regressor, one quarterly target, plus one injected outlier.
target_df, regressors_df, info = sample_combo_data(n_quarters=60, seed=42)
outlier = info["outlier_date"]

midas_monthly_1 = MidasSpec(
    "monthly_1", method="almon", n_lags=6, dummy_periods=[outlier]
)
midas_monthly_2 = MidasSpec(
    "monthly_2", method="almon", n_lags=6, dummy_periods=[outlier]
)
midas_monthly_3 = MidasSpec(
    "monthly_3", method="unrestricted", n_lags=3, dummy_periods=[outlier]
)
ols_quarterly_1 = OLSSpec("quarterly_1", n_lags=1, dummy_periods=[outlier])

soft = ComboSpec(
    "soft",
    sources=[midas_monthly_1, midas_monthly_2, midas_monthly_3],
    method="mse",  # inverse mean-squared-error weights — see docs/methods/combo.md
    window=8,
    discount_rate=0.95,
)
final = ComboSpec("final", sources=[soft, ols_quarterly_1], method="regression")

model = MidasCombo(combo_specs=final, horizons=3)  # horizons is a COUNT, not an index
model.fit(target=target_df, regressors=regressors_df)

oos = model.forecast()  # long format: one row per (spec, horizon step);
# columns horizon, spec, value, date
print(oos.head())
model.summary(horizon=0)  # prints and returns the text
```

This is the same example as [`docs/index.md`](docs/index.md); runnable end-to-end scripts are in [`examples/`](examples/) and rendered in
[Worked examples](docs/examples/core_models.md).

## Selected documentation
* [SC-MIDAS framework](docs/methods/sc_midas_framework.md)
* [Weighting schemes](docs/methods/combo.md)
* [MIDAS model](docs/guide/midas_model.md)
* [SC-MIDAS pipeline](docs/guide/sc_midas.md)
* [Real-time analysis](docs/guide/realtime.md)

## Selected examples
* [MIDAS, MultiMIDAS, and SC-MIDAS](examples/midas.py)
* [SC-MIDAS with forecast-realtime](examples/realtime_midas.py)

## Contributing
* See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to the code.
* Open an issue with questions or ideas.

## Main references
* [Moreira (2025)](https://www.bankofengland.co.uk/macro-technical-paper/2025/nowcasting-gdp-at-the-bank-of-england-a-staggered-combination-midas-approach)
* [Ghysels, Santa-Clara and Valkanov (2004)](https://escholarship.org/uc/item/9mf223rs)
* [Ghysels, Sinko and Valkanov (2007)](https://doi.org/10.1080/07474930600972186)

## Data Classification
Bank of England Data Classification: OFFICIAL BLUE
