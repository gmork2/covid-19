# A logistic and exponential model comparison applied to Covid-19

This is simple mathematical analysis of the infection growth to predict the
evolution of the coronavirus. The data collection used for this purpose is publicly
available as open data on github: 

- https://github.com/CSSEGISandData/COVID-19

Read the Johns Hopkins CSSE data repository terms of use for more information.

![](plot.png)
*Spain — logistic vs exponential model comparison*

## Requirements

- Python 3.14 (pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip` with a virtualenv

## Installation

### With uv (recommended)

```bash
uv sync
```

This creates the `.venv` virtual environment and installs every dependency from
the lockfile (`uv.lock`), guaranteeing reproducible installs.

### With pip

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
pip install -r requirements.txt
```

## Usage

Start Jupyter Notebook and open `covid19_model_comparison.ipynb`:

```bash
uv run jupyter notebook covid19_model_comparison.ipynb
```

Or, if you installed the environment with `pip`:

```bash
jupyter notebook covid19_model_comparison.ipynb
```

The notebook downloads the Johns Hopkins CSSE *time series* of COVID-19 deaths
(a single, schema-stable CSV), fits a logistic model to the full series and an
exponential model to the early outbreak phase for the selected country (default:
`Spain`), and plots the comparison together with the mean squared error
(residuals) of each model.

## Managing dependencies

```bash
uv add <package>      # add a dependency
uv remove <package>   # remove a dependency
```

After changing dependencies, regenerate the pip-compatible `requirements.txt`:

```bash
uv export --format requirements-txt --no-hashes -o requirements.txt
```

## Tests and linting

Run the unit test suite with:

```bash
uv run pytest
```

The tests cover the extracted models (`logistic_model`, `exponential_model`),
the `date_range` helper, the `country_deaths` data loader and the
`logistic_end_day` helper (see `tests/test_models.py`).

Run the linter and the formatter with:

```bash
uv run ruff check .
uv run ruff format --check .
```

