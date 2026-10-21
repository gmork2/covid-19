"""Core models and utilities for the Covid-19 growth analysis.

These functions are extracted from ``covid19_model_comparison.ipynb`` so they can
be unit-tested independently of the notebook.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

#: Date format used throughout the project.
FMT = "%m-%d-%Y"

#: Date format used for the columns of the Johns Hopkins time-series files.
TS_FMT = "%m/%d/%y"

#: Reference date used to convert calendar dates into "day numbers".
DATE_ZERO = datetime.strptime("01-01-2020", FMT)

TIME_SERIES_DEATHS_URL = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
    "time_series_covid19_deaths_global.csv"
)

#: Columns of the time-series files that are metadata rather than daily counts.
_NON_DATE_COLUMNS = ("Province/State", "Country/Region", "Lat", "Long")

__all__ = [
    "FMT",
    "TS_FMT",
    "DATE_ZERO",
    "TIME_SERIES_DEATHS_URL",
    "date_range",
    "logistic_model",
    "exponential_model",
    "logistic_end_day",
    "load_deaths",
    "country_deaths",
]


def date_range(d1: datetime, now: datetime | None = None) -> Iterator[str]:
    """Yield the dates from ``d1`` up to (but excluding) ``now``.

    :param d1: Start date.
    :param now: End date. Defaults to the current time; injectable for testing.
    :return: Iterator of date strings formatted as ``FMT``.
    """
    if now is None:
        now = datetime.now()
    while d1 < now:
        yield d1.strftime(FMT)
        d1 += timedelta(days=1)


def logistic_model(x, a: float, b: float, c: float):
    """Growth of the population of a pathogen agent.

    :param x: Time.
    :param a: Infection speed (width of the transition).
    :param b: Day with the maximum infections occurred (inflection point).
    :param c: Total number of recorded infected people at the infection's end.
    :return: Model value(s) at ``x``.
    """
    return c / (1 + np.exp(-(x - b) / a))


def exponential_model(x, a: float, b: float, c: float):
    """Exponential growth model ``a * exp(b * (x - c))``.

    Note that ``a`` and ``c`` are not independently identifiable: the model only
    depends on the product ``a * exp(-b * c)``.

    :param x: Time.
    :param a: Amplitude scale.
    :param b: Growth rate.
    :param c: Time offset.
    :return: Model value(s) at ``x``.
    """
    return a * np.exp(b * (x - c))


def logistic_end_day(a: float, b: float, c: float, fraction: float = 0.99) -> float:
    """Return the day at which the logistic model reaches ``fraction`` of ``c``.

    The logistic curve only reaches its asymptote ``c`` at ``x == +inf``, so
    asking it to equal ``c`` exactly is ill-defined. This helper solves
    ``logistic_model(x, a, b, c) == fraction * c`` analytically.

    :param a: Infection speed.
    :param b: Inflection point (day of maximum infections).
    :param c: Asymptote (total infections at the end).
    :param fraction: Fraction of the asymptote used as the "end" threshold.
    :return: Day ``x`` at which the model reaches ``fraction`` of ``c``.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in the open interval (0, 1)")
    return b - a * np.log(1.0 / fraction - 1.0)


def load_deaths(url: str = TIME_SERIES_DEATHS_URL) -> pd.DataFrame:
    """Download the Johns Hopkins global COVID-19 deaths time series.

    :param url: Location of the time-series CSV.
    :return: DataFrame with one row per country/province and one column per date.
    """
    return pd.read_csv(url)


def country_deaths(df: pd.DataFrame, country: str):
    """Aggregate the total deaths for ``country`` into ``(x, y)`` arrays.

    :param df: Time-series DataFrame as returned by :func:`load_deaths`.
    :param country: ``Country/Region`` label (e.g. ``'Spain'``).
    :return: Tuple ``(x, y)`` where ``x`` holds the day number since 2020-01-01
        and ``y`` the cumulative deaths for that day.
    :raises ValueError: If ``country`` is not present in ``df``.
    """
    mask = df["Country/Region"] == country
    if not mask.any():
        raise ValueError(f"No data found for country: {country!r}")

    date_columns = [c for c in df.columns if c not in _NON_DATE_COLUMNS]
    deaths = df.loc[mask, date_columns].sum(axis=0)

    x = np.array([_day_number(d) for d in date_columns], dtype=float)
    y = deaths.to_numpy(dtype=float)
    return x, y


def _day_number(date_str: str, fmt: str = TS_FMT) -> int:
    """Convert a time-series date label into days since ``DATE_ZERO``."""
    return (datetime.strptime(date_str, fmt) - DATE_ZERO).days
