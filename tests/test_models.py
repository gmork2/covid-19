"""Unit tests for the ``covid`` module extracted from ``covid19_model_comparison.ipynb``."""
from datetime import datetime

import numpy as np
import pytest
from scipy.optimize import curve_fit

from covid import date_range, exponential_model, logistic_model


def test_logistic_model_upper_asymptote_is_c():
    # As x -> +inf the model saturates at c.
    assert logistic_model(1e6, a=2.0, b=100.0, c=20000.0) == pytest.approx(20000.0)


def test_logistic_model_lower_asymptote_is_zero():
    # As x -> -inf the model tends to 0.
    assert logistic_model(-1000.0, a=2.0, b=100.0, c=20000.0) == pytest.approx(0.0)


def test_logistic_model_midpoint_is_half_of_c():
    # At x == b the model equals c / 2.
    assert logistic_model(100.0, a=2.0, b=100.0, c=20000.0) == pytest.approx(10000.0)


def test_logistic_model_supports_numpy_arrays():
    x = np.array([0.0, 100.0, 200.0])
    y = logistic_model(x, 2.0, 100.0, 20000.0)
    assert y.shape == x.shape
    assert np.all(y > 0.0)
    assert np.all(y <= 20000.0)


def test_exponential_model_equals_a_at_c():
    # At x == c the exponential model returns a.
    assert exponential_model(5.0, a=3.0, b=0.1, c=5.0) == pytest.approx(3.0)


def test_exponential_model_grows_when_b_is_positive():
    assert exponential_model(10.0, 1.0, 0.5, 0.0) > exponential_model(0.0, 1.0, 0.5, 0.0)


def test_date_range_yields_consecutive_formatted_dates():
    d1 = datetime(2020, 1, 22)
    now = datetime(2020, 1, 25)
    assert list(date_range(d1, now=now)) == [
        '01-22-2020',
        '01-23-2020',
        '01-24-2020',
    ]


def test_date_range_is_empty_when_end_is_before_start():
    d1 = datetime(2020, 1, 25)
    now = datetime(2020, 1, 22)
    assert list(date_range(d1, now=now)) == []


def test_curve_fit_recovers_logistic_parameters():
    rng = np.random.default_rng(0)
    a_true, b_true, c_true = 2.0, 100.0, 20000.0
    x = np.linspace(0.0, 200.0, 200)
    y = logistic_model(x, a_true, b_true, c_true) + rng.normal(0.0, 5.0, size=x.shape)
    (a, b, c), _ = curve_fit(logistic_model, x, y, p0=[2, 100, 20000])
    assert a == pytest.approx(a_true, rel=0.1)
    assert b == pytest.approx(b_true, rel=0.01)
    assert c == pytest.approx(c_true, rel=0.01)
