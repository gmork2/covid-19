"""Core models and utilities for the Covid-19 growth analysis.

These functions are extracted from ``pred.ipynb`` so they can be unit-tested
independently of the notebook.
"""
from datetime import datetime, timedelta

import numpy as np

FMT = '%m-%d-%Y'


def date_range(d1, now=None):
    """
    Yields the dates from d1 to now.

    :param d1: Start date.
    :param now: End date. Defaults to the current time; injectable for testing.
    :return: Generator of date strings formatted as ``FMT``.
    """
    if now is None:
        now = datetime.now()
    while d1 < now:
        yield d1.strftime(FMT)
        d1 += timedelta(days=1)


def logistic_model(x, a, b, c):
    """
    Growth of the population of a pathogen agent.

    :param x: Time.
    :param a: Refers to the infection speed.
    :param b: Day with the maximum infections occurred.
    :param c: Total number of recorded infected people at the infection's end.
    :return:
    """
    return c / (1 + np.exp(-(x - b) / a))


def exponential_model(x, a, b, c):
    return a * np.exp(b * (x - c))
