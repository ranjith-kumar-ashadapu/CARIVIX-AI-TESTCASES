import numpy as np
import pandas as pd
from predictive_analytics import moving_average_forecast, arima_forecast, growth_rate, confidence_interval


def test_moving_average_forecast_returns_correct_length():
    result = moving_average_forecast([10, 20, 30, 40, 50], window=3, steps_ahead=2)
    assert len(result) == 2


def test_moving_average_forecast_is_reasonable():
    result = moving_average_forecast([10, 10, 10], window=3, steps_ahead=1)
    assert result[0] == 10.0


def test_growth_rate_calculates_percentage_change():
    result = growth_rate([100, 110, 121])
    assert round(result[0], 2) == 10.0
    assert round(result[1], 2) == 10.0


def test_confidence_interval_returns_expected_keys():
    result = confidence_interval([10, 12, 11, 13, 9])
    assert "mean" in result
    assert "lower" in result
    assert "upper" in result
    assert result["lower"] <= result["mean"] <= result["upper"]


def test_arima_forecast_returns_requested_steps():
    series = np.linspace(100, 150, 24) + np.random.RandomState(0).normal(0, 2, 24)
    result = arima_forecast(list(series), order=(1, 1, 1), steps_ahead=4)
    assert result is not None
    assert len(result["forecast"]) == 4