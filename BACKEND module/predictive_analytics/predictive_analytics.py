"""
CARIVIX AI - Predictive Analytics Engine
Reusable forecasting helper functions covering Business, Economic,
Government, and Smart City domains.

Optional heavier dependencies (statsmodels for ARIMA, prophet) are
imported lazily so this module still loads if they aren't installed;
install with:
    pip install statsmodels prophet --break-system-packages
"""

import logging
from typing import List, Dict, Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("carivix.predictive_analytics")


def train_test_split_timeseries(df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, pd.DataFrame]:
    """
    Chronological train/test split for time-series data (no shuffling,
    since forecasting must respect time order). df must already be sorted
    by date ascending.
    """
    split_idx = int(len(df) * (1 - test_size))
    return {"train": df.iloc[:split_idx].reset_index(drop=True),
            "test": df.iloc[split_idx:].reset_index(drop=True)}


def moving_average_forecast(series: Sequence[float], window: int = 3, steps_ahead: int = 1) -> List[float]:
    """
    Simple, dependency-free baseline forecast using a rolling average.
    Good as a sanity-check baseline before comparing against ARIMA/LSTM/etc.
    """
    values = list(series)
    forecasts = []
    for _ in range(steps_ahead):
        window_slice = values[-window:]
        next_val = float(np.mean(window_slice))
        forecasts.append(next_val)
        values.append(next_val)
    return forecasts


def arima_forecast(
    series: Sequence[float],
    order: tuple = (1, 1, 1),
    steps_ahead: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    Fit an ARIMA model and forecast N steps ahead.
    Used for GDP/inflation, demand, and traffic-volume style forecasting.

    Requires: pip install statsmodels --break-system-packages
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        logger.error("statsmodels is not installed. Run: pip install statsmodels --break-system-packages")
        return None

    model = ARIMA(series, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=steps_ahead)
    return {
        "forecast": list(forecast),
        "aic": fitted.aic,
        "order": order,
    }


def prophet_forecast(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    periods_ahead: int = 30,
    freq: str = "D",
) -> Optional[pd.DataFrame]:
    """
    Fit Meta's Prophet model for trend + seasonality forecasting
    (e.g. traffic, energy demand, sales).

    df must have a date column and a numeric value column.
    Requires: pip install prophet --break-system-packages
    """
    try:
        from prophet import Prophet
    except ImportError:
        logger.error("prophet is not installed. Run: pip install prophet --break-system-packages")
        return None

    prophet_df = df[[date_column, value_column]].rename(
        columns={date_column: "ds", value_column: "y"}
    )
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods_ahead, freq=freq)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def train_regression_forecaster(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 200,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a Random Forest regressor for feature-driven forecasting
    (e.g. revenue/demand forecasting using engineered features rather
    than pure time-series). Returns the fitted model plus evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "r2": r2_score(y_test, preds),
    }
    logger.info("Random Forest forecaster trained. Metrics: %s", metrics)
    return {"model": model, "metrics": metrics}


def confidence_interval(forecast_values: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    """
    Compute a simple normal-approximation confidence interval around a
    set of forecast values (e.g. across multiple model runs or bootstrap
    samples), for use in executive-dashboard "range" displays.
    """
    values = np.array(forecast_values, dtype=float)
    mean = values.mean()
    std_err = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
    return {"mean": float(mean), "lower": float(mean - z * std_err), "upper": float(mean + z * std_err)}


def growth_rate(series: Sequence[float]) -> List[float]:
    """
    Period-over-period percentage growth rate — the basic building block
    for GDP growth, revenue growth, and population growth comparisons.
    """
    values = list(series)
    rates = []
    for i in range(1, len(values)):
        prev, curr = values[i - 1], values[i]
        rates.append(((curr - prev) / prev) * 100 if prev != 0 else float("nan"))
    return rates
