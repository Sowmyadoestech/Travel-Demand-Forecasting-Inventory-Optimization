"""Main API module exposing train, predict and optimise functions."""

from __future__ import annotations

import pandas as pd

from demand_forecasting import (
    TrainingArtifacts,
    forecast_with_xgboost,
    plot_forecasts_vs_actuals,
    train_models,
)
from optimization import optimize_inventory_pricing


def train(df: pd.DataFrame) -> TrainingArtifacts:
    """Train XGBoost and SARIMA models and return training artifacts."""
    return train_models(df)


def predict(model: TrainingArtifacts, df: pd.DataFrame) -> pd.DataFrame:
    """Generate demand forecasts using trained XGBoost model."""
    return forecast_with_xgboost(model.xgb_model, df)


def optimise(
    forecasts: pd.DataFrame,
    capacity: int = 500,
    constraints: dict | None = None,
) -> pd.DataFrame:
    """Optimize inventory allocation and pricing with linear programming."""
    return optimize_inventory_pricing(forecasts, capacity=capacity, constraints=constraints)


def plot_evaluation(model: TrainingArtifacts, save_path: str | None = None) -> None:
    """Plot forecast comparison from held-out test data."""
    plot_forecasts_vs_actuals(model.test_frame, save_path=save_path)
