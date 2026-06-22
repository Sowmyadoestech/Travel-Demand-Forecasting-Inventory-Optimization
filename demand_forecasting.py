"""Demand forecasting module with XGBoost and SARIMA baseline."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from xgboost import XGBRegressor

from statsmodels.tsa.statespace.sarimax import SARIMAX


FEATURE_COLUMNS = [
    "week_of_year",
    "days_to_departure",
    "route_code",
    "rolling_avg_4w",
    "rolling_avg_12w",
]


@dataclass
class TrainingArtifacts:
    xgb_model: XGBRegressor
    sarima_forecasts: pd.DataFrame
    evaluation: dict[str, dict[str, float]]
    feature_columns: list[str]
    test_frame: pd.DataFrame


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["week_start"] = pd.to_datetime(data["week_start"])
    data = data.sort_values(["route", "week_start"]).reset_index(drop=True)

    data["week_of_year"] = data["week_start"].dt.isocalendar().week.astype(int)
    data["route_code"] = data["route"].astype("category").cat.codes
    data["rolling_avg_4w"] = (
        data.groupby("route")["bookings"]
        .transform(lambda s: s.shift(1).rolling(window=4, min_periods=1).mean())
        .fillna(data["bookings"].mean())
    )
    data["rolling_avg_12w"] = (
        data.groupby("route")["bookings"]
        .transform(lambda s: s.shift(1).rolling(window=12, min_periods=1).mean())
        .fillna(data["bookings"].mean())
    )
    return data


def _time_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_weeks = np.sort(df["week_start"].unique())
    split_index = int(len(unique_weeks) * (1 - test_size))
    cutoff = unique_weeks[split_index]
    train_df = df[df["week_start"] < cutoff].copy()
    test_df = df[df["week_start"] >= cutoff].copy()
    return train_df, test_df


def train_models(df: pd.DataFrame) -> TrainingArtifacts:
    prepared = _prepare_features(df)
    train_df, test_df = _time_split(prepared)

    xgb_model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    xgb_model.fit(train_df[FEATURE_COLUMNS], train_df["bookings"])

    xgb_preds = np.maximum(0, xgb_model.predict(test_df[FEATURE_COLUMNS]))

    sarima_forecasts: list[pd.DataFrame] = []
    for route, route_train in train_df.groupby("route"):
        route_test = test_df[test_df["route"] == route].copy()
        sarima = SARIMAX(
            route_train["bookings"],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 52),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = sarima.fit(disp=False)
        pred = fitted.forecast(steps=len(route_test))
        route_test["sarima_pred"] = np.maximum(0, pred.values)
        sarima_forecasts.append(route_test[["week_start", "route", "sarima_pred"]])

    sarima_forecast_df = pd.concat(sarima_forecasts, ignore_index=True)
    comparison = test_df.merge(sarima_forecast_df, on=["week_start", "route"], how="left")
    comparison["xgb_pred"] = xgb_preds

    evaluation = {
        "xgboost": {
            "mape": float(
                mean_absolute_percentage_error(comparison["bookings"], comparison["xgb_pred"])
                * 100
            ),
            "rmse": float(
                np.sqrt(mean_squared_error(comparison["bookings"], comparison["xgb_pred"]))
            ),
        },
        "sarima": {
            "mape": float(
                mean_absolute_percentage_error(
                    comparison["bookings"], comparison["sarima_pred"]
                )
                * 100
            ),
            "rmse": float(
                np.sqrt(mean_squared_error(comparison["bookings"], comparison["sarima_pred"]))
            ),
        },
    }

    return TrainingArtifacts(
        xgb_model=xgb_model,
        sarima_forecasts=sarima_forecast_df,
        evaluation=evaluation,
        feature_columns=FEATURE_COLUMNS,
        test_frame=comparison,
    )


def forecast_with_xgboost(model: XGBRegressor, df: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_features(df)
    prepared["forecast"] = np.maximum(0, model.predict(prepared[FEATURE_COLUMNS]))
    return prepared[["week_start", "route", "forecast", "base_cost"]]


def plot_forecasts_vs_actuals(test_frame: pd.DataFrame, save_path: str | None = None) -> None:
    route_actual = (
        test_frame.groupby("week_start", as_index=False)["bookings"].sum().rename(columns={"bookings": "actual"})
    )
    route_xgb = (
        test_frame.groupby("week_start", as_index=False)["xgb_pred"].sum().rename(columns={"xgb_pred": "xgboost"})
    )
    route_sarima = (
        test_frame.groupby("week_start", as_index=False)["sarima_pred"].sum().rename(columns={"sarima_pred": "sarima"})
    )

    plot_df = route_actual.merge(route_xgb, on="week_start").merge(route_sarima, on="week_start")

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["week_start"], plot_df["actual"], label="Actual", linewidth=2)
    plt.plot(plot_df["week_start"], plot_df["xgboost"], label="XGBoost", linestyle="--")
    plt.plot(plot_df["week_start"], plot_df["sarima"], label="SARIMA", linestyle=":")
    plt.title("Forecasts vs Actuals (Aggregated Across Routes)")
    plt.xlabel("Week")
    plt.ylabel("Bookings")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
    else:
        plt.show()
