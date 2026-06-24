"""
forecasting.py
===============
Demand forecasting layer.

Modeling choice
----------------
A single global XGBoost regressor is trained across all packages
(Package_id is passed in as a feature) rather than three separate
per-package models. This is the standard production pattern for
multi-series forecasting with shared seasonal/calendar structure: pooling
series gives the tree model far more rows to learn day-of-week / holiday
/ price-elasticity patterns from, while Package_id + the package-specific
lag/rolling values still let it specialize per package.

Forecasting strategy
---------------------
We use a *recursive* (a.k.a. iterative one-step) forecast:
    1. Train the model to predict next day's Bookings from
       yesterday-and-earlier lag/rolling features.
    2. To project 14 days ahead, predict day t+1, append that prediction
       to the history, recompute lag/rolling features, predict t+2, etc.

This mirrors how the forecast will actually be consumed in production
(rolled forward daily) and lets short lags (lag_1, lag_7) progressively
pick up the model's own prior predictions for longer horizons, which is
exactly the dynamic a real "what will bookings look like 2 weeks out"
system has to live with.

Price assumption during forecasting
-------------------------------------
The forecast represents "baseline" demand: what bookings would look
like if pricing simply continued at its recent observed level. We carry
forward each package's most recent observed price (and a 28-day rolling
mean) into the future feature rows. The optimization layer (next stage)
is responsible for asking "what if we changed price?" via an estimated
elasticity curve -- the forecasting model itself is not re-run per
candidate price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from feature_engineering import FEATURE_COLUMNS, build_feature_table

TARGET = "Bookings"


def time_based_train_test_split(feat_df: pd.DataFrame, test_days: int = 30):
    """
    Holds out the last `test_days` calendar days (across all packages) as
    a test set -- a proper walk-forward style split for time series,
    never a random shuffle split.
    """
    cutoff = feat_df["Date"].max() - pd.Timedelta(days=test_days)
    train = feat_df[feat_df["Date"] <= cutoff].dropna(subset=FEATURE_COLUMNS)
    test = feat_df[feat_df["Date"] > cutoff].dropna(subset=FEATURE_COLUMNS)
    return train, test


def train_demand_model(train_df: pd.DataFrame, params: dict | None = None) -> xgb.XGBRegressor:
    """Fits an XGBoost regressor on the engineered feature table."""
    default_params = dict(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        objective="count:poisson",   # bookings are non-negative counts
        random_state=42,
    )
    if params:
        default_params.update(params)

    model = xgb.XGBRegressor(**default_params)
    model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET])
    return model


def evaluate_model(model: xgb.XGBRegressor, test_df: pd.DataFrame):
    """Computes overall + per-package MAE/RMSE on the holdout set."""
    preds = model.predict(test_df[FEATURE_COLUMNS])
    preds = np.clip(preds, 0, None)

    results = {
        "overall": {
            "MAE": mean_absolute_error(test_df[TARGET], preds),
            "RMSE": np.sqrt(mean_squared_error(test_df[TARGET], preds)),
        },
        "by_package": {},
    }

    eval_df = test_df.copy()
    eval_df["Prediction"] = preds
    for pkg, g in eval_df.groupby("Package"):
        results["by_package"][pkg] = {
            "MAE": mean_absolute_error(g[TARGET], g["Prediction"]),
            "RMSE": np.sqrt(mean_squared_error(g[TARGET], g["Prediction"])),
            "Avg_Actual": g[TARGET].mean(),
        }
    return results, eval_df


def get_feature_importance(model: xgb.XGBRegressor) -> pd.DataFrame:
    """Returns a tidy, sorted feature-importance table (gain-based)."""
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df


def recursive_forecast(model: xgb.XGBRegressor, history_df: pd.DataFrame,
                        package_categories: list, horizon_days: int = 14) -> pd.DataFrame:
    """
    Rolls the trained model forward day-by-day for `horizon_days`, for
    every package in `package_categories`, recomputing lag/rolling
    features at each step from the growing (history + predictions) frame.

    Parameters
    ----------
    history_df : the raw (Date, Package, Bookings, Price, Is_Holiday, ...)
        table -- NOT yet feature-engineered. Must contain enough trailing
        history (>= max(LAG_DAYS, ROLLING_WINDOWS) days) per package.
    package_categories : the exact category list/order the model was
        trained with (so Package_id encoding matches).

    Returns
    -------
    DataFrame with one row per (Date, Package) forecast, columns:
        Date, Package, Forecast_Bookings, Price (assumed baseline price)
    """
    from data_simulation import HOLIDAY_MMDD  # reuse the same fixed holiday calendar

    work = history_df.copy().sort_values(["Package", "Date"])
    last_date = work["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    forecasts = []

    # Carry forward each package's most recent observed price as the
    # "planned baseline price" assumption for the forecast horizon.
    last_price = work.groupby("Package")["Price"].last().to_dict()

    for current_date in future_dates:
        day_rows = []
        for pkg in package_categories:
            day_rows.append({
                "Date": current_date,
                "Package": pkg,
                "Bookings": np.nan,  # unknown -- to be filled by prediction
                "Price": last_price[pkg],
                "Is_Holiday": 1 if current_date.strftime("%m-%d") in HOLIDAY_MMDD else 0,
                "Day_Of_Week": current_date.dayofweek,
                "Is_Weekend": int(current_date.dayofweek in [4, 5, 6]),
            })
        day_df = pd.DataFrame(day_rows)
        work = pd.concat([work, day_df], ignore_index=True)

        # Recompute features on the full (history + forecasts-so-far) frame.
        feat, _ = build_feature_table(work, package_categories)
        today_feat = feat[feat["Date"] == current_date].copy()

        preds = model.predict(today_feat[FEATURE_COLUMNS])
        preds = np.clip(np.round(preds), 0, None)

        # Write predictions back into `work` so tomorrow's lag/rolling
        # features see today's forecast, exactly like a true recursive
        # multi-step forecast.
        for pkg, pred in zip(today_feat["Package"].values, preds):
            mask = (work["Date"] == current_date) & (work["Package"] == pkg)
            work.loc[mask, "Bookings"] = pred

        out = today_feat[["Date", "Package", "Price"]].copy()
        out["Forecast_Bookings"] = preds
        forecasts.append(out)

    return pd.concat(forecasts, ignore_index=True)


if __name__ == "__main__":
    from data_simulation import generate_synthetic_dataset

    raw = generate_synthetic_dataset()
    feat, categories = build_feature_table(raw)
    train, test = time_based_train_test_split(feat, test_days=30)
    print(f"Train rows: {len(train)}, Test rows: {len(test)}")

    model = train_demand_model(train)
    metrics, eval_df = evaluate_model(model, test)
    print("\nOverall holdout metrics:", metrics["overall"])
    print("\nPer-package holdout metrics:")
    for pkg, m in metrics["by_package"].items():
        print(f"  {pkg}: MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  Avg_Actual={m['Avg_Actual']:.2f}")

    print("\nTop 10 feature importances:")
    print(get_feature_importance(model).head(10))

    fc = recursive_forecast(model, raw, categories, horizon_days=14)
    print("\n14-day forecast (head):")
    print(fc.head(10))
