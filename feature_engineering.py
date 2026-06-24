"""
feature_engineering.py
=======================
Turns the raw daily history into a model-ready feature matrix.

Feature families:
    1. Calendar encodings  - day-of-week, month, week-of-year, cyclical
                              sin/cos transforms of day-of-year (captures
                              seasonality smoothly without one-hot blowup),
                              Is_Weekend, Is_Holiday.
    2. Trend               - a simple integer day-index since the start of
                              history, lets tree models learn a level shift
                              over time.
    3. Lag features        - Bookings_lag_{1,7,14,21,28}: what demand
                              looked like exactly k days ago for the SAME
                              package.
    4. Rolling features    - rolling mean/std of Bookings over trailing
                              7/14/28-day windows, *shifted by 1 day* so
                              no future information leaks into the row.
    5. Price features      - current price, lag-1 price, and a 28-day
                              rolling mean price (captures whether today's
                              price is a discount or a premium vs. recent
                              norm -- this is what lets the model implicitly
                              learn part of the elasticity relationship).

IMPORTANT - leakage discipline:
    All rolling/lag features are computed strictly from information that
    would have been available *before* the day being predicted. This
    matters doubly here because the same lag/rolling machinery is reused
    at inference time to roll the forecast forward day-by-day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAG_DAYS = [1, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14, 28]

# The full list of model feature columns, in a fixed order. Kept as a
# module-level constant so forecasting.py and the recursive forecast loop
# always agree on exactly what the model expects.
CALENDAR_FEATURES = [
    "Day_Of_Week", "Month", "WeekOfYear", "Is_Weekend", "Is_Holiday",
    "DayOfYear_sin", "DayOfYear_cos", "DOW_sin", "DOW_cos", "trend_idx",
]
PRICE_FEATURES = ["Price", "Price_lag_1", "Price_roll_mean_28"]
LAG_FEATURES = [f"Bookings_lag_{k}" for k in LAG_DAYS]
ROLLING_FEATURES = (
    [f"Bookings_roll_mean_{w}" for w in ROLLING_WINDOWS] +
    [f"Bookings_roll_std_{w}" for w in ROLLING_WINDOWS]
)
CATEGORICAL_FEATURES = ["Package_id"]

FEATURE_COLUMNS = (
    CALENDAR_FEATURES + PRICE_FEATURES + LAG_FEATURES + ROLLING_FEATURES + CATEGORICAL_FEATURES
)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    day_of_year = df["Date"].dt.dayofyear
    df["DayOfYear_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["DayOfYear_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df["DOW_sin"] = np.sin(2 * np.pi * df["Day_Of_Week"] / 7)
    df["DOW_cos"] = np.cos(2 * np.pi * df["Day_Of_Week"] / 7)
    min_date = df["Date"].min()
    df["trend_idx"] = (df["Date"] - min_date).dt.days
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lag + rolling-window features, computed per package so history
    from one package never bleeds into another's lags.
    """
    df = df.sort_values(["Package", "Date"]).copy()
    grp = df.groupby("Package")["Bookings"]

    for k in LAG_DAYS:
        df[f"Bookings_lag_{k}"] = grp.shift(k)

    # shift(1) before rolling => rolling stats only use information up to
    # yesterday, never today's own (target) value.
    shifted = df.groupby("Package")["Bookings"].shift(1)
    for w in ROLLING_WINDOWS:
        df[f"Bookings_roll_mean_{w}"] = (
            shifted.groupby(df["Package"]).rolling(w, min_periods=max(2, w // 3)).mean().reset_index(level=0, drop=True)
        )
        df[f"Bookings_roll_std_{w}"] = (
            shifted.groupby(df["Package"]).rolling(w, min_periods=max(2, w // 3)).std().reset_index(level=0, drop=True)
        )

    price_grp = df.groupby("Package")["Price"]
    df["Price_lag_1"] = price_grp.shift(1)
    df["Price_roll_mean_28"] = (
        price_grp.shift(1).groupby(df["Package"]).rolling(28, min_periods=5).mean().reset_index(level=0, drop=True)
    )
    return df


def encode_package(df: pd.DataFrame, package_categories: list[str] | None = None):
    """Integer-encode Package for use as a tree-model categorical feature."""
    df = df.copy()
    categories = package_categories or sorted(df["Package"].unique().tolist())
    cat_map = {name: i for i, name in enumerate(categories)}
    df["Package_id"] = df["Package"].map(cat_map).astype(int)
    return df, categories


def build_feature_table(df: pd.DataFrame, package_categories: list[str] | None = None):
    """
    Full feature pipeline: calendar + lag/rolling + categorical encoding.
    Returns (feature_df, package_categories_used).
    """
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    df, categories = encode_package(df, package_categories)
    return df, categories


if __name__ == "__main__":
    from data_simulation import generate_synthetic_dataset

    raw = generate_synthetic_dataset()
    feat, cats = build_feature_table(raw)
    print("Package categories:", cats)
    print(feat[["Date", "Package"] + FEATURE_COLUMNS + ["Bookings"]].tail(10))
    print("\nNaN counts in feature columns (expected at start of each series due to lags):")
    print(feat[FEATURE_COLUMNS].isna().sum())
