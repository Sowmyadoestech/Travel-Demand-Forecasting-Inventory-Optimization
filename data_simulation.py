"""
data_simulation.py
==================
Generates a synthetic daily-grain historical dataset for multiple travel
packages (e.g. flight + hotel bundles). The simulation deliberately bakes in
the demand drivers a revenue management model needs to learn:

    - Annual seasonality (each package has its own peak season)
    - A slow multi-year growth trend
    - Day-of-week effects (leisure travel searches/bookings spike on
      weekends)
    - Public-holiday lift
    - Price elasticity (higher price -> lower bookings, modeled with a
      constant-elasticity / power-law response)
    - Random promotional price noise + Poisson booking noise

The output is the "ground truth" history that the forecasting layer will
never see the underlying parameters of -- it only sees Date, Package,
Price, Is_Holiday, Bookings, just like a real booking engine export.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Package configuration: each package has its own seasonality phase,
# base demand level, base price, price elasticity, and trend.
# ----------------------------------------------------------------------
PACKAGE_CONFIG = {
    "Mediterranean Luxury": {
        "base_demand": 18,          # average daily bookings at base price
        "base_price": 2400.0,       # USD per package
        "min_price": 1800.0,
        "max_price": 3200.0,
        "peak_day_of_year": 200,    # mid-July -> summer peak
        "seasonal_amplitude": 0.85, # how strongly demand swings with season
        "elasticity": -1.4,         # luxury segment, fairly price sensitive
        "annual_growth": 0.06,      # 6% YoY growth in baseline demand
        "weekend_boost": 0.25,
        "holiday_boost": 0.35,
        "noise_sigma": 0.12,
    },
    "Alpine Adventure": {
        "base_demand": 12,
        "base_price": 1600.0,
        "min_price": 1100.0,
        "max_price": 2200.0,
        "peak_day_of_year": 15,     # mid-January -> winter/ski peak
        "seasonal_amplitude": 0.95,
        "elasticity": -0.9,         # adventure travelers less price sensitive
        "annual_growth": 0.04,
        "weekend_boost": 0.30,
        "holiday_boost": 0.30,
        "noise_sigma": 0.15,
    },
    "Tropical Budget": {
        "base_demand": 25,
        "base_price": 750.0,
        "min_price": 500.0,
        "max_price": 1100.0,
        "peak_day_of_year": 190,    # summer peak, but broader/flatter
        "seasonal_amplitude": 0.55,
        "elasticity": -2.1,         # budget segment, highly price sensitive
        "annual_growth": 0.08,
        "weekend_boost": 0.15,
        "holiday_boost": 0.20,
        "noise_sigma": 0.18,
    },
}

# Fixed-date holidays (MM-DD) used to flag Is_Holiday across all years.
# A small illustrative set covering major Western leisure-travel triggers.
HOLIDAY_MMDD = {
    "01-01", "02-14", "03-17", "05-26", "07-04", "09-01",
    "10-31", "11-27", "11-28", "12-24", "12-25", "12-31",
}


def _seasonal_multiplier(day_of_year: np.ndarray, peak_day: int, amplitude: float) -> np.ndarray:
    """
    Smooth annual seasonality as a cosine wave centered on `peak_day`.
    Returns a multiplier centered at 1.0, e.g. 1.0 + amplitude at the peak
    and 1.0 - amplitude at the trough.
    """
    phase = 2 * np.pi * (day_of_year - peak_day) / 365.25
    return 1.0 + amplitude * np.cos(phase)


def _simulate_price_path(n_days: int, base_price: float, min_price: float,
                          max_price: float, seasonal_mult: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Simulate a realistic *historical* price path: prices drift up in
    high season (airlines/hotels raise rack rates with demand) and get
    discounted with small random promotions. This is the price the
    booking actually happened at -- it is what creates the price/demand
    variation the elasticity model will later need to estimate.
    """
    # Prices follow season (slightly), plus a slow random walk, plus
    # occasional promo discounts.
    seasonal_price = base_price * (1 + 0.18 * (seasonal_mult - 1))
    random_walk = np.cumsum(rng.normal(0, 1.5, size=n_days))
    random_walk -= random_walk.mean()  # keep it centered, avoid runaway drift
    promo = rng.choice([0, 1], size=n_days, p=[0.92, 0.08]) * rng.uniform(0.08, 0.20, size=n_days)
    price = seasonal_price + random_walk
    price = price * (1 - promo)
    return np.clip(price, min_price * 0.95, max_price * 1.05)


def simulate_package_demand(package_name: str, cfg: dict, dates: pd.DatetimeIndex,
                             rng: np.random.Generator) -> pd.DataFrame:
    """Simulate the full daily history for a single package."""
    n_days = len(dates)
    day_of_year = dates.dayofyear.values
    years_elapsed = (dates - dates[0]).days / 365.25

    seasonal_mult = _seasonal_multiplier(day_of_year, cfg["peak_day_of_year"], cfg["seasonal_amplitude"])
    trend_mult = (1 + cfg["annual_growth"]) ** years_elapsed

    dow = dates.dayofweek.values  # 0=Mon ... 6=Sun
    is_weekend = np.isin(dow, [4, 5, 6]).astype(int)  # Fri/Sat/Sun travel bookings spike
    weekend_mult = 1 + cfg["weekend_boost"] * is_weekend

    mmdd = dates.strftime("%m-%d")
    is_holiday = np.array([1 if d in HOLIDAY_MMDD else 0 for d in mmdd])
    holiday_mult = 1 + cfg["holiday_boost"] * is_holiday

    price = _simulate_price_path(n_days, cfg["base_price"], cfg["min_price"],
                                  cfg["max_price"], seasonal_mult, rng)

    # Constant-elasticity price response, centered on the package's base price.
    price_mult = (price / cfg["base_price"]) ** cfg["elasticity"]

    # Multiplicative noise (log-normal) on top of the deterministic mean.
    noise_mult = np.exp(rng.normal(0, cfg["noise_sigma"], size=n_days))

    expected_demand = (cfg["base_demand"] * seasonal_mult * trend_mult *
                        weekend_mult * holiday_mult * price_mult * noise_mult)
    expected_demand = np.clip(expected_demand, 0.1, None)

    # Poisson draw -> realistic non-negative integer bookings count.
    bookings = rng.poisson(expected_demand)

    df = pd.DataFrame({
        "Date": dates,
        "Package": package_name,
        "Bookings": bookings,
        "Price": np.round(price, 2),
        "Is_Holiday": is_holiday,
        "Day_Of_Week": dow,
        "Is_Weekend": is_weekend,
    })
    return df


def generate_synthetic_dataset(start_date: str = "2023-01-01", n_days: int = 1095,
                                seed: int = 42) -> pd.DataFrame:
    """
    Generate the full multi-package synthetic dataset.

    Parameters
    ----------
    start_date : str
        First date of the simulated history.
    n_days : int
        Number of days to simulate (default ~3 years).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame sorted by Package, Date with columns:
        Date, Package, Bookings, Price, Is_Holiday, Day_Of_Week, Is_Weekend
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    frames = [
        simulate_package_demand(pkg_name, cfg, dates, rng)
        for pkg_name, cfg in PACKAGE_CONFIG.items()
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["Package", "Date"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    data = generate_synthetic_dataset()
    print(data.head())
    print("\nShape:", data.shape)
    print("\nBookings summary by package:")
    print(data.groupby("Package")["Bookings"].describe()[["mean", "std", "min", "max"]])
    data.to_csv("synthetic_travel_demand.csv", index=False)
    print("\nSaved synthetic_travel_demand.csv")
