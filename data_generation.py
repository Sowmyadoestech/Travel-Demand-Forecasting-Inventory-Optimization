"""Synthetic data generation for travel demand forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


ROUTES = [
    "NYC-LAX",
    "NYC-MIA",
    "SFO-SEA",
    "LON-PAR",
    "DXB-SIN",
]


def generate_synthetic_data(
    n_weeks: int = 156,
    random_state: int = 42,
    routes: list[str] | None = None,
) -> pd.DataFrame:
    """Generate weekly synthetic booking demand with seasonality and lead-time effects."""
    rng = np.random.default_rng(random_state)
    routes = routes or ROUTES

    start_date = pd.Timestamp("2023-01-01")
    week_dates = pd.date_range(start_date, periods=n_weeks, freq="W")

    route_base_demand = {
        route: base
        for route, base in zip(routes, [220, 170, 130, 190, 160], strict=False)
    }
    route_base_cost = {
        route: cost
        for route, cost in zip(routes, [140, 120, 95, 110, 130], strict=False)
    }
    route_lead_time = {
        route: lead
        for route, lead in zip(routes, [28, 24, 18, 21, 26], strict=False)
    }

    records: list[dict[str, float | int | str | pd.Timestamp]] = []
    for route in routes:
        for idx, date in enumerate(week_dates):
            week_of_year = int(date.isocalendar().week)
            annual_seasonality = 1.0 + 0.22 * np.sin(2 * np.pi * week_of_year / 52)
            holiday_peak = 1.18 if week_of_year in {1, 26, 27, 51, 52} else 1.0
            trend = 1.0 + 0.0008 * idx

            days_to_departure = max(
                5,
                int(
                    route_lead_time[route]
                    + 4 * np.cos(2 * np.pi * week_of_year / 52)
                    + rng.normal(0, 2)
                ),
            )
            lead_effect = 1.0 + 0.007 * (days_to_departure - route_lead_time[route])

            noisy_demand = (
                route_base_demand[route]
                * annual_seasonality
                * holiday_peak
                * trend
                * lead_effect
                + rng.normal(0, 12)
            )
            bookings = max(20, int(round(noisy_demand)))

            records.append(
                {
                    "week_start": date,
                    "route": route,
                    "bookings": bookings,
                    "days_to_departure": days_to_departure,
                    "base_cost": route_base_cost[route],
                }
            )

    df = pd.DataFrame(records).sort_values(["route", "week_start"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    synthetic_df = generate_synthetic_data()
    print(synthetic_df.head())
    print(f"Generated rows: {len(synthetic_df)}")
