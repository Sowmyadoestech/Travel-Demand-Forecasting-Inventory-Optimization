"""Linear programming model for seat and price optimization."""

from __future__ import annotations

import pandas as pd
import pulp


def optimize_inventory_pricing(
    forecasts: pd.DataFrame,
    capacity: int = 500,
    constraints: dict | None = None,
) -> pd.DataFrame:
    """Optimize seat allocation and price tier selection to maximize expected revenue."""
    constraints = constraints or {}
    min_occupancy = float(constraints.get("min_occupancy", 0.60))
    margin_floor = float(constraints.get("margin_floor", 0.15))
    price_tiers = constraints.get("price_tiers", [0.95, 1.00, 1.05, 1.10])

    latest = forecasts.sort_values("week_start").groupby("route", as_index=False).tail(1)
    latest = latest.rename(columns={"forecast": "demand"}).copy()

    problem = pulp.LpProblem("RevenueMaximization", pulp.LpMaximize)

    alloc_vars: dict[tuple[str, float], pulp.LpVariable] = {}
    revenue_terms = []

    for _, row in latest.iterrows():
        route = str(row["route"])
        demand = float(row["demand"])
        base_cost = float(row["base_cost"])
        min_price = base_cost * (1 + margin_floor)

        for tier in price_tiers:
            price = max(base_cost * float(tier), min_price)
            tier_demand = max(0.0, demand * (1.08 - 0.25 * (price / max(base_cost, 1))))

            var = pulp.LpVariable(f"x_{route.replace('-', '_')}_{tier}", lowBound=0)
            alloc_vars[(route, tier)] = var
            problem += var <= tier_demand
            revenue_terms.append(var * price)

    problem += pulp.lpSum(revenue_terms)

    problem += pulp.lpSum(alloc_vars.values()) <= capacity

    min_route_allocation = capacity * min_occupancy / len(latest)
    for route in latest["route"]:
        problem += (
            pulp.lpSum(alloc_vars[(str(route), tier)] for tier in price_tiers)
            >= min_route_allocation
        )

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    rows = []
    for route in latest["route"]:
        allocations = []
        prices = []
        route = str(route)
        route_cost = float(latest[latest["route"] == route]["base_cost"].iloc[0])

        for tier in price_tiers:
            allocated = float(alloc_vars[(route, tier)].value() or 0.0)
            if allocated > 1e-6:
                price = max(route_cost * float(tier), route_cost * (1 + margin_floor))
                allocations.append(allocated)
                prices.append(price)

        total_alloc = float(sum(allocations))
        weighted_price = float(
            sum(a * p for a, p in zip(allocations, prices, strict=False)) / total_alloc
        ) if total_alloc > 0 else route_cost * (1 + margin_floor)

        rows.append(
            {
                "route": route,
                "optimal_allocation": round(total_alloc, 2),
                "optimal_price": round(weighted_price, 2),
            }
        )

    return pd.DataFrame(rows).sort_values("route").reset_index(drop=True)
