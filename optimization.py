"""
optimization.py
=================
Revenue optimization layer: turns 14-day demand forecasts into an optimal
daily price for each package, subject to inventory constraints.

--------------------------------------------------------------------------
STEP A - Estimating price elasticity from history
--------------------------------------------------------------------------
The forecasting model predicts *baseline* demand at the recently-observed
price. To know what demand would be at a *different* candidate price, we
need each package's price elasticity of demand. We estimate it directly
from the historical data with a log-log regression (the standard
econometric form for constant-elasticity demand):

    ln(Bookings_t) = alpha + beta * ln(Price_t)
                            + gamma_1 * Is_Holiday_t + gamma_2 * Is_Weekend_t
                            + c1*sin(2*pi*doy/365) + c2*cos(2*pi*doy/365)
                            + c3*sin(4*pi*doy/365) + c4*cos(4*pi*doy/365)
                            + delta * trend_idx_t  + epsilon_t

beta is the price elasticity of demand: the % change in bookings for a 1%
change in price, holding seasonality/holiday/weekend/trend effects fixed.
We expect beta < 0. Controlling for seasonality and trend matters because,
in the raw data, prices are *higher* in high season -- a naive regression
of bookings on price alone would be badly confounded (it would look like
raising price increases demand, the opposite of the truth) if we didn't
net out seasonality first. Smooth sin/cos annual harmonics (rather than
coarse month dummies) are used deliberately: a package with a sharp
seasonal peak that falls mid-month (e.g. a ski package peaking in
mid-January) has substantial within-month seasonal swing that month
dummies can't absorb, letting that leftover seasonality leak into -- and
even flip the sign of -- the price coefficient.

--------------------------------------------------------------------------
STEP B - Building a demand-price curve per package, per day
--------------------------------------------------------------------------
For package p on day t, the forecasting layer gives us D0_{p,t}: expected
bookings at the reference/baseline price P0_p (the most recent observed
price). For any candidate price P_k, we project demand using the
constant-elasticity relationship:

    D_{p,t,k} = D0_{p,t} * (P_k / P0_p) ^ beta_p

This is a smooth, continuous demand curve. To keep the resulting pricing
problem a *linear* (technically mixed-integer linear) program -- solvable
quickly and reliably with open-source MILP solvers -- we discretize each
package's allowed price range [min_price_p, max_price_p] into a small set
of K candidate price tiers, and pre-compute D_{p,t,k} for every (package,
day, tier) combination as plain numeric constants. The optimizer's job is
then just to *select* one tier per package per day, not to search a
continuous price space.

--------------------------------------------------------------------------
STEP C - The Mixed-Integer Linear Program
--------------------------------------------------------------------------
Decision variables:
    x_{p,t,k} ∈ {0, 1}   for every package p, day t in the 14-day horizon,
                          and price tier k.
    x_{p,t,k} = 1  <=>  package p is priced at tier k on day t.

Objective (maximize total expected revenue):

    maximize   sum_{p,t,k}  x_{p,t,k} * P_k * D_{p,t,k}

Constraints:

  (1) Exactly one price tier chosen per package per day:
          sum_k  x_{p,t,k} = 1                       for every (p, t)

  (2) Daily operational capacity (e.g. check-in desks, transfer buses,
      max rooms releasable in a single night) -- expected bookings at the
      chosen tier cannot exceed the package's daily capacity:
          sum_k  x_{p,t,k} * D_{p,t,k}  <=  DailyCapacity_p     for every (p, t)

  (3) Total horizon inventory / allotment (e.g. a fixed block of hotel
      rooms or charter-flight seats contracted for the whole 14-day
      window) -- cumulative expected bookings across the full horizon
      cannot exceed the package's total allotment:
          sum_{t,k}  x_{p,t,k} * D_{p,t,k}  <=  TotalInventory_p     for every p

  (4) Price bounds are enforced implicitly: every candidate tier P_k is
      already generated within [min_price_p, max_price_p], so no
      separate bound constraint is needed on price itself.

Because P_k and D_{p,t,k} are pre-computed constants (not decision
variables), the objective and every constraint above are linear in x --
this is a clean 0/1 MILP, solved here with PuLP's bundled CBC solver.

--------------------------------------------------------------------------
Handling infeasibility
--------------------------------------------------------------------------
Constraint (3) can make the problem infeasible if even the
highest-price/lowest-demand tier's cumulative demand exceeds the
allotment (i.e., there is simply not enough inventory to avoid turning
demand away no matter how high the price goes). `solve_pricing_milp`
detects non-Optimal solver statuses and raises a clear, actionable error
rather than silently returning nonsense; see the production-scaling notes
in README.md for how this is best handled live (e.g. soft/slack
constraints, waitlists, or dynamically re-sourcing inventory).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pulp


# --------------------------------------------------------------------------
# STEP A: Elasticity estimation
# --------------------------------------------------------------------------
def estimate_price_elasticity(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits, per package, the log-log demand regression described above via
    OLS (numpy least squares).

    Seasonality control -- why sin/cos harmonics instead of month dummies:
    Calendar-month dummies are too coarse when a package has a sharp
    seasonal peak that straddles a month boundary (e.g. a ski package
    peaking mid-January): within-month swings in the *true* seasonal
    demand level remain unexplained, and since price also moves with
    season, that leftover seasonal signal leaks into the price
    coefficient and can even flip its sign. Using smooth annual sin/cos
    harmonics (plus a 2nd harmonic for sharper, less sinusoidal peaks)
    and a linear trend term controls for seasonality and multi-year
    growth continuously, which matches how the underlying demand process
    actually varies and gives a materially cleaner elasticity estimate.

    Returns
    -------
    DataFrame indexed by Package with columns: elasticity, r_squared
    """
    results = []
    for pkg, g in history_df.groupby("Package"):
        g = g[g["Bookings"] > 0].copy()  # log(0) undefined
        y = np.log(g["Bookings"].values)
        log_price = np.log(g["Price"].values)

        doy = g["Date"].dt.dayofyear.values
        sin1, cos1 = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        sin2, cos2 = np.sin(4 * np.pi * doy / 365.25), np.cos(4 * np.pi * doy / 365.25)
        trend = (g["Date"] - g["Date"].min()).dt.days.values.astype(float)

        X = np.column_stack([
            np.ones(len(g)),
            log_price,
            g["Is_Holiday"].values.astype(float),
            g["Is_Weekend"].values.astype(float),
            sin1, cos1, sin2, cos2,
            trend,
        ])

        coefs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ coefs
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot

        results.append({"Package": pkg, "elasticity": coefs[1], "r_squared": r_squared})

    return pd.DataFrame(results).set_index("Package")


# --------------------------------------------------------------------------
# STEP B: Price tiers + demand curve
# --------------------------------------------------------------------------
def generate_price_tiers(min_price: float, max_price: float, n_tiers: int = 5) -> np.ndarray:
    """Evenly spaced candidate price points within the allowed bounds."""
    return np.linspace(min_price, max_price, n_tiers)


def build_demand_price_table(forecast_df: pd.DataFrame, elasticity_df: pd.DataFrame,
                              price_bounds: dict, n_tiers: int = 5) -> pd.DataFrame:
    """
    For every (Date, Package) forecast row, expands into n_tiers candidate
    price rows with the projected demand at each tier.

    Parameters
    ----------
    forecast_df : output of recursive_forecast() -- columns Date, Package,
        Price (baseline/reference price), Forecast_Bookings (D0).
    elasticity_df : output of estimate_price_elasticity().
    price_bounds : {package_name: (min_price, max_price)}.

    Returns
    -------
    Long-format DataFrame: Date, Package, Tier, Price, Baseline_Price,
        Baseline_Demand, Projected_Demand
    """
    rows = []
    for _, r in forecast_df.iterrows():
        pkg = r["Package"]
        base_price = r["Price"]
        base_demand = r["Forecast_Bookings"]
        beta = elasticity_df.loc[pkg, "elasticity"]
        min_p, max_p = price_bounds[pkg]

        tiers = generate_price_tiers(min_p, max_p, n_tiers)
        for k, price_k in enumerate(tiers):
            projected_demand = base_demand * (price_k / base_price) ** beta
            rows.append({
                "Date": r["Date"],
                "Package": pkg,
                "Tier": k,
                "Price": round(float(price_k), 2),
                "Baseline_Price": round(float(base_price), 2),
                "Baseline_Demand": float(base_demand),
                "Projected_Demand": max(float(projected_demand), 0.0),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# STEP C: The MILP
# --------------------------------------------------------------------------
def solve_pricing_milp(demand_price_df: pd.DataFrame, daily_capacity: dict,
                        total_inventory: dict) -> pd.DataFrame:
    """
    Builds and solves the tier-selection MILP described in the module
    docstring.

    Parameters
    ----------
    demand_price_df : output of build_demand_price_table().
    daily_capacity : {package_name: max bookings sellable in a single day}.
    total_inventory : {package_name: total allotment across the full
        forecast horizon}.

    Returns
    -------
    DataFrame: Date, Package, Optimal_Price, Expected_Bookings,
        Expected_Revenue -- one row per (Date, Package).

    Raises
    ------
    RuntimeError if the solver cannot find an optimal solution (e.g. the
    total_inventory constraint makes the problem infeasible).
    """
    prob = pulp.LpProblem("Travel_Package_Revenue_Optimization", pulp.LpMaximize)

    packages = demand_price_df["Package"].unique().tolist()
    dates = sorted(demand_price_df["Date"].unique())

    # Decision variables: x[(package, date, tier)] -> binary
    x = {}
    for _, row in demand_price_df.iterrows():
        key = (row["Package"], row["Date"], row["Tier"])
        x[key] = pulp.LpVariable(f"x_{row['Package']}_{row['Date'].date()}_{row['Tier']}".replace(" ", "_"),
                                  cat="Binary")

    # Objective: maximize total expected revenue across all (package, day, tier).
    prob += pulp.lpSum(
        x[(row["Package"], row["Date"], row["Tier"])] * row["Price"] * row["Projected_Demand"]
        for _, row in demand_price_df.iterrows()
    ), "Total_Expected_Revenue"

    # Constraint (1): exactly one tier chosen per (package, day).
    for pkg in packages:
        for dt in dates:
            day_rows = demand_price_df[(demand_price_df["Package"] == pkg) & (demand_price_df["Date"] == dt)]
            prob += (
                pulp.lpSum(x[(pkg, dt, t)] for t in day_rows["Tier"]) == 1,
                f"OneTier_{pkg}_{dt.date()}".replace(" ", "_"),
            )

            # Constraint (2): daily operational capacity.
            prob += (
                pulp.lpSum(
                    x[(pkg, dt, t)] * d for t, d in zip(day_rows["Tier"], day_rows["Projected_Demand"])
                ) <= daily_capacity[pkg],
                f"DailyCap_{pkg}_{dt.date()}".replace(" ", "_"),
            )

    # Constraint (3): total horizon inventory / allotment per package.
    for pkg in packages:
        pkg_rows = demand_price_df[demand_price_df["Package"] == pkg]
        prob += (
            pulp.lpSum(
                x[(pkg, row["Date"], row["Tier"])] * row["Projected_Demand"]
                for _, row in pkg_rows.iterrows()
            ) <= total_inventory[pkg],
            f"TotalInventory_{pkg}".replace(" ", "_"),
        )

    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(
            f"Optimization did not reach an optimal solution (status='{status}'). "
            "This typically means total_inventory is too tight relative to forecasted "
            "demand even at maximum price -- see README.md 'Handling infeasibility'."
        )

    # Extract the chosen tier for each (package, day).
    results = []
    for pkg in packages:
        for dt in dates:
            day_rows = demand_price_df[(demand_price_df["Package"] == pkg) & (demand_price_df["Date"] == dt)]
            for _, row in day_rows.iterrows():
                if pulp.value(x[(pkg, dt, row["Tier"])]) > 0.5:
                    results.append({
                        "Date": dt,
                        "Package": pkg,
                        "Optimal_Price": row["Price"],
                        "Expected_Bookings": round(row["Projected_Demand"], 1),
                        "Expected_Revenue": round(row["Price"] * row["Projected_Demand"], 2),
                    })
                    break

    return pd.DataFrame(results).sort_values(["Package", "Date"]).reset_index(drop=True)


if __name__ == "__main__":
    from data_simulation import generate_synthetic_dataset, PACKAGE_CONFIG
    from feature_engineering import build_feature_table
    from forecasting import time_based_train_test_split, train_demand_model, recursive_forecast

    raw = generate_synthetic_dataset()

    elasticity_df = estimate_price_elasticity(raw)
    print("Estimated price elasticities (ground truth in parentheses):")
    for pkg, row in elasticity_df.iterrows():
        truth = PACKAGE_CONFIG[pkg]["elasticity"]
        print(f"  {pkg}: estimated={row['elasticity']:.2f}  (ground truth={truth})  R2={row['r_squared']:.2f}")

    feat, categories = build_feature_table(raw)
    train, _ = time_based_train_test_split(feat, test_days=30)
    model = train_demand_model(train)
    forecast_df = recursive_forecast(model, raw, categories, horizon_days=14)

    price_bounds = {pkg: (cfg["min_price"], cfg["max_price"]) for pkg, cfg in PACKAGE_CONFIG.items()}
    dp_table = build_demand_price_table(forecast_df, elasticity_df, price_bounds, n_tiers=5)
    print("\nDemand-price table sample:")
    print(dp_table.head(10))

    daily_capacity = {"Mediterranean Luxury": 15, "Alpine Adventure": 40, "Tropical Budget": 60}
    total_inventory = {"Mediterranean Luxury": 75, "Alpine Adventure": 380, "Tropical Budget": 300}

    result = solve_pricing_milp(dp_table, daily_capacity, total_inventory)
    print("\nOptimization result sample:")
    print(result.head(10))
    print("\nTotal expected revenue:", result["Expected_Revenue"].sum())
