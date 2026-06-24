"""
main.py
========
End-to-end execution script for the travel package demand forecasting +
revenue optimization prototype.

Pipeline
--------
1. Simulate (or load) 3 years of daily booking history for 3 packages.
2. Engineer features and train an XGBoost demand model; report holdout
   accuracy (MAE/RMSE) and feature importances.
3. Recursively forecast the next 14 days of demand per package.
4. Estimate price elasticity per package from history.
5. Build a discretized price-tier x demand table for the 14-day horizon.
6. Solve the revenue-maximizing MILP subject to daily capacity and total
   horizon inventory constraints.
7. Assemble and save the final decision-support summary table:
       Date | Package | Forecasted_Demand | Optimal_Price |
       Expected_Revenue | Inventory_Utilized
8. Render two diagnostic charts (forecast vs. history, optimized price &
   revenue by day) to PNG files.

Run with:  python main.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless rendering -> PNG files, no display needed
import matplotlib.pyplot as plt
import pandas as pd

from data_simulation import generate_synthetic_dataset, PACKAGE_CONFIG
from feature_engineering import build_feature_table
from forecasting import (
    time_based_train_test_split, train_demand_model, evaluate_model,
    get_feature_importance, recursive_forecast,
)
from optimization import (
    estimate_price_elasticity, build_demand_price_table, solve_pricing_milp,
)

# --------------------------------------------------------------------------
# Business configuration: capacity & inventory assumptions per package.
# In production these would come from contracted allotments (hotel block
# bookings, charter seat allocations) rather than being hardcoded.
# --------------------------------------------------------------------------
DAILY_CAPACITY = {
    "Mediterranean Luxury": 15,   # e.g. boutique resort, limited nightly check-ins
    "Alpine Adventure": 40,
    "Tropical Budget": 60,
}
TOTAL_INVENTORY_14D = {
    "Mediterranean Luxury": 75,   # fixed allotment contracted for the 2-week window
    "Alpine Adventure": 380,
    "Tropical Budget": 300,
}
N_PRICE_TIERS = 5
FORECAST_HORIZON_DAYS = 14
HOLDOUT_TEST_DAYS = 30


def run_pipeline():
    print("=" * 70)
    print("STEP 1/6: Simulating 3 years of synthetic daily booking history")
    print("=" * 70)
    history = generate_synthetic_dataset(start_date="2023-01-01", n_days=1095, seed=42)
    print(f"Generated {len(history):,} rows across {history['Package'].nunique()} packages "
          f"({history['Date'].min().date()} to {history['Date'].max().date()})")

    print("\n" + "=" * 70)
    print("STEP 2/6: Feature engineering + training the XGBoost demand model")
    print("=" * 70)
    feat_df, package_categories = build_feature_table(history)
    train_df, test_df = time_based_train_test_split(feat_df, test_days=HOLDOUT_TEST_DAYS)
    print(f"Train rows: {len(train_df):,} | Holdout test rows (last {HOLDOUT_TEST_DAYS} days): {len(test_df):,}")

    model = train_demand_model(train_df)
    metrics, eval_df = evaluate_model(model, test_df)

    print(f"\nHoldout accuracy -- Overall: MAE={metrics['overall']['MAE']:.2f}, "
          f"RMSE={metrics['overall']['RMSE']:.2f}")
    for pkg, m in metrics["by_package"].items():
        pct_mae = 100 * m["MAE"] / m["Avg_Actual"]
        print(f"  {pkg:<22s} MAE={m['MAE']:5.2f}  RMSE={m['RMSE']:5.2f}  "
              f"Avg actual={m['Avg_Actual']:5.1f}  (MAE = {pct_mae:.1f}% of avg demand)")

    importance_df = get_feature_importance(model)
    print("\nTop 10 feature importances:")
    print(importance_df.head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print(f"STEP 3/6: Recursively forecasting the next {FORECAST_HORIZON_DAYS} days")
    print("=" * 70)
    forecast_df = recursive_forecast(model, history, package_categories, horizon_days=FORECAST_HORIZON_DAYS)
    print(forecast_df.pivot(index="Date", columns="Package", values="Forecast_Bookings").to_string())

    print("\n" + "=" * 70)
    print("STEP 4/6: Estimating price elasticity per package from history")
    print("=" * 70)
    elasticity_df = estimate_price_elasticity(history)
    print(elasticity_df.to_string())

    print("\n" + "=" * 70)
    print("STEP 5/6: Building price tiers & solving the revenue-maximizing MILP")
    print("=" * 70)
    price_bounds = {pkg: (cfg["min_price"], cfg["max_price"]) for pkg, cfg in PACKAGE_CONFIG.items()}
    demand_price_df = build_demand_price_table(forecast_df, elasticity_df, price_bounds, n_tiers=N_PRICE_TIERS)

    try:
        opt_result = solve_pricing_milp(demand_price_df, DAILY_CAPACITY, TOTAL_INVENTORY_14D)
    except RuntimeError as e:
        print(f"\n[INFEASIBLE] {e}")
        print("Retrying with total inventory relaxed by +25% per package as a fallback policy...")
        relaxed_inventory = {pkg: int(v * 1.25) for pkg, v in TOTAL_INVENTORY_14D.items()}
        opt_result = solve_pricing_milp(demand_price_df, DAILY_CAPACITY, relaxed_inventory)

    print("\n" + "=" * 70)
    print("STEP 6/6: Assembling the final decision-support summary table")
    print("=" * 70)
    summary = build_summary_table(forecast_df, opt_result)
    print(summary.to_string(index=False))

    total_rev = summary["Expected_Revenue"].sum()
    print(f"\nTOTAL expected revenue across all packages, {FORECAST_HORIZON_DAYS} days: ${total_rev:,.2f}")

    # Persist outputs
    summary.to_csv("forecast_pricing_summary.csv", index=False)
    history.to_csv("synthetic_travel_demand.csv", index=False)
    print("\nSaved: forecast_pricing_summary.csv, synthetic_travel_demand.csv")

    render_charts(history, forecast_df, opt_result)
    print("Saved: chart_demand_forecast.png, chart_price_revenue.png")

    return {
        "history": history,
        "model": model,
        "metrics": metrics,
        "importance_df": importance_df,
        "forecast_df": forecast_df,
        "elasticity_df": elasticity_df,
        "demand_price_df": demand_price_df,
        "opt_result": opt_result,
        "summary": summary,
    }


def build_summary_table(forecast_df: pd.DataFrame, opt_result: pd.DataFrame) -> pd.DataFrame:
    """
    Joins the baseline forecast with the optimization decision into the
    final table the revenue management team consumes:

        Date | Package | Forecasted_Demand | Optimal_Price |
        Expected_Revenue | Inventory_Utilized

    - Forecasted_Demand: the ML model's baseline demand prediction (at the
      recently-observed reference price), i.e. "what would happen if we
      changed nothing".
    - Optimal_Price / Expected_Revenue: the MILP's chosen price tier and
      the resulting projected revenue at that price.
    - Inventory_Utilized: cumulative expected bookings consumed against
      each package's total 14-day allotment, as of that date (running
      total) -- lets the RM team see pacing against the allotment.
    """
    merged = forecast_df.merge(
        opt_result, on=["Date", "Package"], how="inner", suffixes=("_baseline", "")
    )
    merged = merged.rename(columns={"Forecast_Bookings": "Forecasted_Demand"})
    merged = merged.sort_values(["Package", "Date"])

    merged["Inventory_Utilized"] = (
        merged.groupby("Package")["Expected_Bookings"].cumsum().round(1)
    )

    cols = ["Date", "Package", "Forecasted_Demand", "Optimal_Price",
            "Expected_Bookings", "Expected_Revenue", "Inventory_Utilized"]
    out = merged[cols].rename(columns={"Expected_Bookings": "Bookings_At_Optimal_Price"})
    out["Date"] = out["Date"].dt.date
    return out.reset_index(drop=True)


def render_charts(history: pd.DataFrame, forecast_df: pd.DataFrame, opt_result: pd.DataFrame):
    """Saves two diagnostic PNG charts for quick visual sanity-checking."""
    packages = sorted(history["Package"].unique())
    colors = {"Mediterranean Luxury": "#1f77b4", "Alpine Adventure": "#2ca02c", "Tropical Budget": "#d62728"}

    # Chart 1: last 60 days of actual history + 14-day forecast, per package.
    fig, axes = plt.subplots(len(packages), 1, figsize=(11, 9), sharex=False)
    for ax, pkg in zip(axes, packages):
        hist_pkg = history[history["Package"] == pkg].tail(60)
        fc_pkg = forecast_df[forecast_df["Package"] == pkg]
        ax.plot(hist_pkg["Date"], hist_pkg["Bookings"], label="Actual (last 60 days)",
                color=colors.get(pkg, "gray"), linewidth=1.5)
        ax.plot(fc_pkg["Date"], fc_pkg["Forecast_Bookings"], label="14-day forecast",
                color=colors.get(pkg, "gray"), linestyle="--", marker="o", markersize=3)
        ax.axvline(hist_pkg["Date"].max(), color="black", linestyle=":", linewidth=0.8)
        ax.set_title(pkg, fontsize=11, fontweight="bold")
        ax.set_ylabel("Bookings")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle("Demand: Recent History vs. 14-Day Forecast", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("chart_demand_forecast.png", dpi=130)
    plt.close(fig)

    # Chart 2: optimized price + expected revenue over the 14-day horizon.
    fig, axes = plt.subplots(len(packages), 1, figsize=(11, 9), sharex=False)
    for ax, pkg in zip(axes, packages):
        g = opt_result[opt_result["Package"] == pkg].sort_values("Date")
        ax2 = ax.twinx()
        ax.bar(g["Date"], g["Expected_Revenue"], alpha=0.35, color=colors.get(pkg, "gray"), label="Expected Revenue")
        ax2.plot(g["Date"], g["Optimal_Price"], color="black", marker="o", markersize=3, label="Optimal Price")
        ax.set_title(pkg, fontsize=11, fontweight="bold")
        ax.set_ylabel("Expected Revenue ($)")
        ax2.set_ylabel("Optimal Price ($)")
        ax.grid(alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    fig.suptitle("Optimized Daily Price & Expected Revenue (14-Day Horizon)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("chart_price_revenue.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run_pipeline()
