# Travel Demand Forecasting & Inventory Optimisation Engine

A portfolio project that combines machine learning demand forecasting with linear programming-based inventory and pricing optimization for travel routes.

## What this project includes

### Part 1 — Demand Forecasting
- Synthetic weekly bookings for 5 routes over 3 years (156 weeks)
- Seasonality, lead-time effects, and route-level demand variation
- Two forecasting approaches:
  - **XGBoost** with engineered features
  - **SARIMA baseline** for comparison
- Evaluation on a held-out 20% test split using:
  - MAPE
  - RMSE
- Forecast vs actual visualization with matplotlib

### Part 2 — LP Optimization
- Uses XGBoost forecasts as optimization input
- PuLP model to maximize expected revenue
- Constraints:
  - Total seat capacity per period
  - Minimum 60% occupancy per route
  - Minimum 15% margin floor per seat
- Outputs optimal route-level price and seat allocation

### Part 3 — Packaging
Main API in `travel_engine.py`:
- `train(df)`
- `predict(model, df)`
- `optimise(forecasts, capacity, constraints)`

## Project files
- `data_generation.py` - synthetic data generation
- `demand_forecasting.py` - XGBoost and SARIMA model training/evaluation
- `optimization.py` - PuLP optimization model
- `travel_engine.py` - wrapper API functions
- `demo.ipynb` - notebook walkthrough
- `requirements.txt` - dependencies

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## Quick usage

```python
from data_generation import generate_synthetic_data
from travel_engine import train, predict, optimise, plot_evaluation

# 1) Generate data
df = generate_synthetic_data()

# 2) Train
artifacts = train(df)
print(artifacts.evaluation)

# 3) Predict
forecasts = predict(artifacts, df)
print(forecasts.head())

# 4) Optimize
allocation = optimise(
    forecasts,
    capacity=500,
    constraints={"min_occupancy": 0.60, "margin_floor": 0.15}
)
print(allocation)

# 5) Plot evaluation
plot_evaluation(artifacts)
```

## Notes
- The synthetic generator is deterministic with `random_state` for reproducibility.
- SARIMA is trained per route as a baseline comparator.
- LP optimization uses route-level latest forecast values for next-period pricing and allocation decisions.
