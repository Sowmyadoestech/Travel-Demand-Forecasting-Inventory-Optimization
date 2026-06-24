# Travel Package Demand Forecasting & Revenue Optimization Engine

A prototype pipeline that forecasts 14-day demand for travel packages and
solves for the revenue-maximizing price/inventory allocation, subject to
capacity constraints.

## Structure

```
travel_rm/
├── data_simulation.py      # Synthetic 3-year daily history, 3 packages
├── feature_engineering.py  # Calendar, lag, rolling, price features
├── forecasting.py          # XGBoost model, evaluation, recursive 14-day forecast
├── optimization.py         # Elasticity estimation + MILP pricing optimizer (PuLP)
├── main.py                 # End-to-end execution script
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
python main.py
```

Outputs:
- `synthetic_travel_demand.csv` — the simulated history
- `forecast_pricing_summary.csv` — final decision-support table (Date | Package |
  Forecasted_Demand | Optimal_Price | Bookings_At_Optimal_Price | Expected_Revenue |
  Inventory_Utilized)
- `chart_demand_forecast.png` — actuals vs. 14-day forecast, per package
- `chart_price_revenue.png` — optimized daily price & expected revenue, per package

Each module also runs standalone (`python data_simulation.py`,
`python forecasting.py`, `python optimization.py`) for isolated testing.

## How the pieces fit together

1. **Forecasting predicts baseline demand**, i.e. "what would bookings look
   like 14 days out if pricing simply continued at its recent level." It does
   *not* re-run per candidate price — that would mean training/serving N
   models for N price points, which doesn't scale.
2. **Elasticity estimation** (a log-log regression controlling for
   seasonality, holidays, weekends, and trend) gives each package a single
   number: the % demand response to a 1% price change.
3. **The optimizer** uses elasticity to translate baseline demand into a
   demand curve across discrete price tiers, then picks the revenue-maximizing
   tier per day per package subject to daily capacity and total-allotment
   constraints — a 0/1 MILP solved with PuLP/CBC.

This split (ML forecasts quantity at a reference price; econometrics supplies
the price-response curve; optimization searches price/allocation space) is
deliberate — it avoids needing a demand model that is simultaneously a great
time-series forecaster *and* a well-identified causal price-response model,
which is a much harder combined ask of a single model.

## Scaling to production

**Retraining cadence.** Retrain the demand model on a rolling schedule (e.g.
weekly) using all bookings through "yesterday," with the elasticity
regression refreshed on the same cadence — elasticity is more stable than
day-to-day demand, so even monthly re-estimation is often sufficient, but it
should never go stale across a full season change. Forecasts and the pricing
MILP should re-run daily as the 14-day window rolls forward, since each new
day of confirmed bookings changes the lag/rolling features in the next
forecast and the remaining inventory in the next optimization. Champion/
challenger evaluation (holding out the most recent N days, exactly as in
`time_based_train_test_split`) should gate any model promotion — never push a
retrained model straight to the pricing engine without a holdout MAE/RMSE
check, since the optimizer will faithfully act on a worse forecast.

**Handling optimization infeasibility.** The `TotalInventory` constraint is
the one most likely to make the MILP infeasible — it happens when forecasted
demand at the *highest allowed price* would still exceed the remaining
allotment. Three practical responses, roughly in order of sophistication:
  - *Detect and relax*: catch the solver's non-optimal status (as
    `solve_pricing_milp` does) and retry with a relaxed cap, a wider price
    band, or by flagging the package for manual allotment renegotiation
    (e.g. buying more charter seats or hotel rooms).
  - *Soft constraints*: replace the hard total-inventory constraint with a
    slack variable that allows exceeding inventory at a steep penalty cost in
    the objective (representing rebooking/overbooking penalties) — this keeps
    the model always solvable and lets the penalty cost express how
    undesirable an overbook actually is, rather than a binary
    feasible/infeasible.
  - *Waitlisting*: treat demand above the allotment as a waitlist rather than
    lost revenue, and feed waitlist size back into the next allotment
    negotiation cycle.

**Other production considerations.**
  - *Price-change guardrails*: add a constraint limiting day-over-day price
    movement (e.g. no more than one tier change per day) so the recommended
    price path doesn't whiplash, which both confuses customers checking
    prices repeatedly and looks erratic to the commercial team approving it.
  - *Monitoring*: track forecast bias (not just MAE) per package over time —
    a model that is, say, consistently 10% low during a holiday week is a
    systematic miss the RM team needs to know about, not just noise.
  - *A/B holdout*: route a small % of inventory to a non-optimized control
    price to keep measuring true elasticity going forward, since once the
    optimizer is in control of pricing, organic price/demand variation
    (which the elasticity regression relies on) shrinks — a classic
    "operating the system changes the data the system was trained on"
    feedback loop.
  - *Solver scale*: this prototype's MILP (3 packages × 14 days × 5 tiers =
    210 binary variables) solves in well under a second with CBC. A
    production deployment across hundreds of packages/markets would still be
    comfortably within reach of CBC or a commercial solver (Gurobi/CPLEX) —
    the variable count grows linearly in packages × horizon × tiers, not
    combinatorially, since each (package, day) tier-choice is independent
    except through the shared inventory constraint.
