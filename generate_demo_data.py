from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_chunk(rng: np.random.Generator, rows: int) -> pd.DataFrame:
    age = rng.integers(18, 76, rows)
    income = rng.normal(92_000, 28_000, rows).clip(22_000, 220_000)
    monthly_visits = rng.poisson(7, rows)
    avg_order_value = rng.gamma(6, 18, rows).clip(5, 800)
    support_tickets = rng.poisson(1.2, rows)
    contract_months = rng.integers(1, 72, rows)
    region = rng.choice(["Moscow", "North-West", "Siberia", "Ural", "South"], rows, p=[0.34, 0.2, 0.18, 0.16, 0.12])
    device = rng.choice(["mobile", "desktop", "tablet"], rows, p=[0.62, 0.3, 0.08])
    plan = rng.choice(["basic", "plus", "pro"], rows, p=[0.48, 0.37, 0.15])

    plan_risk = np.select([plan == "basic", plan == "plus", plan == "pro"], [0.48, 0.15, -0.18])
    device_risk = np.where(device == "mobile", 0.12, -0.04)
    logit = (
        -2.25
        + support_tickets * 0.34
        - contract_months * 0.025
        + monthly_visits * 0.018
        + plan_risk
        + device_risk
        - (income - 90_000) / 180_000
    )
    churn_probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, churn_probability)

    revenue_next_month = (
        avg_order_value * (1 + monthly_visits * 0.08)
        + income * 0.0018
        + np.where(plan == "pro", 95, np.where(plan == "plus", 45, 12))
        - churn * 70
        + rng.normal(0, 45, rows)
    ).clip(0)

    return pd.DataFrame(
        {
            "customer_age": age,
            "income": income.round(2),
            "monthly_visits": monthly_visits,
            "avg_order_value": avg_order_value.round(2),
            "support_tickets": support_tickets,
            "contract_months": contract_months,
            "region": region,
            "device": device,
            "plan": plan,
            "churn": churn,
            "revenue_next_month": revenue_next_month.round(2),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a large demo CSV for Big CSV Analyzer.")
    parser.add_argument("--rows", type=int, default=200_000, help="Number of rows to generate.")
    parser.add_argument("--chunk-size", type=int, default=50_000, help="Rows per generated chunk.")
    parser.add_argument("--output", default="data/demo_big_customers.csv", help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    written = 0
    first_chunk = True
    while written < args.rows:
        current_rows = min(args.chunk_size, args.rows - written)
        chunk = build_chunk(rng, current_rows)
        chunk.to_csv(output, mode="w" if first_chunk else "a", index=False, header=first_chunk)
        written += current_rows
        first_chunk = False
        print(f"written {written}/{args.rows}")

    print(f"saved to {output.resolve()}")


if __name__ == "__main__":
    main()
