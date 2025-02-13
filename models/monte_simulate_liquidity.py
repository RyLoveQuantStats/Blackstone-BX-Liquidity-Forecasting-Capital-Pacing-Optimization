#!/usr/bin/env python3

"""
simulate_liquidity_monte_carlo.py

Runs Monte Carlo simulations on the 'capital_calls' column from bx_master_data.
Estimates daily changes, draws random shocks, and calculates possible future states.
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "database/blackstone_data.db"
TABLE_NAME = "bx_master_data"

def load_master_data(db_path=DB_PATH, table=TABLE_NAME):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["Date"])
    conn.close()
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df

def main():
    df = load_master_data()

    if "capital_calls" not in df.columns:
        raise ValueError("❌ 'capital_calls' column not found. Did you rename or compute it?")

    df["calls_change"] = df["capital_calls"].diff().fillna(0)
    mean_change = df["calls_change"].mean()
    std_change = df["calls_change"].std()

    print(f"Mean daily change: {mean_change:.2f}, Std: {std_change:.2f}")

    # Monte Carlo parameters
    n_simulations = 1000
    horizon = 30  # e.g. 30 future periods
    last_value = df["capital_calls"].iloc[-1]

    outcomes = []
    for _ in range(n_simulations):
        shocks = np.random.normal(mean_change, std_change, horizon)
        future_values = [last_value]
        for shock in shocks:
            future_values.append(future_values[-1] + shock)
        outcomes.append(future_values[-1])

    outcomes = np.array(outcomes)
    p5 = np.percentile(outcomes, 5)
    p50 = np.percentile(outcomes, 50)
    p95 = np.percentile(outcomes, 95)

    print(f"5th percentile: {p5:.2f}")
    print(f"Median: {p50:.2f}")
    print(f"95th percentile: {p95:.2f}")

if __name__ == "__main__":
    main()
