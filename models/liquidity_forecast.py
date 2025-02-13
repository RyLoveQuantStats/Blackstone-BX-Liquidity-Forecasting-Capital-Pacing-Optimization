#!/usr/bin/env python3

"""
liquidity_forecasting_arima.py

Loads the merged dataset (bx_master_data),
performs an ARIMA (or SARIMAX) forecast on 'capital_calls' (now derived from actual BX data).
"""

import sqlite3
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import os

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

    # We now rely on a real 'capital_calls' column from the merge script
    if "capital_calls" not in df.columns:
        raise ValueError("❌ 'capital_calls' column not found in bx_master_data. Did you rename or create it?")

    # Drop any rows with missing or invalid capital_calls
    df = df[df["capital_calls"].notnull()]

    # Train/Test Split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    train_y = train_df["capital_calls"]
    test_y = test_df["capital_calls"]

    # For demonstration, let's do a simple ARIMA(2,1,2)
    model = SARIMAX(train_y, order=(2,1,2))
    results = model.fit(disp=False)
    print(results.summary())

    # Forecast
    n_forecast = len(test_df)
    forecast = results.forecast(steps=n_forecast)

    # Evaluate
    test_df["forecast"] = forecast
    mae = np.mean(np.abs(test_df["capital_calls"] - test_df["forecast"]))
    rmse = np.sqrt(np.mean((test_df["capital_calls"] - test_df["forecast"])**2))
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(train_df["capital_calls"], label="Train")
    plt.plot(test_df["capital_calls"], label="Test (Actual)")
    plt.plot(test_df["forecast"], label="Forecast", linestyle="--")
    plt.title("Blackstone (BX) Capital Calls - ARIMA Forecast")
    plt.legend()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/bx_capital_calls_forecast.png")
    plt.show()

if __name__ == "__main__":
    main()
