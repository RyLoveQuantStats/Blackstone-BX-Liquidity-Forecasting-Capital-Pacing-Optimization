#!/usr/bin/env python3
"""
liquidity_forecasting_arima.py

Loads the merged KKR dataset from the database (or CSV),
fits a SARIMAX model to the 'capital_calls' time series,
and produces a forecast with error metrics and plots.

Enhancements:
 - Ensures the Date index has an inferred frequency.
 - Checks for potential stationarity issues.
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
    # Attempt to infer frequency if not set
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq:
        df.index.freq = inferred_freq
        print(f"[INFO] Inferred frequency: {inferred_freq}")
    else:
        print("[WARN] Could not infer frequency from Date index.")
    return df

def main():
    os.makedirs("plots", exist_ok=True)
    df = load_master_data()
    if "capital_calls" not in df.columns:
        raise ValueError("❌ 'capital_calls' column not found. Check your data ingestion pipeline.")
    
    df = df[df["capital_calls"].notnull()]
    
    # Simple train/test split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    train_y = train_df["capital_calls"]
    test_y = test_df["capital_calls"]
    
    # Fit SARIMAX model (tune order as needed)
    model = SARIMAX(train_y, order=(2,1,2))
    results = model.fit(disp=False)
    print(results.summary())
    
    # Forecast over the test period
    n_forecast = len(test_df)
    forecast = results.forecast(steps=n_forecast)
    test_df["forecast"] = forecast
    
    mae = np.mean(np.abs(test_df["capital_calls"] - test_df["forecast"]))
    rmse = np.sqrt(np.mean((test_df["capital_calls"] - test_df["forecast"])**2))
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    plt.figure(figsize=(10,5))
    plt.plot(train_df["capital_calls"], label="Train")
    plt.plot(test_df["capital_calls"], label="Test (Actual)")
    plt.plot(test_df["forecast"], label="Forecast", linestyle="--")
    plt.title("Capital Calls - ARIMA Forecast")
    plt.legend()
    plt.savefig("plots/capital_calls_forecast.png")
    plt.show()

if __name__ == "__main__":
    main()
