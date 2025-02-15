#!/usr/bin/env python3
"""
liquidity_forecasting_arima.py

Loads the merged KKR dataset from the database (or CSV),
fits a SARIMAX model to the 'capital_calls' time series,
and produces a forecast with error metrics and plots.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

DB_PATH = r"C:\Users\ryanl\OneDrive\Desktop\Programming Apps\Python\python_work\BX_Liquidity_Forecasting\database\blackstone_data.db"
print("[DEBUG] Final DB_PATH =", DB_PATH)

TABLE_NAME = "bx_master_data"

def load_master_data(db_path=DB_PATH, table=TABLE_NAME):
    print(f"[DEBUG] Using DB_PATH = {db_path}")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["Date"])
    conn.close()
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    # Attempt to infer frequency
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq:
        df.index.freq = inferred_freq
        print(f"[INFO] Inferred frequency: {inferred_freq}")
    else:
        print("[WARN] Could not infer frequency from Date index.")
    return df

def main():
    os.makedirs("plots", exist_ok=True)
    df = load_master_data()  # uses DB_PATH
    if "capital_calls" not in df.columns:
        raise ValueError("❌ 'capital_calls' column not found.")
    
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
    summary_text = results.summary().as_text()
    
    # Forecast
    n_forecast = len(test_df)
    forecast = results.forecast(steps=n_forecast)
    test_df["forecast"] = forecast
    
    mae = float(np.mean(np.abs(test_df["capital_calls"] - test_df["forecast"])))
    rmse = float(np.sqrt(np.mean((test_df["capital_calls"] - test_df["forecast"])**2)))
    
    # Save plot
    plt.figure(figsize=(10,5))
    plt.plot(train_df["capital_calls"], label="Train")
    plt.plot(test_df["capital_calls"], label="Test (Actual)")
    plt.plot(test_df["forecast"], label="Forecast", linestyle="--")
    plt.title("Capital Calls - ARIMA Forecast")
    plt.legend()
    plot_path = "plots/capital_calls_forecast.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Return results
    return {
        "summary": summary_text,
        "mae": mae,
        "rmse": rmse,
        "forecast": forecast.tolist(),
        "plot": plot_path
    }

def run():
    return main()

if __name__ == "__main__":
    print(run())
