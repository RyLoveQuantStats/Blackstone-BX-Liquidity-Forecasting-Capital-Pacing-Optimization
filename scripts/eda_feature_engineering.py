#!/usr/bin/env python3

"""
eda_and_feature_engineering.py

Loads the merged 'master' dataset (bx_master_data) from the database or CSV,
performs basic Exploratory Data Analysis (EDA), and demonstrates 
feature engineering steps.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DB_PATH = "database/blackstone_data.db"
MASTER_TABLE = "bx_master_data"

def load_master_data_from_sql(db_path=DB_PATH, table=MASTER_TABLE):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()

    # Convert index column if it exists
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
    elif "index" in df.columns:
        df.rename(columns={"index": "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

    return df

def eda_plots(df):
    """
    A few EDA plots: time-series chart, histogram, correlation heatmap.
    Adjust columns as appropriate for your data.
    """
    # 1. Simple time-series plot of BX closing price
    if "Close" in df.columns:
        df["Close"].plot(figsize=(12,6), title="BX Close Price Over Time")
        plt.show()

    # 2. Distribution of daily returns (if present)
    if "Daily Return" in df.columns:
        df["Daily Return"].hist(bins=50)
        plt.title("Distribution of Daily Returns")
        plt.show()

    # 3. Correlation Heatmap (for numeric columns)
    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def feature_engineering(df):
    """
    Example of creating new features:
      - Lag features
      - Ratios (Debt/Assets)
      - Growth rates
    Adjust to fit your data context.
    """
    # Example: Create a lag of 10Y Treasury yield if it exists
    if "10Y Treasury Yield" in df.columns:
        df["10Y_Treasury_Lag30"] = df["10Y Treasury Yield"].shift(30)

    # Example: create a financial ratio if totalLiab and totalAssets exist
    if "totalLiab" in df.columns and "totalAssets" in df.columns:
        # Avoid division by zero
        df["Debt_to_Assets"] = df.apply(
            lambda row: (row["totalLiab"] / row["totalAssets"]) 
            if row["totalAssets"] != 0 
            else 0, 
            axis=1
        )

    # Example: create a moving average of close price
    if "Close" in df.columns:
        df["MA_20"] = df["Close"].rolling(20).mean()
        df["MA_50"] = df["Close"].rolling(50).mean()

    # Fill any new NaN introduced by rolling/shift
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    return df

def main():
    # 1. Load the master dataset
    df = load_master_data_from_sql()

    print("Master dataset loaded. Shape:", df.shape)
    print("Columns:\n", df.columns)

    # 2. Basic EDA
    eda_plots(df)

    # 3. Create new features
    df = feature_engineering(df)

    # 4. (Optional) store updated DataFrame back to DB or to CSV
    # Example: store as new table "bx_master_features"
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("bx_master_features", conn, if_exists="replace")
    conn.close()

    os.makedirs("output", exist_ok=True)
    df.to_csv("output/bx_master_features.csv")
    print("✅ Feature-engineered data stored to DB and CSV.")

if __name__ == "__main__":
    main()
