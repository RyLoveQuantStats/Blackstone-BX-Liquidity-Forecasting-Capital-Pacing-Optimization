#!/usr/bin/env python3
"""
eda_and_feature_engineering.py

Loads the merged 'master' dataset (bx_master_data) from the database or CSV,
performs basic Exploratory Data Analysis (EDA), and demonstrates 
feature engineering steps.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import centralized DB and logging utilities.
from utils.db_utils import get_connection, DB_PATH
from utils.logging_utils import setup_logging, log_info, log_error

# Set up logging.
setup_logging()

MASTER_TABLE = "bx_master_data"

def load_master_data_from_sql(db_path=DB_PATH, table=MASTER_TABLE):
    try:
        conn = get_connection()
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        conn.close()
        
        # Convert index column if it exists.
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
        elif "index" in df.columns:
            df.rename(columns={"index": "Date"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
        
        log_info(f"Master data loaded from SQL table '{table}'. Shape: {df.shape}")
        return df
    except Exception as e:
        log_error(f"Error loading master data from SQL: {e}")
        raise

def eda_plots(df):
    """
    A few EDA plots: time-series chart, histogram, correlation heatmap.
    Adjust columns as appropriate for your data.
    """
    # 1. Simple time-series plot of BX closing price.
    if "Close" in df.columns:
        plt.figure(figsize=(12,6))
        df["Close"].plot(title="BX Close Price Over Time")
        plt.show()

    # 2. Distribution of daily returns (if present).
    if "Daily Return" in df.columns:
        plt.figure()
        df["Daily Return"].hist(bins=50)
        plt.title("Distribution of Daily Returns")
        plt.show()

    # 3. Correlation Heatmap (for numeric columns).
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if not numeric_cols.empty:
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
    # Example: Create a lag of 10Y Treasury yield if it exists.
    if "10Y Treasury Yield" in df.columns:
        df["10Y_Treasury_Lag30"] = df["10Y Treasury Yield"].shift(30)

    # Example: Create a financial ratio if totalLiab and totalAssets exist.
    if "totalLiab" in df.columns and "totalAssets" in df.columns:
        df["Debt_to_Assets"] = df.apply(
            lambda row: (row["totalLiab"] / row["totalAssets"]) if row["totalAssets"] != 0 else 0,
            axis=1
        )

    # Example: Create a moving average of the close price.
    if "Close" in df.columns:
        df["MA_20"] = df["Close"].rolling(20).mean()
        df["MA_50"] = df["Close"].rolling(50).mean()

    # Fill any new NaN values introduced by rolling/shift.
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    
    log_info("Feature engineering completed.")
    return df

def main():
    # 1. Load the master dataset.
    df = load_master_data_from_sql()

    log_info(f"Master dataset loaded. Shape: {df.shape}")
    log_info(f"Columns: {list(df.columns)}")

    # 2. Perform basic EDA.
    eda_plots(df)

    # 3. Create new features.
    df = feature_engineering(df)

    # 4. Store the updated DataFrame back to the database and to CSV.
    try:
        conn = get_connection()
        df.to_sql("bx_master_features", conn, if_exists="replace")
        conn.close()
        log_info("Feature-engineered data stored in DB table 'bx_master_features'.")
    except Exception as e:
        log_error(f"Error storing feature-engineered data to DB: {e}")

    os.makedirs("output", exist_ok=True)
    output_csv = "output/bx_master_features.csv"
    df.to_csv(output_csv)
    log_info(f"Feature-engineered data saved to CSV at '{output_csv}'.")

if __name__ == "__main__":
    main()
