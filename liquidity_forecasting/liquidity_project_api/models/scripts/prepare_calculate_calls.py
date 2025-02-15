#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

# Import centralized DB and logging utilities.
from utils.db_utils import get_connection, DB_PATH
from utils.logging_utils import setup_logging, log_info, log_error

# Set up logging.
setup_logging()

OUTPUT_CSV = "output/master_data.csv"

def load_data_from_sql(db_path=DB_PATH):
    try:
        conn = get_connection()
        df_stock = pd.read_sql("SELECT * FROM stock_prices", conn)
        df_financials = pd.read_sql("SELECT * FROM financials", conn)
        df_macro = pd.read_sql("SELECT * FROM macroeconomic_data", conn)
        conn.close()
        log_info("Data loaded successfully from SQL for stock, financials, and macroeconomic data.")
        return df_stock, df_financials, df_macro
    except Exception as e:
        log_error(f"Error loading data from SQL: {e}")
        raise

def prepare_date_column(df):
    """Prepares a proper Date index from whichever column is recognized as date-like."""
    possible_date_cols = ["Date", "date", "index", "Unnamed: 0", "level_0", "period_ending"]
    date_col = next((col for col in possible_date_cols if col in df.columns), None)
    if not date_col:
        raise ValueError("No date-like column found.")
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.set_index("Date").sort_index().drop_duplicates()
    df = df[df.index.notnull()]
    log_info("Date column prepared and set as index.")
    return df

def resample_data(df, freq="D"):
    """Resamples the data to daily frequency and forward-fills missing days."""
    df = df.sort_index()
    resampled_df = df.resample(freq).ffill()
    log_info(f"Data resampled to {freq} frequency.")
    return resampled_df

def merge_data(df_stock, df_financials, df_macro):
    """
    Merges stock prices, financials, and macro data on a daily basis.
    """
    df_financials_daily = resample_data(df_financials, freq="D")
    df_merged = df_stock.merge(df_financials_daily, how="left", left_index=True, right_index=True)

    df_macro_daily = resample_data(df_macro, freq="D")
    df_merged = df_merged.merge(df_macro_daily, how="left", left_index=True, right_index=True)
    log_info("Data merged successfully.")
    return df_merged

def compute_yoy_inflation(df):
    """
    Computes a year-over-year inflation rate from monthly CPI data.
    - First, resamples CPI to monthly means.
    - Then computes yoy = (CPI[t] - CPI[t-12]) / CPI[t-12].
    - Caps extremes.
    Returns a single yoy_inflation value.
    """
    if "CPI" not in df.columns:
        log_info("CPI column not found. Defaulting yoy_inflation to 0.0")
        return 0.0  # no CPI column, fallback

    cpi_monthly = df["CPI"].resample("M").mean().dropna()
    if len(cpi_monthly) < 13:
        log_info("Not enough CPI data for YoY calculation. Defaulting yoy_inflation to 0.0")
        return 0.0

    yoy_series = cpi_monthly.pct_change(periods=12)
    yoy_series.loc[yoy_series > 1.0] = 1.0    # cap at 100% YoY
    yoy_series.loc[yoy_series < -0.5] = -0.5  # floor at -50% YoY
    yoy_series.fillna(0, inplace=True)

    yoy_inflation = yoy_series.iloc[-1]
    log_info(f"Computed YoY inflation: {yoy_inflation:.4f}")
    return yoy_inflation

def calculate_pe_capital_calls(df, base_rate=0.05):
    """
    Example PE approach using totalInvestments, netDebt, operatingCashFlow, retainedEarnings,
    along with:
      - YoY inflation from monthly CPI.
      - Mean interest_rate from 10Y Treasury.
      - Volatility from Volatility_30.
    """
    needed_cols = [
        "totalInvestments", "netDebt", "operatingCashFlow", "retainedEarnings",
        "CPI", "10Y Treasury Yield", "Volatility_30"
    ]
    # Ensure all needed columns exist and fill missing values.
    for col in needed_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    yoy_inflation = compute_yoy_inflation(df)
    interest_rate = df["10Y Treasury Yield"].mean()  # e.g., average over entire period
    vol_series = df["Volatility_30"].fillna(0)
    vol_factor = vol_series.mean() / 100.0
    adjusted_call_rate = base_rate * (1 + yoy_inflation) * (1 + vol_factor) * (1 + interest_rate / 100)

    df["capital_calls"] = ((df["totalInvestments"] * 0.02) + (df["netDebt"] * 0.01)) * adjusted_call_rate
    df["distributions"] = ((df["operatingCashFlow"] * 0.05) + (df["retainedEarnings"] * 0.01)) * (1 + interest_rate / 100)

    log_info("[INFO] yoy_inflation={:.4f}, interest_rate={:.2f}, vol_factor={:.4f}, call_rate={:.4f}".format(
        yoy_inflation, interest_rate, vol_factor, adjusted_call_rate
    ))
    return df

def clean_merged_data(df):
    cleaned_df = df.fillna(0)
    log_info("Merged data cleaned (NaNs filled with 0).")
    return cleaned_df

def store_merged_data(df, table_name="master_data", db_path=DB_PATH):
    try:
        conn = get_connection()
        df.to_sql(table_name, conn, if_exists="replace")
        conn.close()
        log_info(f"Merged data stored in '{table_name}' in '{db_path}'.")
    except Exception as e:
        log_error(f"Error storing merged data: {e}")
        raise

def main():
    os.makedirs("output", exist_ok=True)
    log_info("Output directory created or already exists.")

    # 1. Load data
    df_stock, df_financials, df_macro = load_data_from_sql()
    df_stock = prepare_date_column(df_stock)
    df_financials = prepare_date_column(df_financials)
    df_macro = prepare_date_column(df_macro)

    # 2. Merge data
    df_merged = merge_data(df_stock, df_financials, df_macro)

    # 3. Calculate capital calls using YoY inflation
    df_merged = calculate_pe_capital_calls(df_merged)

    # 4. Final cleaning
    df_merged = clean_merged_data(df_merged)

    # 5. Store merged data in the database and CSV file
    store_merged_data(df_merged, table_name="master_data")
    df_merged.to_csv(OUTPUT_CSV)
    log_info(f"Final data saved as CSV in '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    main()
