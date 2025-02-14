#!/usr/bin/env python3

import sqlite3
import pandas as pd
import numpy as np
import os

DB_PATH = "database/blackstone_data.db"
OUTPUT_CSV = "output/bx_master_data.csv"

def load_data_from_sql(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df_stock = pd.read_sql("SELECT * FROM bx_stock_prices", conn)
    df_financials = pd.read_sql("SELECT * FROM bx_financials", conn)
    df_macro = pd.read_sql("SELECT * FROM macroeconomic_data", conn)
    conn.close()
    return df_stock, df_financials, df_macro

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
    return df

def resample_data(df, freq="D"):
    """Resamples the data to daily frequency and forward-fills missing days."""
    df = df.sort_index()
    return df.resample(freq).ffill()

def merge_data(df_stock, df_financials, df_macro):
    """
    Merges stock prices, financials, and macro data on a daily basis.
    You can confirm that forward-filling CPI daily doesn't create zeros
    by printing a few rows of df_macro before merging.
    """
    df_financials_daily = resample_data(df_financials, freq="D")
    df_merged = df_stock.merge(df_financials_daily, how="left", left_index=True, right_index=True)

    df_macro_daily = resample_data(df_macro, freq="D")
    df_merged = df_merged.merge(df_macro_daily, how="left", left_index=True, right_index=True)
    return df_merged

def compute_yoy_inflation(df):
    """
    Computes a year-over-year inflation rate from monthly CPI data.
    - First, resample CPI to monthly means
    - Then compute yoy = (CPI[t] - CPI[t-12]) / CPI[t-12]
    - Cap extremes
    Returns a single yoy_inflation value (mean or last).
    """
    if "CPI" not in df.columns:
        return 0.0  # no CPI column, fallback

    # Resample CPI to monthly
    cpi_monthly = df["CPI"].resample("M").mean().dropna()
    if len(cpi_monthly) < 13:
        # Not enough data for yoy (need at least 12 months)
        return 0.0

    # yoy series
    yoy_series = cpi_monthly.pct_change(periods=12)
    # Cap extremes
    yoy_series.loc[yoy_series > 1.0] = 1.0    # 100% yoy
    yoy_series.loc[yoy_series < -0.5] = -0.5  # -50% yoy
    yoy_series.fillna(0, inplace=True)

    # For example, take the most recent yoy or an average yoy
    yoy_inflation = yoy_series.iloc[-1]  # or yoy_series.mean()
    return yoy_inflation

def calculate_pe_capital_calls(df, base_rate=0.05):
    """
    Example PE approach using totalInvestments, netDebt, operatingCashFlow, retainedEarnings, plus:
      - yoy_inflation from monthly CPI
      - mean interest_rate from 10Y Treasury
      - volatility from Volatility_30
    """
    needed_cols = [
        "totalInvestments", "netDebt", "operatingCashFlow", "retainedEarnings",
        "CPI", "10Y Treasury Yield", "Volatility_30"
    ]
    # Ensure columns exist
    for col in needed_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    # 1) Compute yoy inflation from monthly data
    yoy_inflation = compute_yoy_inflation(df)

    # 2) interest_rate
    interest_rate = df["10Y Treasury Yield"].mean()  # e.g. average over entire period

    # 3) volatility factor
    # If Volatility_30 is zero or small, ensure no blow-up
    vol_series = df["Volatility_30"].fillna(0)
    vol_factor = vol_series.mean() / 100.0

    # 4) final call rate
    # e.g. base_rate * (1 + yoy_inflation) * ...
    adjusted_call_rate = base_rate * (1 + yoy_inflation) * (1 + vol_factor) * (1 + interest_rate / 100)

    df["capital_calls"] = ((df["totalInvestments"] * 0.02) + (df["netDebt"] * 0.01)) * adjusted_call_rate

    df["distributions"] = ((df["operatingCashFlow"] * 0.05) + (df["retainedEarnings"] * 0.01)) * (1 + interest_rate / 100)

    print("[INFO] yoy_inflation={:.4f}, interest_rate={:.2f}, vol_factor={:.4f}, call_rate={:.4f}".format(
        yoy_inflation, interest_rate, vol_factor, adjusted_call_rate
    ))
    return df

def clean_merged_data(df):
    return df.fillna(0)

def store_merged_data(df, table_name="bx_master_data", db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace")
    conn.close()
    print(f"✅ Merged data stored in '{table_name}' in '{db_path}'")

def main():
    os.makedirs("output", exist_ok=True)

    # 1. Load data
    df_stock, df_financials, df_macro = load_data_from_sql()
    df_stock = prepare_date_column(df_stock)
    df_financials = prepare_date_column(df_financials)
    df_macro = prepare_date_column(df_macro)

    # 2. Merge
    df_merged = merge_data(df_stock, df_financials, df_macro)

    # 3. Capital calls using yoy inflation
    df_merged = calculate_pe_capital_calls(df_merged)

    # 4. Final cleaning
    df_merged = clean_merged_data(df_merged)

    # 5. Store
    store_merged_data(df_merged, table_name="bx_master_data")
    df_merged.to_csv(OUTPUT_CSV)
    print(f"✅ Final data also saved as '{OUTPUT_CSV}'")

if __name__ == "__main__":
    main()
