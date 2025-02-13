#!/usr/bin/env python3

"""
merge_and_clean_data.py

This script connects to the SQLite database, retrieves the tables:
    1) bx_stock_prices  (Stock data)
    2) bx_financials    (Financial statements from FMP)
    3) macroeconomic_data (FRED or other macro data)

Then it:
- Detects & prepares the date columns
- Renames/combines certain columns in bx_financials into "capital_calls" and "distributions"
- Resamples if necessary (daily vs quarterly)
- Merges everything into one "master" DataFrame: bx_master_data
- Cleans missing values
- Stores the final merged DataFrame back in the database & CSV
"""

import sqlite3
import pandas as pd
import os

DB_PATH = "database/blackstone_data.db"

def load_data_from_sql(db_path=DB_PATH):
    """
    Connects to the SQLite database and loads the three relevant tables
    into pandas DataFrames.
    """
    conn = sqlite3.connect(db_path)
    df_stock = pd.read_sql("SELECT * FROM bx_stock_prices", conn)
    df_financials = pd.read_sql("SELECT * FROM bx_financials", conn)
    df_macro = pd.read_sql("SELECT * FROM macroeconomic_data", conn)
    conn.close()
    return df_stock, df_financials, df_macro

def prepare_date_column(df):
    """
    Detects/renames the date column to "Date", sets as datetime index.
    Handles common date-like columns: ["Date", "date", "index", "Unnamed: 0", ...]
    """
    possible_date_cols = ["Date", "date", "index", "Unnamed: 0", "level_0", "period_ending"]
    date_col = None

    for col in possible_date_cols:
        if col in df.columns:
            date_col = col
            break

    if not date_col:
        raise ValueError(
            "No date-like column found. "
            f"Columns: {', '.join(df.columns)}"
        )

    df.rename(columns={date_col: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df.drop_duplicates(inplace=True)
    # Remove rows where the index is NaT
    df = df[df.index.notnull()]

    return df

def resample_data(df, freq="D"):
    """
    If data is at a lower frequency (quarterly/annual),
    resample to daily for alignment with daily stock prices.
    We'll forward-fill to keep the last known value.
    """
    df.sort_index(inplace=True)
    df = df.resample(freq).ffill()
    return df

def merge_data(df_stock, df_financials, df_macro):
    """
    Merge the three DataFrames on the date index.
    1) Stock (daily)
    2) Financials (likely quarterly)
    3) Macro (could be daily or monthly)
    """
    # 1) Resample financials to daily
    df_financials_daily = resample_data(df_financials, freq="D")

    # 2) Merge stock + financials
    df_merged = df_stock.merge(
        df_financials_daily,
        how="left",
        left_index=True,
        right_index=True,
        suffixes=("_stock", "_fin")
    )

    # 3) Resample macro to daily (if monthly) and merge
    df_macro_daily = resample_data(df_macro, freq="D")
    df_merged = df_merged.merge(
        df_macro_daily,
        how="left",
        left_index=True,
        right_index=True,
        suffixes=("", "_macro")
    )

    return df_merged

def create_proxies_for_calls_and_distributions(df):
    """
    In the bx_financials table, we assume:
      - 'capital_calls' = absolute value of (dividendsPaid + repurchaseOfStock)
        (Because these are typically outflows of cash.)
      - 'distributions' = absolute value of (commonStockIssued)
        (Because this is typically an inflow of cash.)

    Adjust if your columns are named differently or if the sign is reversed.
    """
    # If these columns don't exist, fill them with 0 or skip
    # NOTE: These columns from FMP are often times 0 if no data is present
    if "dividendsPaid" not in df.columns:
        df["dividendsPaid"] = 0
    if "repurchaseOfStock" not in df.columns:
        df["repurchaseOfStock"] = 0
    if "commonStockIssued" not in df.columns:
        df["commonStockIssued"] = 0

    # Create capital_calls
    df["capital_calls"] = (
        df["dividendsPaid"].abs() + df["repurchaseOfStock"].abs()
    )

    # Create distributions
    df["distributions"] = (
        df["commonStockIssued"].abs()
    )

    return df

def clean_merged_data(df):
    """
    Handle missing values or outliers in the merged DataFrame.
    Simple forward-fill/back-fill or fill with 0 as needed.
    """
    # Example: fill any remaining NaNs with 0
    df.fillna(0, inplace=True)
    return df

def store_merged_data(df, table_name="bx_master_data", db_path=DB_PATH):
    """
    Stores the final merged (cleaned) DataFrame back into the database.
    """
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace")
    conn.close()
    print(f"✅ Merged data stored in '{table_name}' table within '{db_path}'")

def main():
    df_stock, df_financials, df_macro = load_data_from_sql()

    # Prepare date columns
    df_stock = prepare_date_column(df_stock)
    df_financials = prepare_date_column(df_financials)
    df_macro = prepare_date_column(df_macro)

    # Merge everything
    df_merged = merge_data(df_stock, df_financials, df_macro)

    # Create 'capital_calls' & 'distributions' from real financial statement items
    df_merged = create_proxies_for_calls_and_distributions(df_merged)

    # Clean
    df_merged = clean_merged_data(df_merged)

    # Store final
    store_merged_data(df_merged, table_name="bx_master_data")

    # Also store as CSV
    os.makedirs("output", exist_ok=True)
    csv_path = os.path.join("output", "bx_master_data.csv")
    df_merged.to_csv(csv_path)
    print(f"✅ Merged data also saved locally as '{csv_path}'")

if __name__ == "__main__":
    main()


