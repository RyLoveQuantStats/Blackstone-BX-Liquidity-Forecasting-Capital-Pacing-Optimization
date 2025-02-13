#!/usr/bin/env python3

"""
merge_and_clean_data.py

This script connects to the SQLite database, retrieves the tables:
    1) bx_stock_prices
    2) bx_financials
    3) macroeconomic_data

Then it:
- Detects which column has the date (e.g. Date, index, Unnamed: 0)
- Converts that column to datetime
- Resamples financials if necessary (quarterly -> daily)
- Merges everything into one "master" DataFrame
- Cleans missing values
- (Optional) Stores the final merged DataFrame back in the database
"""

import sqlite3
import pandas as pd
import os

DB_PATH = "database/blackstone_data.db"

def load_data_from_sql(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)

    df_stock = pd.read_sql("SELECT * FROM bx_stock_prices", conn)
    print("\n[DEBUG] bx_stock_prices columns:", df_stock.columns.tolist())
    print(df_stock.head(5))

    df_financials = pd.read_sql("SELECT * FROM bx_financials", conn)
    print("\n[DEBUG] bx_financials columns:", df_financials.columns.tolist())
    print(df_financials.head(5))

    df_macro = pd.read_sql("SELECT * FROM macroeconomic_data", conn)
    print("\n[DEBUG] macroeconomic_data columns:", df_macro.columns.tolist())
    print(df_macro.head(5))

    conn.close()
    return df_stock, df_financials, df_macro


def prepare_dataframe(df):
    """
    Converts the date-like column to datetime, sets as index, sorts, drops duplicates.
    Tries to detect possible date columns among ["Date", "date", "index", "Unnamed: 0", "level_0"].
    """
    possible_date_cols = ["Date", "date", "index", "Unnamed: 0", "level_0"]
    date_col = None
    
    # Detect which column is the date
    for col in possible_date_cols:
        if col in df.columns:
            date_col = col
            break
    
    if not date_col:
        raise ValueError(
            "No date-like column found in DataFrame. "
            "Columns are: " + ", ".join(df.columns)
        )
    
    # Rename that column to "Date"
    df.rename(columns={date_col: "Date"}, inplace=True)
    
    # Convert "Date" to datetime, set index
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.set_index("Date", inplace=True)
    
    # Sort, drop duplicates, drop rows with NaN in index
    df.sort_index(inplace=True)
    df.drop_duplicates(inplace=True)
    df = df[df.index.notnull()]

    
    return df

def resample_financials(financials_df, freq="D"):
    """
    If the financial statements are quarterly or annual, you can forward-fill
    them to daily frequency so it aligns with daily stock data.
    """
    financials_df.sort_index(inplace=True)
    df_resampled = financials_df.resample(freq).ffill()
    return df_resampled

def merge_data(df_stock, df_financials, df_macro):
    """
    Merge the three DataFrames into a single DataFrame on the date index.
    The approach here is:
      1) Stock is daily
      2) Financials might be quarterly -> resample to daily
      3) Macro might be daily or monthly
    """
    # 1) Resample financials to daily
    df_financials_daily = resample_financials(df_financials, freq="D")
    
    # 2) Merge stock + financials
    df_merged = df_stock.merge(
        df_financials_daily,
        how="left",
        left_index=True,
        right_index=True,
        suffixes=("_stock", "_fin")
    )
    
    # 3) Merge macro data
    # If macro is monthly or weekly, forward-fill to daily
    df_macro_daily = df_macro.resample("D").ffill()
    
    df_merged = df_merged.merge(
        df_macro_daily,
        how="left",
        left_index=True,
        right_index=True,
        suffixes=("", "_macro")
    )
    
    return df_merged

def clean_merged_data(df):
    """
    Handles missing values or outliers in the merged DataFrame.
    Here we do a simple forward-fill/back-fill, 
    but it depends on your data logic.
    """
    # Check the proportion of missing values in each column
    missing_summary = df.isnull().sum() / len(df) * 100
    print("\nMissing values (as % of total rows):\n", missing_summary)
    
    # Simple fill strategy:
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    
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
    # 1. Load data from SQLite
    df_stock, df_financials, df_macro = load_data_from_sql()
    
    # 2. Prepare each DataFrame
    df_stock = prepare_dataframe(df_stock)
    df_financials = prepare_dataframe(df_financials)
    df_macro = prepare_dataframe(df_macro)
    
    # 3. Merge
    df_merged = merge_data(df_stock, df_financials, df_macro)
    
    # 4. Clean merged data
    df_merged = clean_merged_data(df_merged)
    
    # 5. Store final merged DataFrame back in SQL (and optionally as CSV)
    store_merged_data(df_merged, table_name="bx_master_data")
    
    os.makedirs("output", exist_ok=True)
    csv_path = os.path.join("output", "bx_master_data.csv")
    df_merged.to_csv(csv_path)
    print(f"✅ Merged data also saved locally as '{csv_path}'")

if __name__ == "__main__":
    main()
