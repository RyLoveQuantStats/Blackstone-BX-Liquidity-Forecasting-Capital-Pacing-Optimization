#!/usr/bin/env python3

"""
merge_and_clean_data.py

Loads:
  - bx_stock_prices
  - bx_financials
  - macroeconomic_data

Merges them into one daily DataFrame. Then attempts two methods to create
'capital_calls' and 'distributions':

Method A:
  capital_calls = abs(dividendsPaid) + abs(commonStockRepurchased)
  distributions = abs(commonStockIssued)

Method B (fallback if Method A yields all zero):
  capital_calls   = max(0, -netCashUsedProvidedByFinancingActivities)
  distributions   = max(0,  netCashUsedProvidedByFinancingActivities)

Finally stores to 'bx_master_data' table + CSV.
"""

import sqlite3
import pandas as pd
import os

DB_PATH = "database/blackstone_data.db"

def load_data_from_sql(db_path=DB_PATH):
    """Load the three relevant tables from SQLite."""
    conn = sqlite3.connect(db_path)
    df_stock = pd.read_sql("SELECT * FROM bx_stock_prices", conn)
    df_financials = pd.read_sql("SELECT * FROM bx_financials", conn)
    df_macro = pd.read_sql("SELECT * FROM macroeconomic_data", conn)
    conn.close()
    return df_stock, df_financials, df_macro

def prepare_date_column(df):
    """
    Detect/rename the date column to "Date", set as datetime index.
    Remove rows with NaT.
    """
    possible_date_cols = [
        "Date", "date", "index", "Unnamed: 0", "level_0", "period_ending"
    ]
    date_col = None
    for col in possible_date_cols:
        if col in df.columns:
            date_col = col
            break

    if not date_col:
        raise ValueError("No date-like column found. Columns: " + ", ".join(df.columns))

    df.rename(columns={date_col: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df.drop_duplicates(inplace=True)
    df = df[df.index.notnull()]
    return df

def resample_data(df, freq="D"):
    """Resample to daily freq (forward-fill) to align with daily stock data."""
    df.sort_index(inplace=True)
    return df.resample(freq).ffill()

def merge_data(df_stock, df_financials, df_macro):
    """
    1) Resample financials to daily
    2) Merge stock + financials
    3) Resample macro to daily, merge
    """
    df_financials_daily = resample_data(df_financials, freq="D")
    df_merged = df_stock.merge(
        df_financials_daily, how="left",
        left_index=True, right_index=True,
        suffixes=("_stock", "_fin")
    )
    df_macro_daily = resample_data(df_macro, freq="D")
    df_merged = df_merged.merge(
        df_macro_daily, how="left",
        left_index=True, right_index=True,
        suffixes=("", "_macro")
    )
    return df_merged

def method_a_classic_shareholder_outflows(df):
    """
    capital_calls = abs(dividendsPaid) + abs(commonStockRepurchased)
    distributions = abs(commonStockIssued)
    """
    # Ensure these columns exist, fill with 0
    for col in ["dividendsPaid", "commonStockRepurchased", "commonStockIssued"]:
        if col not in df.columns:
            df[col] = 0
        df[col].fillna(0, inplace=True)

    # Drop old columns
    for col in ["capital_calls", "distributions"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Compute
    df["capital_calls"] = (
        df["dividendsPaid"].abs() + df["commonStockRepurchased"].abs()
    )
    df["distributions"] = df["commonStockIssued"].abs()

    return df

def method_b_net_financing_flows(df):
    """
    If Method A yields all zeros, fallback to netCashUsedProvidedByFinancingActivities:
      capital_calls   = max(0, -finFlow)
      distributions   = max(0, finFlow)
    """
    if "netCashUsedProvidedByFinancingActivities" not in df.columns:
        # If it's not there, fallback or set everything 0
        df["capital_calls"] = 0
        df["distributions"] = 0
        return df

    df["netCashUsedProvidedByFinancingActivities"].fillna(0, inplace=True)

    # Overwrite old columns
    for col in ["capital_calls", "distributions"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Build from netCashUsedProvidedByFinancingActivities
    df["capital_calls"] = df["netCashUsedProvidedByFinancingActivities"].apply(
        lambda x: max(0, -x)
    )
    df["distributions"] = df["netCashUsedProvidedByFinancingActivities"].apply(
        lambda x: max(0, x)
    )
    return df

def create_capital_calls_distributions(df):
    """
    1) Try Method A
    2) If it yields all zeros for capital_calls, fallback to Method B
    """
    df = method_a_classic_shareholder_outflows(df)

    # Check if capital_calls is all zero
    if (df["capital_calls"].sum() == 0):
        print("[INFO] Method A yields zero. Using Method B (netFinFlow).")
        df = method_b_net_financing_flows(df)
    else:
        print("[INFO] Using Method A (div+repurchase). Non-zero capital_calls found.")

    return df

def clean_merged_data(df):
    """Simple fill of any remaining NaN with 0, etc."""
    df.fillna(0, inplace=True)
    return df

def store_merged_data(df, table_name="bx_master_data", db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace")
    conn.close()
    print(f"✅ Merged data stored in '{table_name}' in '{db_path}'")

def main():
    df_stock, df_financials, df_macro = load_data_from_sql()

    df_stock = prepare_date_column(df_stock)
    df_financials = prepare_date_column(df_financials)
    df_macro = prepare_date_column(df_macro)

    # Merge
    df_merged = merge_data(df_stock, df_financials, df_macro)

    # Create capital_calls/distributions via 2-step approach
    df_merged = create_capital_calls_distributions(df_merged)

    # Clean
    df_merged = clean_merged_data(df_merged)

    # Store final
    store_merged_data(df_merged, table_name="bx_master_data")

    # Save CSV
    os.makedirs("output", exist_ok=True)
    csv_path = os.path.join("output", "bx_master_data.csv")
    df_merged.to_csv(csv_path)
    print(f"✅ Final data also saved as '{csv_path}'")

if __name__ == "__main__":
    main()
