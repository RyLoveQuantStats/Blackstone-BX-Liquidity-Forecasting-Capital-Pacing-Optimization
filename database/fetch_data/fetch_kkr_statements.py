"""
Script to fetch KKR financial statements from OpenBB,
merge them into one DataFrame, remove the "index" column,
use only "Date" as the date column, and store the final table in SQL.
"""

import pandas as pd
from openbb import obb
from utils.db_utils import get_connection, store_dataframe
from utils.logging_utils import setup_logging, log_info, log_error

setup_logging()

TICKER = "KKR"
FINANCIALS_TABLE = "kkr_statements"

def fetch_and_store_openbb_data():
    """
    1. Fetches annual financial statements from OpenBB for KKR (balance, income, cash).
    2. Merges them on their index (reporting period).
    3. Resets the merged index to a "Date" column.
    4. Drops any leftover "index" column if present.
    5. Filters out rows with dates before 2020-01-01.
    6. Fills missing numeric values with 0.
    7. Stores the final DataFrame into the 'kkr_statements' table without the DataFrame index.
    """
    log_info(f"Fetching financial statements for {TICKER} from OpenBB")
    try:
        # Fetch annual statements (adjust period or limit if needed)
        balance_df = obb.equity.fundamental.balance(
            TICKER, provider="yfinance", period="annual", limit=5
        ).to_df()
        income_df = obb.equity.fundamental.income(
            TICKER, provider="yfinance", period="annual", limit=5
        ).to_df()
        cash_df = obb.equity.fundamental.cash(
            TICKER, provider="yfinance", period="annual", limit=5
        ).to_df()
    except Exception as e:
        log_error(f"Error fetching data from OpenBB for {TICKER}: {e}")
        return

    log_info("Fetched balance sheet, income statement, and cash flow statement.")
    log_info(f"Balance sheet shape: {balance_df.shape}")
    log_info(f"Income statement shape: {income_df.shape}")
    log_info(f"Cash flow shape: {cash_df.shape}")

    try:
        # Merge the DataFrames on their index
        merged_df = balance_df.merge(
            income_df, left_index=True, right_index=True, how="outer", suffixes=("_balance", "_income")
        )
        merged_df = merged_df.merge(
            cash_df, left_index=True, right_index=True, how="outer", suffixes=("", "_cash")
        )
    except Exception as e:
        log_error(f"Error merging financial statements for {TICKER}: {e}")
        return

    # If the merged index has no name, give it one.
    if merged_df.index.name is None:
        merged_df.index.name = "Date"

    # Reset the index to become a "Date" column.
    merged_df = merged_df.reset_index()

    # If the DataFrame has an "index" column left over, drop it.
    # (Sometimes merging or older data might have introduced one.)
    if "index" in merged_df.columns:
        merged_df.drop(columns=["index"], inplace=True, errors="ignore")
        log_info("Dropped leftover 'index' column from the merged DataFrame.")

    # Convert "Date" column to datetime
    try:
        merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors="coerce")
    except Exception as e:
        log_error(f"Error converting 'Date' column for {TICKER}: {e}")

    # Filter out rows with dates before 2020-01-01
    merged_df = merged_df[merged_df["Date"] >= pd.Timestamp("2020-01-01")]
    log_info(f"Shape after filtering dates: {merged_df.shape}")

    # Fill missing numeric values with 0
    numeric_cols = merged_df.select_dtypes(include=["float64", "int64"]).columns
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)
    log_info("Final merged financial statements:")
    log_info(merged_df.head().to_string())

    # Store the final DataFrame in the database without the DataFrame index
    try:
        # 'index=False' ensures that the DataFrame index is NOT stored in SQL.
        # So the only date column is "Date".
        store_dataframe(merged_df, FINANCIALS_TABLE, if_exists="replace", index=False)
        log_info(f"Data for {TICKER} stored successfully in '{FINANCIALS_TABLE}'.")
    except Exception as e:
        log_error(f"Error storing data for {TICKER}: {e}")

def main():
    fetch_and_store_openbb_data()

if __name__ == "__main__":
    main()
