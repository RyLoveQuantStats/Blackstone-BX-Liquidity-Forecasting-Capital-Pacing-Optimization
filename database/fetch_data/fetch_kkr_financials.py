'''
Fetches the most recent KKR financial statements from OpenBB,
merges the data into one DataFrame, cleans the data, and stores it in the database.
'''

import pandas as pd
from openbb import obb
from utils.db_utils import store_dataframe  # Connects to the database and stores the DataFrame
from utils.logging_utils import setup_logging, log_info, log_error  # Logging utilities

def fetch_and_store_openbb_data():
    """
    Fetches KKR financial statements (balance sheet, income statement, cash flow statement)
    from OpenBB, merges the data into one DataFrame, cleans it, and stores it in the database.
    """
    ticker = "KKR"
    log_info(f"Fetching financial statements for {ticker} from OpenBB")
    
    try:
        # Fetch annual financial statements (you can change period="quarter" if needed)
        balance_df = obb.equity.fundamental.balance(
            ticker, provider="yfinance", period="annual", limit=5
        ).to_df()
        income_df = obb.equity.fundamental.income(
            ticker, provider="yfinance", period="annual", limit=5
        ).to_df()
        cash_df = obb.equity.fundamental.cash(
            ticker, provider="yfinance", period="annual", limit=5
        ).to_df()
    except Exception as e:
        log_error(f"Error fetching data from OpenBB for {ticker}: {e}")
        return

    log_info("Fetched balance sheet, income statement, and cash flow statement.")

    try:
        # Merge the DataFrames on their index (which should represent the reporting period/date)
        merged_df = balance_df.merge(
            income_df, left_index=True, right_index=True, how="outer", suffixes=("_balance", "_income")
        )
        merged_df = merged_df.merge(
            cash_df, left_index=True, right_index=True, how="outer", suffixes=("", "_cash")
        )
    except Exception as e:
        log_error(f"Error merging financial statements for {ticker}: {e}")
        return

    # Rename the index to "Date" (if not already named) and reset it as a column
    if merged_df.index.name is None:
        merged_df.index.name = "Date"
    merged_df = merged_df.reset_index()

    # Convert the "Date" column to datetime
    try:
        merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors="coerce")
    except Exception as e:
        log_error(f"Error converting Date column to datetime for {ticker}: {e}")

    # Fill missing numeric values with 0
    numeric_cols = merged_df.select_dtypes(include=["float64", "int64"]).columns
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)

    log_info("Merged financial statements:")
    log_info(merged_df.head().to_string())

    # Store the merged DataFrame in the database
    try:
        store_dataframe(merged_df, "kkr_statements", if_exists="append")
        log_info(f"Data for {ticker} stored successfully.")
    except Exception as e:
        log_error(f"Error storing data for {ticker}: {e}")

if __name__ == "__main__":
    setup_logging()
    fetch_and_store_openbb_data()
