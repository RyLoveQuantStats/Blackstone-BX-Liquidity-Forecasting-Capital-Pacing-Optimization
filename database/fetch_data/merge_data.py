'''
Description: Script to merge stock_prices, kkr_statements, and macroeconomic_data tables 
into a master_data table for the period 2020-2024.
'''

import pandas as pd
import numpy as np
import os

from utils.logging_utils import setup_logging, log_info, log_error
from utils.db_utils import get_connection, DB_PATH

# Set up logging.
setup_logging()

# Define constants.
STOCK_TABLE = "stock_prices"
FINANCIALS_TABLE = "kkr_statements"
MACRO_TABLE = "macroeconomic_data"
OUTPUT_CSV = "output/master_data.csv"

def load_table(table_name, parse_dates=True):
    """Loads an entire table from the database."""
    try:
        conn = get_connection()
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=parse_dates)
        conn.close()
        log_info(f"Table '{table_name}' loaded successfully. Columns: {list(df.columns)}")
        return df
    except Exception as e:
        log_error(f"Error loading table '{table_name}': {e}")
        raise

def load_all_tables():
    """Loads stock, financials, and macro data from the DB."""
    df_stock = load_table(STOCK_TABLE)
    df_fin = load_table(FINANCIALS_TABLE)
    df_macro = load_table(MACRO_TABLE)
    
    # For the financials table, rename 'period_ending_balance' to 'Date'
    if "period_ending_balance" in df_fin.columns:
        df_fin.rename(columns={"period_ending_balance": "Date"}, inplace=True)
        log_info("Renamed 'period_ending_balance' to 'Date' in financials table.")
    
    # For stock and financials, if 'Date' isn't found but 'date' exists, rename it.
    if "Date" not in df_stock.columns and "date" in df_stock.columns:
        df_stock.rename(columns={"date": "Date"}, inplace=True)
    if "Date" not in df_fin.columns and "date" in df_fin.columns:
        df_fin.rename(columns={"date": "Date"}, inplace=True)
    
    # For macro, check for 'Date', 'date', or 'index'
    if "Date" not in df_macro.columns:
        if "date" in df_macro.columns:
            df_macro.rename(columns={"date": "Date"}, inplace=True)
        elif "index" in df_macro.columns:
            df_macro.rename(columns={"index": "Date"}, inplace=True)
        else:
            raise KeyError("Macro table does not have a date-like column (expected 'Date', 'date', or 'index').")
    
    log_info("All tables loaded. Stock columns: " + str(list(df_stock.columns)))
    log_info("Financials columns: " + str(list(df_fin.columns)))
    log_info("Macro columns: " + str(list(df_macro.columns)))
    return df_stock, df_fin, df_macro

def prepare_dataframe(df):
    """
    Prepares a DataFrame by:
      1. Dropping duplicate columns.
      2. Converting the 'Date' column to datetime,
      3. Setting it as the index, sorting by date,
      4. Removing duplicate dates.
    """
    # Drop duplicate columns (if any)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Convert 'Date' to datetime and set as index.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    log_info(f"Data prepared. New shape: {df.shape}")
    return df

def merge_all_data(df_stock, df_fin, df_macro):
    """
    Merges the three DataFrames on the date index.
    Uses a union of all dates and forward-fills missing data.
    Adds a prefix to each table's columns to prevent collisions.
    """
    full_index = df_stock.index.union(df_fin.index).union(df_macro.index)
    df_stock_full = df_stock.reindex(full_index).ffill().add_prefix("stock_")
    df_fin_full = df_fin.reindex(full_index).ffill().add_prefix("fin_")
    df_macro_full = df_macro.reindex(full_index).ffill().add_prefix("macro_")
    
    df_merged = pd.concat([df_stock_full, df_fin_full, df_macro_full], axis=1)
    log_info("Data merged successfully. Merged shape: " + str(df_merged.shape))
    return df_merged

def filter_dates(df, start_date="2020-01-01", end_date="2024-12-31"):
    """
    Filters the DataFrame to only include dates within the specified range.
    """
    df_filtered = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    log_info(f"Data filtered to dates {start_date} to {end_date}. New shape: {df_filtered.shape}")
    return df_filtered

def main():
    os.makedirs("output", exist_ok=True)
    log_info("Output directory created or already exists.")
    
    # 1. Load all tables.
    df_stock, df_fin, df_macro = load_all_tables()
    
    # 2. Prepare each DataFrame.
    df_stock = prepare_dataframe(df_stock)
    df_fin = prepare_dataframe(df_fin)
    df_macro = prepare_dataframe(df_macro)
    
    # 3. Log the column names for each DataFrame.
    log_info("Stock columns: " + str(list(df_stock.columns)))
    log_info("Financials columns: " + str(list(df_fin.columns)))
    log_info("Macro columns: " + str(list(df_macro.columns)))
    
    # 4. Merge all data on the Date column.
    df_merged = merge_all_data(df_stock, df_fin, df_macro)
    
    # 5. Filter data to the period 2020-2024.
    df_merged = filter_dates(df_merged, start_date="2020-01-01", end_date="2024-12-31")
    
    # 6. Store the final merged DataFrame as the master_data table and export to CSV.
    try:
        conn = get_connection()
        df_merged.to_sql("master_data", conn, if_exists="replace", index=True)
        conn.close()
        log_info("Master data table stored successfully in the database.")
    except Exception as e:
        log_error(f"Error storing master data: {e}")
        raise
    
    df_merged.to_csv(OUTPUT_CSV)
    log_info(f"Final merged data saved as CSV in '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    main()
