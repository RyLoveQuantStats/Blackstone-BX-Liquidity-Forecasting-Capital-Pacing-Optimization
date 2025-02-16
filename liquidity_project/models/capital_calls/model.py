"""
Enhanced Capital Calls & Distributions Update Script with Diagnostics
-----------------------------------------------------------------------
This script updates your master_data CSV from the database, calculates enhanced capital calls
and distributions, and exports the results. It now includes diagnostic logging and plotting to
QC the input data and intermediate calculations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from utils.logging_utils import setup_logging, log_info, log_error
from utils.db_utils import get_connection, DB_PATH

# Set up logging.
setup_logging()

# Save Locations
MASTER_DATA_CSV = "output/csv/master_data.csv"
OUTPUT_ANALYSIS_CSV = "output/csv/master_data_with_capital_calls.csv"
TABLE_NAME = "master_data"

# Flag to enable or disable outlier removal.
REMOVE_OUTLIERS = False  # Set to False if outlier removal is making data constant.

# --------------------------------------------------------------------------
# 1. Data Loading Functions
# --------------------------------------------------------------------------

def load_master_data_from_db(db_path=DB_PATH, table=TABLE_NAME):
    try:
        conn = get_connection()
        df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["Date"])
        conn.close()
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        log_info(f"Loaded {df.shape[0]} rows from the database.")
        return df
    except Exception as e:
        log_error(f"Error loading data from DB: {e}")
        raise

def update_master_data_csv(csv_path=MASTER_DATA_CSV, db_path=DB_PATH, table=TABLE_NAME):
    if os.path.exists(csv_path):
        log_info(f"CSV '{csv_path}' exists. Loading existing data.")
        existing_df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        last_date = existing_df.index.max()
        log_info(f"Last date in CSV: {last_date}")
        try:
            conn = get_connection()
            query = f"SELECT * FROM {table} WHERE Date > '{last_date}'"
            new_df = pd.read_sql(query, conn, parse_dates=["Date"])
            conn.close()
            if not new_df.empty:
                new_df.set_index("Date", inplace=True)
                log_info(f"Found {new_df.shape[0]} new rows.")
                updated_df = pd.concat([existing_df, new_df]).drop_duplicates().sort_index()
            else:
                log_info("No new rows found.")
                updated_df = existing_df
        except Exception as e:
            log_error(f"Error updating CSV from DB: {e}")
            updated_df = existing_df
    else:
        log_info(f"CSV '{csv_path}' not found. Loading full data from DB.")
        updated_df = load_master_data_from_db(db_path, table)
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    updated_df.to_csv(csv_path)
    log_info(f"Updated master data saved to '{csv_path}'.")
    return updated_df

# --------------------------------------------------------------------------
# 2. Data QC Functions
# --------------------------------------------------------------------------

def qc_print_stats(df, cols):
    """Print summary statistics for key columns."""
    for col in cols:
        if col in df.columns:
            log_info(f"Stats for {col}:\n{df[col].describe()}")
        else:
            log_info(f"Column {col} not found in data.")

def plot_time_series(df, col, output_dir="output/qc_plots"):
    """Plot time series for a given column."""
    os.makedirs(output_dir, exist_ok=True)
    if col in df.columns:
        plt.figure(figsize=(10,4))
        plt.plot(df.index, df[col])
        plt.title(f"Time Series of {col}")
        save_path = os.path.join(output_dir, f"{col}_timeseries.png")
        plt.savefig(save_path)
        plt.close()
        log_info(f"Saved time series plot for {col} to {save_path}")

# --------------------------------------------------------------------------
# 3. New Capital Call Calculation Logic
# --------------------------------------------------------------------------

def calculate_capital_calls_new(df):
    """
    New logic for capital call estimation:
      - Investment Needs: Absolute value of negative investing cash flow.
      - Debt Repayment: When net debt decreases (delta < 0).
      - Available Funds: Operating cash flow + Cash and cash equivalents.
      - Capital Calls: max(0, Investment Needs + Debt Repayment - Available Funds)
    """
    # Ensure necessary columns are present and numeric.
    required_cols = ['fin_investing_cash_flow', 'fin_operating_cash_flow', 
                     'fin_cash_and_cash_equivalents', 'fin_net_debt']
    for col in required_cols:
        if col not in df.columns:
            log_info(f"Column '{col}' missing; setting to 0.")
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calculate investment needs: Only consider negative cash flows (outflows).
    df['investment_needs'] = df['fin_investing_cash_flow'].apply(lambda x: abs(x) if x < 0 else 0)
    
    # Calculate debt repayment: If net debt decreases (delta negative), treat that as repayment.
    df['delta_net_debt'] = df['fin_net_debt'].diff().fillna(0)
    df['debt_repayment'] = df['delta_net_debt'].apply(lambda x: abs(x) if x < 0 else 0)
    
    # Available funds from operations and current cash.
    df['available_funds'] = df['fin_operating_cash_flow'] + df['fin_cash_and_cash_equivalents']
    
    # Calculate capital call shortfall.
    df['capital_call_shortfall'] = df['investment_needs'] + df['debt_repayment'] - df['available_funds']
    df['capital_calls_new'] = df['capital_call_shortfall'].apply(lambda x: x if x > 0 else 0)
    
    log_info("New capital call calculation complete. Descriptive stats:\n" +
             str(df['capital_calls_new'].describe()))
    return df

# --------------------------------------------------------------------------
# 4. Existing Distributions Calculation (unchanged)
# --------------------------------------------------------------------------

def calculate_distributions_enhanced(df, window=7, growth_weight=0.5):
    for col in ['fin_operating_cash_flow', 'fin_retained_earnings']:
        if col not in df.columns:
            log_info(f"Column '{col}' missing; setting to 0.")
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df["fin_base_distributions"] = df["fin_operating_cash_flow"] + df["fin_retained_earnings"]
    df["fin_ocf_growth"] = df["fin_operating_cash_flow"].pct_change(periods=window).fillna(0)
    if "macro_10Y Treasury Yield" in df.columns:
        dist_factor = 1 + df["macro_10Y Treasury Yield"].mean()/100
    else:
        dist_factor = 1.0
    overall_adj = (1 + growth_weight * df["fin_ocf_growth"]) * dist_factor
    df["distributions"] = df["fin_base_distributions"] * overall_adj
    log_info("Enhanced distributions computed. Descriptive stats:\n" +
             str(df["distributions"].describe()))
    return df

# --------------------------------------------------------------------------
# 5. Storage & Export
# --------------------------------------------------------------------------

def store_master_data(df, db_path=DB_PATH, table=TABLE_NAME):
    try:
        conn = get_connection()
        df.to_sql(table, conn, if_exists="replace", index=True)
        conn.close()
        log_info("Updated master data stored in the database.")
    except Exception as e:
        log_error(f"Error storing master data: {e}")
        raise

# --------------------------------------------------------------------------
# 6. Main Routine
# --------------------------------------------------------------------------

def main():
    os.makedirs("output", exist_ok=True)
    # Update or load CSV data.
    df_master = update_master_data_csv(csv_path=MASTER_DATA_CSV, db_path=DB_PATH, table=TABLE_NAME)
    
    # QC: Print summary stats and plot key columns to verify variability.
    key_cols = ['fin_investing_cash_flow', 'fin_operating_cash_flow', 'fin_cash_and_cash_equivalents', 
                'fin_net_debt', 'macro_inflation_rate', 'stock_Volatility_30']
    qc_print_stats(df_master, key_cols)
    for col in key_cols:
        plot_time_series(df_master, col, output_dir="output/qc_plots")
    
    # New Capital Call Calculation using the updated logic.
    df_master = calculate_capital_calls_new(df_master)
    
    # Enhanced distributions calculation.
    df_master = calculate_distributions_enhanced(df_master, window=7, growth_weight=0.5)
    
    # Store updated data.
    store_master_data(df_master, db_path=DB_PATH, table=TABLE_NAME)
    df_master.to_csv(OUTPUT_ANALYSIS_CSV)
    log_info(f"Enhanced master data exported to '{OUTPUT_ANALYSIS_CSV}'.")
    
    return df_master

def run():
    return main()

if __name__ == "__main__":
    updated_df = run()
    print(updated_df.head())
