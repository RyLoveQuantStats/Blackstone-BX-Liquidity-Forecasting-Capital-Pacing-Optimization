'''
Description: Advanced analysis script to diagnose and analyze capital calls and distributions
from the master_data table (merged from stock_prices, kkr_statements, and macroeconomic_data)
for the period 2020-2024. This script inspects the underlying financial components (such as capex,
working capital, net debt), computes descriptive statistics, performs yearly aggregation, and 
computes a correlation matrix. The updated analysis is saved back into the master_data table in the 
database and exported to CSV.
'''

import pandas as pd
import numpy as np
import os

from utils.logging_utils import setup_logging, log_info, log_error
from utils.db_utils import get_connection, DB_PATH

# Set up logging.
setup_logging()

ANALYSIS_OUTPUT = "output/pe_capital_calls_correlation_from_db.csv"

def load_master_data_from_db():
    """
    Loads the master_data table from the database, parsing the Date column as datetime and setting it as the index.
    """
    try:
        conn = get_connection()
        df = pd.read_sql("SELECT * FROM master_data", conn, parse_dates=["Date"], index_col="Date")
        conn.close()
        log_info(f"Master data loaded from database. Shape: {df.shape}")
        return df
    except Exception as e:
        log_error(f"Error loading master data from database: {e}")
        raise

def inspect_underlying_components(df):
    """
    Inspects the underlying financial components that contribute to capital calls.
    Logs unique values and descriptive statistics for:
      - fin_capital_expenditure
      - fin_current_assets
      - fin_current_liabilities
      - fin_net_debt
    """
    components = ["fin_capital_expenditure", "fin_current_assets", "fin_current_liabilities", "fin_net_debt"]
    for comp in components:
        if comp in df.columns:
            unique_vals = np.unique(df[comp].dropna())
            desc = df[comp].describe()
            log_info(f"{comp} descriptive statistics:\n{desc}")
            log_info(f"Unique values in {comp} (first 10): {unique_vals[:10]}")
            print(f"{comp} descriptive statistics:\n{desc}")
            print(f"Unique values in {comp} (first 10): {unique_vals[:10]}")
        else:
            log_info(f"{comp} NOT FOUND in master_data.")
            print(f"{comp} NOT FOUND in master_data.")

def inspect_nonzero_columns(df):
    """
    Logs the nonzero counts for key columns.
    """
    columns_to_check = [
        "capital_calls", "distributions",
        "netCashProvidedByOperatingActivities", "retainedEarnings",
        "longTermDebt", "shortTermDebt"
    ]
    log_info("Inspecting nonzero counts for key columns:")
    for col in columns_to_check:
        if col in df.columns:
            count = (df[col] != 0).sum()
            log_info(f"{col}: nonzero rows = {count} / {len(df)}")
            print(f"{col}: nonzero rows = {count} / {len(df)}")
        else:
            log_info(f"{col}: NOT FOUND in master_data.")
            print(f"{col}: NOT FOUND in master_data.")

def analyze_capital_calls(df):
    """
    Analyzes the capital_calls and distributions columns.
    Logs descriptive statistics and the top 5 largest values.
    """
    log_info("Analyzing capital_calls & distributions:")
    
    if "capital_calls" in df.columns:
        calls_desc = df["capital_calls"].describe()
        calls_largest = df["capital_calls"].nlargest(5)
        log_info("capital_calls summary:\n" + str(calls_desc))
        log_info("Top 5 capital_calls:\n" + str(calls_largest))
        print("capital_calls summary:\n", calls_desc)
        print("Top 5 capital_calls:\n", calls_largest)
    else:
        log_info("No 'capital_calls' column found in master_data.")
        print("No 'capital_calls' column found in master_data.")
    
    if "distributions" in df.columns:
        dist_desc = df["distributions"].describe()
        dist_largest = df["distributions"].nlargest(5)
        log_info("distributions summary:\n" + str(dist_desc))
        log_info("Top 5 distributions:\n" + str(dist_largest))
        print("distributions summary:\n", dist_desc)
        print("Top 5 distributions:\n", dist_largest)
    else:
        log_info("No 'distributions' column found in master_data.")
        print("No 'distributions' column found in master_data.")

def analyze_yearly_trends(df):
    """
    Aggregates the data by year and logs the yearly mean, sum, and standard deviation
    for capital_calls and distributions.
    """
    df_yearly = df.resample("YE").agg({
        "capital_calls": ["mean", "sum", "std"],
        "distributions": ["mean", "sum", "std"]
    })
    df_yearly.columns = ['_'.join(col) for col in df_yearly.columns]
    log_info("Yearly aggregated statistics for capital_calls and distributions:\n" + str(df_yearly))
    print("Yearly aggregated statistics:\n", df_yearly)
    return df_yearly


def main():
    os.makedirs("output", exist_ok=True)
    log_info("Output directory created or already exists.")
    
    # Load master_data directly from the database.
    df_master = load_master_data_from_db()
    
    # Inspect underlying components to ensure they vary.
    inspect_underlying_components(df_master)
    
    # Inspect nonzero counts for key columns.
    inspect_nonzero_columns(df_master)
    
    # Analyze capital calls & distributions.
    analyze_capital_calls(df_master)
    
    # Analyze yearly trends.
    analyze_yearly_trends(df_master)


if __name__ == "__main__":
    main()
