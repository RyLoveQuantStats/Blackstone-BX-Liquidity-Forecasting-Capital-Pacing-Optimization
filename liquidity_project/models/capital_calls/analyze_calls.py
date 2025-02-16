import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from utils.logging_utils import setup_logging, log_info, log_error
from utils.db_utils import get_connection, DB_PATH

# Set up logging and output paths
setup_logging()
ANALYSIS_OUTPUT = "output/csv/synthetic_capital_calls_analysis.csv"
CORR_CSV = "output/csv/synthetic_correlation.csv"

def load_synthetic_master_data():
    """
    Loads the synthetic_master_data table from the database.
    Assumes that the table has a 'Date' column (or index) with valid datetime values.
    """
    try:
        conn = get_connection()
        # Here, we assume the synthetic_master_data table stores Date as a column.
        df = pd.read_sql("SELECT * FROM synthetic_master_data", conn, parse_dates=["Date"], index_col="Date")
        conn.close()
        log_info(f"Synthetic master data loaded from database. Shape: {df.shape}")
        return df
    except Exception as e:
        log_error(f"Error loading synthetic master data from database: {e}")
        raise

def inspect_financial_components(df):
    """
    Inspects key financial metrics from the synthetic data.
    Logs descriptive statistics and unique values for a set of selected columns.
    """
    components = [
        "fin_investing_cash_flow", "fin_financing_cash_flow",
        "fin_operating_cash_flow", "cash_and_cash_equivalents",
        "macro_10Y Treasury Yield", "capital_call_proxy"
    ]
    for comp in components:
        if comp in df.columns:
            desc = df[comp].describe()
            unique_vals = np.unique(df[comp].dropna())
            log_info(f"{comp} stats:\n{desc}")
            log_info(f"Unique values (first 10) in {comp}: {unique_vals[:10]}")
            print(f"{comp} stats:\n", desc)
            print(f"Unique values (first 10) in {comp}: {unique_vals[:10]}")
        else:
            log_info(f"{comp} not found in synthetic_master_data.")
            print(f"{comp} not found in synthetic_master_data.")

def analyze_yearly_trends(df):
    """
    Aggregates synthetic data by year and computes the yearly mean, sum, and std for capital_call_proxy.
    Returns the aggregated DataFrame.
    """
    df_yearly = df.resample("Y").agg({
        "capital_call_proxy": ["mean", "sum", "std"]
    })
    df_yearly.columns = ['_'.join(col) for col in df_yearly.columns]
    log_info("Yearly aggregated statistics for capital_call_proxy:\n" + str(df_yearly))
    print("Yearly aggregated statistics:\n", df_yearly)
    return df_yearly

def main():
    os.makedirs("output", exist_ok=True)
    
    # Load synthetic master data from SQL
    df_synthetic = load_synthetic_master_data()
    
    # Inspect key financial components
    inspect_financial_components(df_synthetic)
    
    # Analyze yearly trends for the capital call proxy
    df_yearly = analyze_yearly_trends(df_synthetic)
    
    # Save yearly analysis results to CSV
    df_yearly.to_csv(ANALYSIS_OUTPUT)
    log_info(f"Yearly aggregated analysis saved as CSV in '{ANALYSIS_OUTPUT}'.")
    
    return df_synthetic

if __name__ == "__main__":
    synthetic_df = main()
