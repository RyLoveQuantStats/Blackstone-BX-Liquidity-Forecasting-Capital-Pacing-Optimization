'''
Description: Script to compute and enhance approximate capital calls and distributions for the master_data table
(merged from stock_prices, kkr_statements, and macroeconomic_data) for the period 2020-2024.
This advanced version uses rolling growth rates along with macro and volatility adjustments.
The updated data (with new columns "capital_calls" and "distributions") is saved back into the 
master_data table in the DB and exported to CSV.
'''

import pandas as pd
import numpy as np
import os

from utils.logging_utils import setup_logging, log_info, log_error
from utils.db_utils import get_connection, DB_PATH

# Set up logging.
setup_logging()

MASTER_DATA_CSV = "output/master_data.csv"
OUTPUT_ANALYSIS_CSV = "output/master_data_with_capital_calls.csv"

def load_master_data(file_path=MASTER_DATA_CSV):
    """
    Loads the master_data CSV file, parsing the Date column as datetime and setting it as the index.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
        log_info(f"Master data loaded from '{file_path}'. Shape: {df.shape}")
        return df
    except Exception as e:
        log_error(f"Error loading master data from '{file_path}': {e}")
        raise

def calculate_capital_calls_enhanced(df, window=90, growth_weight=0.5):
    """
    Enhanced calculation for capital calls.
    
    Steps:
      1. Compute base capital calls as:
         fin_base_calls = fin_capital_expenditure + Δ(fin_working_capital) + Δ(fin_net_debt)
      2. Compute a rolling percentage growth for fin_capital_expenditure over the specified window.
      3. Compute macro and volatility adjustments:
         - macro_adj = 1 + (average(macro_inflation_rate)/100) if available, else 1.0.
         - vol_adj = 1 + (average(stock_Volatility_30)/100) if available, else 1.0.
      4. Overall adjustment = (1 + growth_weight * fin_capex_growth) * macro_adj * vol_adj.
      5. Final capital_calls = fin_base_calls * overall_adjustment.
    """
    # Ensure required columns exist.
    req_cols = ['fin_capital_expenditure', 'fin_net_debt', 'fin_current_assets', 'fin_current_liabilities']
    for col in req_cols:
        if col not in df.columns:
            log_info(f"Column '{col}' missing. Setting '{col}' to 0.")
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Compute working capital and its changes.
    df["fin_working_capital"] = df["fin_current_assets"] - df["fin_current_liabilities"]
    df["fin_delta_working_capital"] = df["fin_working_capital"].diff().clip(lower=0)
    df["fin_delta_net_debt"] = df["fin_net_debt"].diff().clip(lower=0)
    df["fin_base_calls"] = df["fin_capital_expenditure"] + df["fin_delta_working_capital"] + df["fin_delta_net_debt"]
    log_info("Total base capital calls (sum): " + str(df["fin_base_calls"].sum()))
    
    # Compute rolling growth rate for capital expenditure.
    df["fin_capex_growth"] = df["fin_capital_expenditure"].pct_change(periods=window)
    df["fin_capex_growth"] = df["fin_capex_growth"].fillna(0).clip(lower=-1, upper=1)
    log_info("Rolling capex growth (first 5 rows):\n" + str(df["fin_capex_growth"].head()))
    
    # Compute macro adjustment factor.
    if "macro_inflation_rate" in df.columns:
        macro_adj = 1 + df["macro_inflation_rate"].mean() / 100
        log_info(f"Macro adjustment factor: {macro_adj:.4f}")
    else:
        macro_adj = 1.0
        log_info("No 'macro_inflation_rate' column found. Using macro adjustment of 1.0.")
    
    # Compute volatility adjustment factor.
    if "stock_Volatility_30" in df.columns:
        vol_adj = 1 + df["stock_Volatility_30"].mean() / 100
        log_info(f"Volatility adjustment factor: {vol_adj:.4f}")
    else:
        vol_adj = 1.0
        log_info("No 'stock_Volatility_30' column found. Using volatility adjustment of 1.0.")
    
    # Overall adjustment factor.
    overall_adjustment = (1 + growth_weight * df["fin_capex_growth"]) * macro_adj * vol_adj
    log_info("Overall adjustment factor (first 5 rows):\n" + str(overall_adjustment.head()))
    
    # Calculate enhanced capital calls.
    df["capital_calls"] = df["fin_base_calls"] * overall_adjustment
    log_info("Enhanced capital calls computed. Descriptive stats:\n" + str(df["capital_calls"].describe()))
    
    return df

def calculate_distributions_enhanced(df, window=90, growth_weight=0.5):
    """
    Enhanced calculation for distributions.
    
    Steps:
      1. Compute base distributions:
         fin_base_distributions = fin_operating_cash_flow + fin_retained_earnings
      2. Compute a rolling growth rate for fin_operating_cash_flow over the specified window.
      3. Compute a distribution factor from macro data:
         dist_factor = 1 + (average(macro_10Y Treasury Yield) / 100) if available, else 1.0.
      4. Overall adjustment = (1 + growth_weight * fin_ocf_growth) * dist_factor.
      5. Final distributions = fin_base_distributions * overall adjustment.
    """
    for col in ['fin_operating_cash_flow', 'fin_retained_earnings']:
        if col not in df.columns:
            log_info(f"Column '{col}' missing. Setting '{col}' to 0.")
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df["fin_base_distributions"] = df["fin_operating_cash_flow"] + df["fin_retained_earnings"]
    
    df["fin_ocf_growth"] = df["fin_operating_cash_flow"].pct_change(periods=window)
    df["fin_ocf_growth"] = df["fin_ocf_growth"].fillna(0).clip(lower=-1, upper=1)
    log_info("Rolling OCF growth (first 5 rows):\n" + str(df["fin_ocf_growth"].head()))
    
    if "macro_10Y Treasury Yield" in df.columns:
        dist_factor = 1 + df["macro_10Y Treasury Yield"].mean() / 100
        log_info(f"Distribution factor from macro_10Y Treasury Yield: {dist_factor:.4f}")
    else:
        dist_factor = 1.0
        log_info("No 'macro_10Y Treasury Yield' column found. Using distribution factor of 1.0.")
    
    overall_adjustment = (1 + growth_weight * df["fin_ocf_growth"]) * dist_factor
    log_info("Overall distribution adjustment factor (first 5 rows):\n" + str(overall_adjustment.head()))
    
    df["distributions"] = df["fin_base_distributions"] * overall_adjustment
    log_info("Enhanced distributions computed. Descriptive stats:\n" + str(df["distributions"].describe()))
    
    return df

def store_master_data(df):
    """
    Stores the updated master_data DataFrame (with capital_calls and distributions)
    into the master_data table in the database.
    """
    try:
        conn = get_connection()
        df.to_sql("master_data", conn, if_exists="replace", index=True)
        conn.close()
        log_info("Updated master_data table stored successfully in the database.")
    except Exception as e:
        log_error(f"Error storing master_data: {e}")
        raise

def main():
    os.makedirs("output", exist_ok=True)
    log_info("Output directory created or already exists.")
    
    # Load master_data from CSV.
    df_master = load_master_data(MASTER_DATA_CSV)
    
    # (Optional) Log column names.
    log_info("Master data columns: " + str(list(df_master.columns)))
    
    # Enhanced calculations.
    df_master = calculate_capital_calls_enhanced(df_master, window=90, growth_weight=0.5)
    df_master = calculate_distributions_enhanced(df_master, window=90, growth_weight=0.5)
    
    # Store the updated master_data back into the database.
    store_master_data(df_master)
    
    # Export the updated DataFrame to CSV.
    df_master.to_csv(OUTPUT_ANALYSIS_CSV)
    log_info(f"Final master data with enhanced capital calls and distributions saved as CSV in '{OUTPUT_ANALYSIS_CSV}'.")

if __name__ == "__main__":
    main()
