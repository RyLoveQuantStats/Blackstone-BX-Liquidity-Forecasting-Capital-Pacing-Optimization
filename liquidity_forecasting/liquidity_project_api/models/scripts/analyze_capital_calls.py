#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

# Import centralized logging and DB utilities.
from utils.logging_utils import setup_logging, log_info, log_error
from utils.db_utils import get_connection, DB_PATH  # DB_PATH available if needed
setup_logging()

FILE_PATH = "output/master_data.csv"
ANALYSIS_OUTPUT = "output/pe_capital_calls_analysis.csv"

def load_merged_data(file_path=FILE_PATH):
    try:
        df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
        df.fillna(0, inplace=True)
        log_info(f"Merged data loaded from '{file_path}'.")
        return df
    except Exception as e:
        log_error(f"Error loading merged data from '{file_path}': {e}")
        raise

def inspect_nonzero_columns(df):
    columns_to_check = [
        "netCashProvidedByOperatingActivities",
        "netCashUsedForInvestingActivites",
        "dividendsPaid",
        "purchasesOfInvestments",
        "retainedEarnings",
        "commonStockIssued",
        "commonStockRepurchased",
        "debtRepayment",
        "debtIssued",
        "longTermDebt",
        "shortTermDebt",
        "netCashUsedProvidedByFinancingActivities",
        "capital_calls",
        "distributions",
    ]
    log_info("Non-zero counts for key columns:")
    for col in columns_to_check:
        if col in df.columns:
            non_zero_count = (df[col] != 0).sum()
            log_info(f"{col}: non-zero rows = {non_zero_count} / {len(df)}")
        else:
            log_info(f"{col}: NOT FOUND in df.columns")

def analyze_capital_calls(df):
    log_info("Analyzing capital_calls & distributions:")
    if "capital_calls" in df.columns:
        log_info("capital_calls summary:\n" + str(df["capital_calls"].describe()))
        log_info("Largest capital calls:\n" + str(df["capital_calls"].nlargest(5)))
    else:
        log_info("No 'capital_calls' column found.")

    if "distributions" in df.columns:
        log_info("distributions summary:\n" + str(df["distributions"].describe()))
        log_info("Largest distributions:\n" + str(df["distributions"].nlargest(5)))
    else:
        log_info("No 'distributions' column found.")

def main():
    os.makedirs("output", exist_ok=True)
    log_info("Output directory created or already exists.")

    df = load_merged_data(FILE_PATH)
    inspect_nonzero_columns(df)
    analyze_capital_calls(df)

    # Example correlation analysis with selected columns.
    corr_cols = [
        "capital_calls", "distributions",
        "netCashProvidedByOperatingActivities", "retainedEarnings",
        "longTermDebt", "shortTermDebt"
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]
    if corr_cols:
        corr_matrix = df[corr_cols].corr()
        log_info("Correlation matrix computed:\n" + str(corr_matrix))
        corr_matrix.to_csv(ANALYSIS_OUTPUT)
        log_info(f"Analysis correlation matrix saved to '{ANALYSIS_OUTPUT}'.")
    else:
        log_info("No columns available for correlation analysis.")

if __name__ == "__main__":
    main()
