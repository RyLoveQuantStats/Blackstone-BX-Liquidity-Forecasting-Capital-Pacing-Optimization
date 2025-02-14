#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

FILE_PATH = "output/bx_master_data.csv"
ANALYSIS_OUTPUT = "output/pe_capital_calls_analysis.csv"

def load_merged_data(file_path=FILE_PATH):
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    df.fillna(0, inplace=True)
    return df

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
    print("Non-zero counts for key columns:\n")
    for col in columns_to_check:
        if col in df.columns:
            non_zero_count = (df[col] != 0).sum()
            print(f"{col}: non-zero rows = {non_zero_count} / {len(df)}")
        else:
            print(f"{col}: NOT FOUND in df.columns")

def analyze_capital_calls(df):
    print("\n[INFO] Analyzing capital_calls & distributions:\n")
    if "capital_calls" in df.columns:
        print(df["capital_calls"].describe())
        print("\nLargest capital calls:\n", df["capital_calls"].nlargest(5))
    else:
        print("No 'capital_calls' column found.")

    if "distributions" in df.columns:
        print("\nDistributions describe:\n", df["distributions"].describe())
        print("\nLargest distributions:\n", df["distributions"].nlargest(5))
    else:
        print("No 'distributions' column found.")

def main():
    os.makedirs("output", exist_ok=True)
    df = load_merged_data(FILE_PATH)

    inspect_nonzero_columns(df)
    analyze_capital_calls(df)

    # Example correlation analysis with columns that might exist
    corr_cols = [
        "capital_calls", "distributions",
        "netCashProvidedByOperatingActivities", "retainedEarnings",
        "longTermDebt", "shortTermDebt"
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]
    if corr_cols:
        corr_matrix = df[corr_cols].corr()
        print("\nCorrelation matrix:\n", corr_matrix)
        corr_matrix.to_csv(ANALYSIS_OUTPUT)
        print(f"\nAnalysis correlation matrix saved to {ANALYSIS_OUTPUT}")
    else:
        print("\nNo columns available for correlation analysis.")

if __name__ == "__main__":
    main()
