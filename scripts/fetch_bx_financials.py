import os
import pandas as pd
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta

# Define stock ticker
ticker = "BX"

# Create Ticker object for BX
stock = yf.Ticker(ticker)

# Fetch financial statements from Yahoo Finance
# Yahoo provides these as DataFrames with dates as columns and line items as the index.
income = stock.financials       # Income Statement
balance = stock.balance_sheet   # Balance Sheet
cashflow = stock.cashflow       # Cash Flow Statement

# Check that data exists
if income.empty or balance.empty or cashflow.empty:
    print("❌ One or more financial statements are empty. Unable to retrieve data.")
else:
    # Transpose each DataFrame so that rows represent reporting periods,
    # and reset the index to get the reporting period as a column named "period_ending".
    income_df = income.transpose().reset_index().rename(columns={"index": "period_ending"})
    balance_df = balance.transpose().reset_index().rename(columns={"index": "period_ending"})
    cashflow_df = cashflow.transpose().reset_index().rename(columns={"index": "period_ending"})

    # Convert the reporting period to datetime
    income_df["period_ending"] = pd.to_datetime(income_df["period_ending"])
    balance_df["period_ending"] = pd.to_datetime(balance_df["period_ending"])
    cashflow_df["period_ending"] = pd.to_datetime(cashflow_df["period_ending"])

    # Filter data for the last 5 years (Yahoo typically provides only this range for financials)
    five_years_ago = datetime.now() - timedelta(days=5 * 365)
    income_df = income_df[income_df["period_ending"] >= five_years_ago]
    balance_df = balance_df[balance_df["period_ending"] >= five_years_ago]
    cashflow_df = cashflow_df[cashflow_df["period_ending"] >= five_years_ago]

    # Merge the financial statements on the common key "period_ending"
    merged_df = income_df.merge(balance_df, on="period_ending", how="outer", suffixes=("_income", "_balance"))
    merged_df = merged_df.merge(cashflow_df, on="period_ending", how="outer", suffixes=("", "_cashflow"))

    # Preprocessing:
    # Rename the merge key to "Date", convert to datetime, set as index, and fill missing values.
    merged_df.rename(columns={"period_ending": "Date"}, inplace=True)
    merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors="coerce")
    merged_df.set_index("Date", inplace=True)
    merged_df.fillna(0, inplace=True)

    # Final check for missing values (optional)
    print("Final missing values:\n", merged_df.isnull().sum())

    # Connect to SQLite database and store the merged data into the bx_financials table
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect("database/blackstone_data.db")
    merged_df.to_sql("bx_financials", conn, if_exists="replace", index=True)
    conn.close()

    print("✅ Financial statement data stored successfully in the 'bx_financials' table.")
