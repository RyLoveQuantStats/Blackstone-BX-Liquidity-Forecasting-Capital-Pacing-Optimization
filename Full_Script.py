import os
import requests
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

API_KEY = "167b0f020738341948c35033b570748e599f5e632b62593d64e1c926967d28ac"
BASE_URL = "https://api.sec-api.io"

ticker = "KKR"

# SQL Database connection
os.makedirs("database", exist_ok=True)
engine = create_engine("sqlite:///database/kkr_data.db")

# List KKR 10-K and 10-Q filings
def list_kkr_filings():
    query = {
        "query": {"query_string": {"query": "ticker:KKR AND (formType:\"10-K\" OR formType:\"10-Q\")"}},
        "from": "0", "size": "10", "sort": [{"filedAt": {"order": "desc"}}]
    }
    response = requests.post(f"{BASE_URL}/filings", json=query, headers={"Authorization": API_KEY})
    return [(f['accessionNo'], f['filingUrl']) for f in response.json()['filings']]

# Extract financials from SEC API
def extract_financials(accession_no):
    url = f"{BASE_URL}/xbrl-to-json?accessionNo={accession_no}"
    resp = requests.get(url, headers={"Authorization": API_KEY})
    return resp.json() if resp.status_code == 200 else None

filings = list_kkr_filings()
all_data = []

for accession_no, url in filings:
    print(f"Fetching data for {accession_no}")
    data = extract_financials(accession_no)
    if data:
        income_df = pd.DataFrame(data['IncomeStatement'])
        balance_df = pd.DataFrame(data['BalanceSheet'])
        cashflow_df = pd.DataFrame(data['CashFlowStatement'])

        merged_df = income_df.merge(balance_df, on='period_ending', how='outer').merge(cashflow_df, on='period_ending', how='outer')
        merged_df.rename(columns={'period_ending': 'Date'}, inplace=True)
        merged_df['Date'] = pd.to_datetime(merged_df['Date'], errors='coerce')
        numeric_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns
        merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)

        merged_df.to_sql("kkr_financials", con=engine, if_exists="append", index=False)

print("✅ KKR financial data successfully stored in SQL database.")


import pandas as pd
import numpy as np
import sqlite3
import yfinance as yf

# Define stock ticker
ticker = "KKR"

# Fetch stock price data
bx_data = yf.download(ticker, start="2019-01-01", end="2024-01-01")
print(bx_data.columns)

# Flatten MultiIndex columns
bx_data.columns = [col[0] for col in bx_data.columns]
print("Updated Columns:", bx_data.columns)

# Compute additional metrics using 'Close' instead of 'Adj Close'
if "Close" in bx_data.columns:
    bx_data["Daily Return"] = bx_data["Close"].pct_change()
    bx_data["Log Return"] = np.log(bx_data["Close"] / bx_data["Close"].shift(1))
    bx_data["SMA_50"] = bx_data["Close"].rolling(window=50).mean()
    bx_data["SMA_200"] = bx_data["Close"].rolling(window=200).mean()
    bx_data["EMA_50"] = bx_data["Close"].ewm(span=50, adjust=False).mean()
    bx_data["Volatility_30"] = bx_data["Daily Return"].rolling(window=30).std()
    
    # Compute RSI
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    bx_data["RSI_14"] = compute_rsi(bx_data["Close"])

    # 🔹 Fix NULL values before storing in SQL

    # Backfill missing values for Moving Averages (SMA)
    bx_data["SMA_50"] = bx_data["SMA_50"].fillna(method="bfill")
    bx_data["SMA_200"] = bx_data["SMA_200"].fillna(method="bfill")

    # Fill missing values for Volatility with 0
    bx_data["Volatility_30"] = bx_data["Volatility_30"].fillna(0)

    # RSI: Fill missing values with neutral RSI (50)
    bx_data["RSI_14"] = bx_data["RSI_14"].fillna(50)

    # Ensure all NaNs are replaced with 0 before storing in SQL
    bx_data = bx_data.fillna(0)

    # Final check to confirm no NaN values remain
    print("Final missing values:\n", bx_data.isnull().sum())

    # Connect to database and store cleaned data
    conn = sqlite3.connect("database/blackstone_data.db")
    bx_data.to_sql("bx_stock_prices", conn, if_exists="replace", index=True)
    conn.close()

    print("✅ Stock price data stored successfully with cleaned metrics.")
else:
    print("❌ Error: 'Close' column not found in downloaded data.")

import pandas as pd
import sqlite3
from fredapi import Fred

# 1. Initialize the Fred client with your API key
fred = Fred(api_key="e795295f1d454318e2ac436f480317d2")

# 2. Define FRED tickers (Series IDs)
fred_tickers = {
    "10Y Treasury Yield": "DGS10",
    "5Y Treasury Yield":  "DGS5",
    "2Y Treasury Yield":  "DGS2",
    "30Y Treasury Yield": "DGS30",
    "CPI":                "CPIAUCSL",
    "BAA Corporate Bond Spread": "BAA10Y",
    "High Yield Bond Index":     "BAMLH0A0HYM2"
}

# 3. Fetch data from FRED
def fetch_fred_data(tickers, start="2019-12-31", end="2024-01-01"):
    """
    Uses fredapi to download each series between the specified date range,
    then concatenates them into a single pandas DataFrame.
    """
    data_frames = []

    for label, series_id in tickers.items():
        try:
            # Get the time series as a pandas Series
            series = fred.get_series(series_id, observation_start=start, observation_end=end)
            # Convert to DataFrame and rename the column
            df = series.to_frame(name=label)
            data_frames.append(df)
        except Exception as e:
            print(f"❌ Error fetching '{label}' (ID: {series_id}) from FRED: {e}")

    if data_frames:
        # Concatenate all DataFrames along columns
        return pd.concat(data_frames, axis=1)
    else:
        return pd.DataFrame()

# 4. Retrieve the macro data
macro_data = fetch_fred_data(fred_tickers)

# 5. Forward-fill missing values
macro_data.ffill(inplace=True)  
macro_data.bfill(inplace=True)

# 6. Display a sample of the (now filled) data
print(macro_data.head(20))

# 7. Store the data in SQLite
def store_in_sqlite(df, table_name, db_path="blackstone_data.db"):
    """
    Stores a DataFrame into a specified SQLite database file,
    replacing the table if it already exists.
    """
    conn = sqlite3.connect("database/blackstone_data.db")
    df.to_sql(table_name, conn, if_exists="replace", index=True)
    conn.close()
    print(f"✅ Data successfully stored in '{table_name}' in {db_path}")

store_in_sqlite(macro_data, "macroeconomic_data")

#!/usr/bin/env python3

import sqlite3
import pandas as pd
import numpy as np
import os

DB_PATH = "database/blackstone_data.db"
OUTPUT_CSV = "output/bx_master_data.csv"

def load_data_from_sql(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df_stock = pd.read_sql("SELECT * FROM bx_stock_prices", conn)
    df_financials = pd.read_sql("SELECT * FROM bx_financials", conn)
    df_macro = pd.read_sql("SELECT * FROM macroeconomic_data", conn)
    conn.close()
    return df_stock, df_financials, df_macro

def prepare_date_column(df):
    """Prepares a proper Date index from whichever column is recognized as date-like."""
    possible_date_cols = ["Date", "date", "index", "Unnamed: 0", "level_0", "period_ending"]
    date_col = next((col for col in possible_date_cols if col in df.columns), None)
    if not date_col:
        raise ValueError("No date-like column found.")
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.set_index("Date").sort_index().drop_duplicates()
    df = df[df.index.notnull()]
    return df

def resample_data(df, freq="D"):
    """Resamples the data to daily frequency and forward-fills missing days."""
    df = df.sort_index()
    return df.resample(freq).ffill()

def merge_data(df_stock, df_financials, df_macro):
    """
    Merges stock prices, financials, and macro data on a daily basis.
    You can confirm that forward-filling CPI daily doesn't create zeros
    by printing a few rows of df_macro before merging.
    """
    df_financials_daily = resample_data(df_financials, freq="D")
    df_merged = df_stock.merge(df_financials_daily, how="left", left_index=True, right_index=True)

    df_macro_daily = resample_data(df_macro, freq="D")
    df_merged = df_merged.merge(df_macro_daily, how="left", left_index=True, right_index=True)
    return df_merged

def compute_yoy_inflation(df):
    """
    Computes a year-over-year inflation rate from monthly CPI data.
    - First, resample CPI to monthly means
    - Then compute yoy = (CPI[t] - CPI[t-12]) / CPI[t-12]
    - Cap extremes
    Returns a single yoy_inflation value (mean or last).
    """
    if "CPI" not in df.columns:
        return 0.0  # no CPI column, fallback

    # Resample CPI to monthly
    cpi_monthly = df["CPI"].resample("M").mean().dropna()
    if len(cpi_monthly) < 13:
        # Not enough data for yoy (need at least 12 months)
        return 0.0

    # yoy series
    yoy_series = cpi_monthly.pct_change(periods=12)
    # Cap extremes
    yoy_series.loc[yoy_series > 1.0] = 1.0    # 100% yoy
    yoy_series.loc[yoy_series < -0.5] = -0.5  # -50% yoy
    yoy_series.fillna(0, inplace=True)

    # For example, take the most recent yoy or an average yoy
    yoy_inflation = yoy_series.iloc[-1]  # or yoy_series.mean()
    return yoy_inflation

def calculate_pe_capital_calls(df, base_rate=0.05):
    """
    Example PE approach using totalInvestments, netDebt, operatingCashFlow, retainedEarnings, plus:
      - yoy_inflation from monthly CPI
      - mean interest_rate from 10Y Treasury
      - volatility from Volatility_30
    """
    needed_cols = [
        "totalInvestments", "netDebt", "operatingCashFlow", "retainedEarnings",
        "CPI", "10Y Treasury Yield", "Volatility_30"
    ]
    # Ensure columns exist
    for col in needed_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    # 1) Compute yoy inflation from monthly data
    yoy_inflation = compute_yoy_inflation(df)

    # 2) interest_rate
    interest_rate = df["10Y Treasury Yield"].mean()  # e.g. average over entire period

    # 3) volatility factor
    # If Volatility_30 is zero or small, ensure no blow-up
    vol_series = df["Volatility_30"].fillna(0)
    vol_factor = vol_series.mean() / 100.0

    # 4) final call rate
    # e.g. base_rate * (1 + yoy_inflation) * ...
    adjusted_call_rate = base_rate * (1 + yoy_inflation) * (1 + vol_factor) * (1 + interest_rate / 100)

    df["capital_calls"] = ((df["totalInvestments"] * 0.02) + (df["netDebt"] * 0.01)) * adjusted_call_rate

    df["distributions"] = ((df["operatingCashFlow"] * 0.05) + (df["retainedEarnings"] * 0.01)) * (1 + interest_rate / 100)

    print("[INFO] yoy_inflation={:.4f}, interest_rate={:.2f}, vol_factor={:.4f}, call_rate={:.4f}".format(
        yoy_inflation, interest_rate, vol_factor, adjusted_call_rate
    ))
    return df

def clean_merged_data(df):
    return df.fillna(0)

def store_merged_data(df, table_name="bx_master_data", db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace")
    conn.close()
    print(f"✅ Merged data stored in '{table_name}' in '{db_path}'")

def main():
    os.makedirs("output", exist_ok=True)

    # 1. Load data
    df_stock, df_financials, df_macro = load_data_from_sql()
    df_stock = prepare_date_column(df_stock)
    df_financials = prepare_date_column(df_financials)
    df_macro = prepare_date_column(df_macro)

    # 2. Merge
    df_merged = merge_data(df_stock, df_financials, df_macro)

    # 3. Capital calls using yoy inflation
    df_merged = calculate_pe_capital_calls(df_merged)

    # 4. Final cleaning
    df_merged = clean_merged_data(df_merged)

    # 5. Store
    store_merged_data(df_merged, table_name="bx_master_data")
    df_merged.to_csv(OUTPUT_CSV)
    print(f"✅ Final data also saved as '{OUTPUT_CSV}'")

if __name__ == "__main__":
    main()

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

#!/usr/bin/env python3

"""
eda_and_feature_engineering.py

Loads the merged 'master' dataset (bx_master_data) from the database or CSV,
performs basic Exploratory Data Analysis (EDA), and demonstrates 
feature engineering steps.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DB_PATH = "database/blackstone_data.db"
MASTER_TABLE = "bx_master_data"

def load_master_data_from_sql(db_path=DB_PATH, table=MASTER_TABLE):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()

    # Convert index column if it exists
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
    elif "index" in df.columns:
        df.rename(columns={"index": "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

    return df

def eda_plots(df):
    """
    A few EDA plots: time-series chart, histogram, correlation heatmap.
    Adjust columns as appropriate for your data.
    """
    # 1. Simple time-series plot of BX closing price
    if "Close" in df.columns:
        df["Close"].plot(figsize=(12,6), title="BX Close Price Over Time")
        plt.show()

    # 2. Distribution of daily returns (if present)
    if "Daily Return" in df.columns:
        df["Daily Return"].hist(bins=50)
        plt.title("Distribution of Daily Returns")
        plt.show()

    # 3. Correlation Heatmap (for numeric columns)
    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def feature_engineering(df):
    """
    Example of creating new features:
      - Lag features
      - Ratios (Debt/Assets)
      - Growth rates
    Adjust to fit your data context.
    """
    # Example: Create a lag of 10Y Treasury yield if it exists
    if "10Y Treasury Yield" in df.columns:
        df["10Y_Treasury_Lag30"] = df["10Y Treasury Yield"].shift(30)

    # Example: create a financial ratio if totalLiab and totalAssets exist
    if "totalLiab" in df.columns and "totalAssets" in df.columns:
        # Avoid division by zero
        df["Debt_to_Assets"] = df.apply(
            lambda row: (row["totalLiab"] / row["totalAssets"]) 
            if row["totalAssets"] != 0 
            else 0, 
            axis=1
        )

    # Example: create a moving average of close price
    if "Close" in df.columns:
        df["MA_20"] = df["Close"].rolling(20).mean()
        df["MA_50"] = df["Close"].rolling(50).mean()

    # Fill any new NaN introduced by rolling/shift
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    return df

def main():
    # 1. Load the master dataset
    df = load_master_data_from_sql()

    print("Master dataset loaded. Shape:", df.shape)
    print("Columns:\n", df.columns)

    # 2. Basic EDA
    eda_plots(df)

    # 3. Create new features
    df = feature_engineering(df)

    # 4. (Optional) store updated DataFrame back to DB or to CSV
    # Example: store as new table "bx_master_features"
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("bx_master_features", conn, if_exists="replace")
    conn.close()

    os.makedirs("output", exist_ok=True)
    df.to_csv("output/bx_master_features.csv")
    print("✅ Feature-engineered data stored to DB and CSV.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
capital_pacing_optimization.py

A simple example using scipy.optimize to determine optimal capital commitments 
across three strategies while ensuring liquidity constraints are met.

Note: The multipliers (e.g., 2% for totalInvestments, 1% for netDebt) are placeholders.
In practice, calibrate these factors to historical KKR data.
"""

import numpy as np
from scipy.optimize import minimize

# Hypothetical parameters (tweak or calibrate these as needed)
expected_returns = np.array([0.12, 0.10, 0.15])  # Expected returns for three strategies
initial_liquidity = 100_000_000                  # Total liquidity available
max_commit = np.array([50_000_000, 60_000_000, 40_000_000])  # Maximum commitments allowed

def total_returns(commitments):
    """Objective: maximize total return (we minimize the negative)."""
    return -np.sum(commitments * expected_returns)

def liquidity_constraint(commitments):
    """# On Windows:
    env\Scripts\activate
    Liquidity constraint: Assume forecasted capital calls (e.g., 10M) and maintain a 30% liquidity buffer.
    Maximum available capital = initial_liquidity - forecasted_calls - buffer.
    """
    forecasted_calls = 10_000_000  # Replace with model forecast if available.
    buffer = initial_liquidity * 0.30
    max_available = initial_liquidity - forecasted_calls - buffer
    return max_available - np.sum(commitments)

def main():
    cons = ({ "type": "ineq", "fun": liquidity_constraint },)
    bounds = [(0, mc) for mc in max_commit]
    x0 = np.zeros(len(expected_returns))
    
    solution = minimize(total_returns, x0, constraints=cons, bounds=bounds, method="SLSQP")
    if solution.success:
        optimal_commitments = solution.x
        print("Optimal commitments:", optimal_commitments)
        print("Max Return Achieved:", -solution.fun)
    else:
        print("Optimization failed:", solution.message)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
liquidity_forecasting_arima.py

Loads the merged KKR dataset from the database (or CSV),
fits a SARIMAX model to the 'capital_calls' time series,
and produces a forecast with error metrics and plots.

Enhancements:
 - Ensures the Date index has an inferred frequency.
 - Checks for potential stationarity issues.
"""

import sqlite3
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import os

DB_PATH = "database/blackstone_data.db"
TABLE_NAME = "bx_master_data"

def load_master_data(db_path=DB_PATH, table=TABLE_NAME):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["Date"])
    conn.close()
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    # Attempt to infer frequency if not set
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq:
        df.index.freq = inferred_freq
        print(f"[INFO] Inferred frequency: {inferred_freq}")
    else:
        print("[WARN] Could not infer frequency from Date index.")
    return df

def main():
    os.makedirs("plots", exist_ok=True)
    df = load_master_data()
    if "capital_calls" not in df.columns:
        raise ValueError("❌ 'capital_calls' column not found. Check your data ingestion pipeline.")
    
    df = df[df["capital_calls"].notnull()]
    
    # Simple train/test split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    train_y = train_df["capital_calls"]
    test_y = test_df["capital_calls"]
    
    # Fit SARIMAX model (tune order as needed)
    model = SARIMAX(train_y, order=(2,1,2))
    results = model.fit(disp=False)
    print(results.summary())
    
    # Forecast over the test period
    n_forecast = len(test_df)
    forecast = results.forecast(steps=n_forecast)
    test_df["forecast"] = forecast
    
    mae = np.mean(np.abs(test_df["capital_calls"] - test_df["forecast"]))
    rmse = np.sqrt(np.mean((test_df["capital_calls"] - test_df["forecast"])**2))
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    plt.figure(figsize=(10,5))
    plt.plot(train_df["capital_calls"], label="Train")
    plt.plot(test_df["capital_calls"], label="Test (Actual)")
    plt.plot(test_df["forecast"], label="Forecast", linestyle="--")
    plt.title("Capital Calls - ARIMA Forecast")
    plt.legend()
    plt.savefig("plots/capital_calls_forecast.png")
    plt.show()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
simulate_liquidity_monte_carlo.py

Runs Monte Carlo simulations on the 'capital_calls' time series from the merged data,
estimates daily changes, and computes potential future outcomes.

Enhancements:
 - Clips simulated outcomes to a floor of zero (since negative capital calls are unrealistic).
 - Prints simulation percentiles.
"""

import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "database/blackstone_data.db"
TABLE_NAME = "bx_master_data"

def load_master_data(db_path=DB_PATH, table=TABLE_NAME):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["Date"])
    conn.close()
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df

def main():
    df = load_master_data()
    if "capital_calls" not in df.columns:
        raise ValueError("❌ 'capital_calls' column not found. Check your data ingestion pipeline.")
    
    df["calls_change"] = df["capital_calls"].diff().fillna(0)
    mean_change = df["calls_change"].mean()
    std_change = df["calls_change"].std()
    print(f"Mean daily change: {mean_change:.2f}, Std: {std_change:.2f}")
    
    n_simulations = 100000
    horizon = 30  # Forecast horizon (e.g., 30 days)
    last_value = df["capital_calls"].iloc[-1]
    
    outcomes = []
    for _ in range(n_simulations):
        shocks = np.random.normal(mean_change, std_change, horizon)
        simulated_value = last_value + np.sum(shocks)
        # Capital calls cannot be negative: clip to 0
        outcomes.append(max(0, simulated_value))
    
    outcomes = np.array(outcomes)
    p5 = np.percentile(outcomes, 5)
    p50 = np.percentile(outcomes, 50)
    p95 = np.percentile(outcomes, 95)
    
    print(f"5th percentile: {p5:.2f}")
    print(f"Median: {p50:.2f}")
    print(f"95th percentile: {p95:.2f}")

if __name__ == "__main__":
    main()
