import pandas as pd
import numpy as np
import sqlite3
import yfinance as yf

# Define stock ticker
ticker = "KKR"

# Fetch stock price data
bx_data = yf.download(ticker, start="2019-12-31", end="2024-01-01")
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
