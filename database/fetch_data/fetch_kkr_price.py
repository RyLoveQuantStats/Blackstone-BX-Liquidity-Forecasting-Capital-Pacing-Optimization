import pandas as pd
import numpy as np
import yfinance as yf
from utils.db_utils import store_dataframe
from utils.logging_utils import setup_logging, log_info, log_error  # Import centralized logging utilities

def fetch_and_store_stock_data():
    log_info("Starting stock data fetch for KKR")
    ticker = "KKR"

    try:
        bx_data = yf.download(ticker, start="2019-01-01", end="2024-01-01")
        bx_data.columns = [col[0] for col in bx_data.columns]
        log_info("Stock data fetched successfully")
    except Exception as e:
        log_error(f"Error fetching stock data: {e}")
        return

    if "Close" in bx_data.columns:
        try:
            bx_data["Daily Return"] = bx_data["Close"].pct_change()
            bx_data["Log Return"] = np.log(bx_data["Close"] / bx_data["Close"].shift(1))
            bx_data["SMA_50"] = bx_data["Close"].rolling(window=50).mean()
            bx_data["SMA_200"] = bx_data["Close"].rolling(window=200).mean()
            bx_data["EMA_50"] = bx_data["Close"].ewm(span=50, adjust=False).mean()
            bx_data["Volatility_30"] = bx_data["Daily Return"].rolling(window=30).std()

            def compute_rsi(series, period=14):
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            bx_data["RSI_14"] = compute_rsi(bx_data["Close"])

            bx_data.fillna({
                "SMA_50": bx_data["SMA_50"].bfill(),
                "SMA_200": bx_data["SMA_200"].bfill(),
                "Volatility_30": 0,
                "RSI_14": 50
            }, inplace=True)

            bx_data.fillna(0, inplace=True)

            store_dataframe(bx_data, "stock_prices")
            log_info("Stock price data stored successfully with cleaned metrics.")
        except Exception as e:
            log_error(f"Error processing stock data: {e}")
    else:
        log_error("'Close' column not found in downloaded data.")

if __name__ == "__main__":
    setup_logging()
    fetch_and_store_stock_data()
