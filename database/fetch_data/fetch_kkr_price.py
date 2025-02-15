import pandas as pd
import numpy as np
import yfinance as yf
from utils.db_utils import store_dataframe
from utils.logging_utils import setup_logging, log_info, log_error

def fetch_and_store_stock_data():
    log_info("Starting stock data fetch for KKR")
    ticker = "KKR"

    try:
        data = yf.download(ticker, start="2019-01-01", end="2024-01-01")
        data.columns = [col[0] for col in data.columns]
        log_info("Stock data fetched successfully")
    except Exception as e:
        log_error(f"Error fetching stock data: {e}")
        return

    if "Close" in data.columns:
        try:
            data["Daily Return"] = data["Close"].pct_change()
            data["Log Return"] = np.log(data["Close"] / data["Close"].shift(1))
            data["SMA_50"] = data["Close"].rolling(window=50).mean()
            data["SMA_200"] = data["Close"].rolling(window=200).mean()
            data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()
            data["Volatility_30"] = data["Daily Return"].rolling(window=30).std()

            def compute_rsi(series, period=14):
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            data["RSI_14"] = compute_rsi(data["Close"])

            data.fillna({
                "SMA_50": data["SMA_50"].bfill(),
                "SMA_200": data["SMA_200"].bfill(),
                "Volatility_30": 0,
                "RSI_14": 50
            }, inplace=True)

            data.fillna(0, inplace=True)

            store_dataframe(data, "stock_prices")
            log_info("Stock price data stored successfully with cleaned metrics.")
        except Exception as e:
            log_error(f"Error processing stock data: {e}")
    else:
        log_error("'Close' column not found in downloaded data.")

if __name__ == "__main__":
    setup_logging()
    fetch_and_store_stock_data()
