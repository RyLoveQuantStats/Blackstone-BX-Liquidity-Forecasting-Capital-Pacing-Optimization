import os
import pandas as pd
from fredapi import Fred
from utils.db_utils import store_dataframe
from utils.logging_utils import setup_logging, log_info, log_error
from constants import FRED_Base_URL, FRED_API_KEY

fred_tickers = {
    "10Y Treasury Yield": "DGS10",
    "5Y Treasury Yield":  "DGS5",
    "2Y Treasury Yield":  "DGS2",
    "30Y Treasury Yield": "DGS30",
    "CPI":                "CPIAUCSL",
    "BAA Corporate Bond Spread": "BAA10Y",
    "High Yield Bond Index":     "BAMLH0A0HYM2"
}

def fetch_and_store_fred_data():
    log_info("Starting to fetch data from FRED API")
    def fetch_fred_data(tickers, start="2019-12-31", end="2024-01-01"):
        data_frames = []
        for label, series_id in tickers.items():
            try:
                series = Fred.get_series(series_id, observation_start=start, observation_end=end)
                df = series.to_frame(name=label)
                data_frames.append(df)
                log_info(f"Successfully fetched data for {label}")
            except Exception as e:
                log_error(f"Error fetching '{label}' (ID: {series_id}) from FRED: {e}")

        return pd.concat(data_frames, axis=1) if data_frames else pd.DataFrame()

    macro_data = fetch_fred_data(fred_tickers)
    macro_data.ffill(inplace=True)
    macro_data.bfill(inplace=True)

    store_dataframe(macro_data, "macroeconomic_data")
    log_info("Macroeconomic data successfully stored in SQL database.")

if __name__ == "__main__":
    setup_logging()
    fetch_and_store_fred_data()
