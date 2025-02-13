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
