import os
import requests
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

# ---------------------------------------------------
# 1. Provide your FMP API key
# ---------------------------------------------------
FMP_API_KEY = "s8afH4ybbBXmxrljmWBm7dOE53MjLTpM"

# ---------------------------------------------------
# 2. Set your ticker and desired statements
# ---------------------------------------------------
ticker = "KKR"
base_url = "https://financialmodelingprep.com/api/v3"

# Endpoints for different statements - set 'period=annual' for annual data
url_income = f"{base_url}/income-statement/{ticker}?apikey={FMP_API_KEY}&limit=100&period=annual"
url_balance = f"{base_url}/balance-sheet-statement/{ticker}?apikey={FMP_API_KEY}&limit=100&period=annual"
url_cashflow = f"{base_url}/cash-flow-statement/{ticker}?apikey={FMP_API_KEY}&limit=100&period=annual"

# ---------------------------------------------------
# 3. Helper function to fetch and transform data
# ---------------------------------------------------
def fetch_financials(url, statement_name):
    """
    Fetch financial statement JSON from FMP and load into a DataFrame.
    Returns an empty DataFrame on failure or if data is missing.
    """
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        
        if not isinstance(data, list) or len(data) == 0:
            print(f"❌ No {statement_name} data found for {ticker}.")
            return pd.DataFrame()
        
        # Convert JSON list to DataFrame
        df = pd.DataFrame(data)
        
        # The FMP statements have a 'date' column indicating the period end (e.g. "2022-12-31").
        # We rename it to 'period_ending' to match your original code style.
        df.rename(columns={'date': 'period_ending'}, inplace=True)
        
        # Convert the period_ending column to datetime
        df['period_ending'] = pd.to_datetime(df['period_ending'], errors='coerce')
        
        # For demonstration, we'll keep all columns. 
        # If you only want certain columns, you can filter them out here.
        
        return df
    except Exception as e:
        print(f"❌ Error fetching {statement_name}: {e}")
        return pd.DataFrame()

# ---------------------------------------------------
# 4. Fetch each statement
# ---------------------------------------------------
income_df = fetch_financials(url_income, "Income Statement")
balance_df = fetch_financials(url_balance, "Balance Sheet")
cashflow_df = fetch_financials(url_cashflow, "Cash Flow Statement")

# ---------------------------------------------------
# 5. Check if data is present
# ---------------------------------------------------
if income_df.empty or balance_df.empty or cashflow_df.empty:
    print("❌ One or more financial statements are empty. Unable to retrieve sufficient data.")
else:
    # ---------------------------------------------------
    # 6. Merge DataFrames on the 'period_ending' column
    # ---------------------------------------------------
    # In FMP data, each row is a single period. We can merge directly on 'period_ending'.
    merged_df = pd.merge(income_df, balance_df, on='period_ending', how='outer', suffixes=('_income', '_balance'))
    merged_df = pd.merge(merged_df, cashflow_df, on='period_ending', how='outer', suffixes=('', '_cashflow'))
    
    # ---------------------------------------------------
    # 7. Rename 'period_ending' to 'Date' and set as index
    # ---------------------------------------------------
    merged_df.rename(columns={'period_ending': 'Date'}, inplace=True)
    merged_df['Date'] = pd.to_datetime(merged_df['Date'], errors='coerce')
    merged_df.set_index('Date', inplace=True)
    
    # ---------------------------------------------------
    # 8. Handling missing values
    # ---------------------------------------------------
    # Fill missing numeric columns with 0 (or use another strategy if desired)
    numeric_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)
    
    # ---------------------------------------------------
    # 9. (Optional) Filter data by date if you only want recent years
    # ---------------------------------------------------
    # Here, we do NOT limit to 5 years, but you could if you wanted to:
    # from datetime import timedelta
    # five_years_ago = datetime.now() - timedelta(days=5*365)
    # merged_df = merged_df[merged_df.index >= five_years_ago]
    
    # ---------------------------------------------------
    # 10. Save data to SQLite
    # ---------------------------------------------------
    os.makedirs("database", exist_ok=True)
    engine = create_engine("sqlite:///database/blackstone_data.db")
    
    merged_df.to_sql("bx_financials", con=engine, if_exists="replace")
    engine.dispose()
    
    print("✅ Financial statement data stored successfully in the 'bx_financials' table.")
    print("Final missing values:\n", merged_df.isnull().sum())
