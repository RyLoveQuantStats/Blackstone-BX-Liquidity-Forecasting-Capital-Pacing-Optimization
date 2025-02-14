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
