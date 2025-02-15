import requests
import pandas as pd
from utils.db_utils import store_dataframe
from utils.logging_utils import setup_logging, log_info, log_error  # Centralized logging import

API_KEY = "6c9f0ac36786b37e718bd67d7544e6c884288b1689398bf749f7c79e5d070d39"
BASE_URL = "https://api.sec-api.io"

def list_kkr_filings():
    log_info("Fetching KKR filings from SEC API")
    query = {
        "query": {"query_string": {"query": "ticker:KKR AND (formType:\"10-K\" OR formType:\"10-Q\")"}},
        "from": "0", "size": "10", "sort": [{"filedAt": {"order": "desc"}}]
    }
    response = requests.post(f"{BASE_URL}/filings", json=query, headers={"Authorization": API_KEY})
    if response.status_code != 200:
        log_error(f"Error fetching data from SEC API: {response.status_code} - {response.text}")
        raise Exception(f"Error fetching data from SEC API: {response.status_code} - {response.text}")
    return [(f['accessionNo'], f['filingUrl']) for f in response.json()['filings']]

def extract_financials(accession_no):
    log_info(f"Extracting financial data for {accession_no}")
    url = f"{BASE_URL}/xbrl-to-json?accessionNo={accession_no}"
    resp = requests.get(url, headers={"Authorization": API_KEY})
    if resp.status_code != 200:
        log_error(f"Failed to extract data for {accession_no}")
    return resp.json() if resp.status_code == 200 else None

def fetch_and_store_sec_data():
    filings = list_kkr_filings()
    for accession_no, url in filings:
        log_info(f"Fetching data for {accession_no}")
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
            store_dataframe(merged_df, "kkr_financials", if_exists="append")
            log_info(f"Data for {accession_no} stored successfully.")
        else:
            log_error(f"No data found for {accession_no}")

if __name__ == "__main__":
    setup_logging()
    fetch_and_store_sec_data()
