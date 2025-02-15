import os
import requests
import pandas as pd
from utils.db_utils import store_dataframe
from utils.logging_utils import setup_logging, log_info, log_error
from constants import SEC_Base_URL, SEC_API_KEY

def list_kkr_filings():
    """
    Fetches the most recent KKR filings (10-K and 10-Q) from the SEC API.

    Returns:
        list of tuples: Each tuple contains an accession number and its corresponding filing URL.
    """
    log_info("Fetching KKR filings from SEC API")
    
    # Define the query to filter for KKR filings with form types 10-K or 10-Q.
    query = {
        "query": {"query_string": {"query": "ticker:KKR AND (formType:\"10-K\" OR formType:\"10-Q\")"}},
        "from": "0", 
        "size": "10", 
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    response = requests.post(f"{SEC_Base_URL}/filings", json=query, headers={"Authorization": SEC_API_KEY})
    if response.status_code != 200:
        log_error(f"Error fetching data from SEC API: {response.status_code} - {response.text}")
        raise Exception(f"Error fetching data from SEC API: {response.status_code} - {response.text}")
    return [(f['accessionNo'], f['filingUrl']) for f in response.json()['filings']]

# Extract financial data from the SEC API.
def extract_financials(accession_no):
    """
    Extracts financial data from the SEC API for a given filing.

    Args:
        accession_no (str): The accession number of the filing.

    Returns:
        dict or None: The financial data as a JSON object if successful, else None.
    """
    log_info(f"Extracting financial data for {accession_no}")
    
    # Construct the URL to extract financial data based on the accession number.
    url = f"{SEC_Base_URL}/xbrl-to-json?accessionNo={accession_no}"
    
    # Make a GET request to fetch the financial data.
    resp = requests.get(url, headers={"Authorization": SEC_API_KEY})
    if resp.status_code != 200:
        log_error(f"Failed to extract data for {accession_no}")
    return resp.json() if resp.status_code == 200 else None

# Fetch and store KKR financial data in the database.
def fetch_and_store_sec_data():
    """
    Fetches KKR filings, extracts their financial data, merges different financial statements into one DataFrame,
    cleans the data, and stores it using the store_dataframe utility.
    """
    filings = list_kkr_filings()
    for accession_no, url in filings:
        log_info(f"Fetching data for {accession_no}")
        data = extract_financials(accession_no)
        
        if data:
            # Convert the Income Statement, Balance Sheet, and Cash Flow Statement data into DataFrames.
            income_df = pd.DataFrame(data['IncomeStatement'])
            balance_df = pd.DataFrame(data['BalanceSheet'])
            cashflow_df = pd.DataFrame(data['CashFlowStatement'])
            
            # Merge the three DataFrames on the 'period_ending' column.
            merged_df = income_df.merge(balance_df, on='period_ending', how='outer') \
                                 .merge(cashflow_df, on='period_ending', how='outer')
            
            # Rename the 'period_ending' column to 'Date'.
            merged_df.rename(columns={'period_ending': 'Date'}, inplace=True)
            
            # Convert the 'Date' column to datetime format.
            merged_df['Date'] = pd.to_datetime(merged_df['Date'], errors='coerce')
            
            # Fill missing numeric values with 0.
            numeric_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns
            merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)
            
            # Store the merged DataFrame in the database.
            store_dataframe(merged_df, "kkr_financials", if_exists="append")
            log_info(f"Data for {accession_no} stored successfully.")
        else:
            log_error(f"No data found for {accession_no}")

if __name__ == "__main__":
    setup_logging()
    fetch_and_store_sec_data()
