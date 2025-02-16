import pandas as pd
import os
from utils.logging_utils import setup_logging, log_info, log_error
from utils.db_utils import get_connection, DB_PATH

# Set up logging.
setup_logging()

def load_master_data():
    """
    Loads the master_data table from the database.
    Expects that the table was saved with the date as the index, with index label 'Date'.
    """
    try:
        conn = get_connection()
        # Read the table; the date column should be in a column called 'Date'
        df = pd.read_sql("SELECT * FROM master_data", conn, parse_dates=["Date"])
        conn.close()
        # Ensure 'Date' is set as the index
        if "Date" in df.columns:
            df.set_index("Date", inplace=True)
        log_info(f"Master data loaded with shape: {df.shape}")
        return df
    except Exception as e:
        log_error(f"Error loading master data: {e}")
        raise

def compute_capital_calls_and_distributions(df):
    """
    Computes both capital calls and distributions.
    
    Capital Calls:
      - net_investment = purchase_of_investment - sale_of_investment
      - capital_call = max(0, net_investment - cash_and_cash_equivalents)
    
    Distributions:
      - distribution = max(0, operating_cash_flow)
    
    Rolling averages (30-day) are computed for smoothing.
    """
    df = df.copy()
    
    # Ensure required columns are available and fill missing values with 0.
    required_cols = ["purchase_of_investment", "sale_of_investment", "cash_and_cash_equivalents", "operating_cash_flow"]
    for col in required_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            log_info(f"Column '{col}' not found in master_data; filling with 0.")
            df[col] = 0

    # Calculate net investment.
    df["net_investment"] = df["purchase_of_investment"] - df["sale_of_investment"]
    
    # Compute capital call: the shortfall between net investment and available cash.
    df["capital_call"] = (df["net_investment"] - df["cash_and_cash_equivalents"]).clip(lower=0)
    
    # Compute distributions: assume only positive operating cash flow counts.
    df["distribution"] = df["operating_cash_flow"].clip(lower=0)
    
    # Compute 30-day rolling averages for smoothing.
    df["capital_call_rolling"] = df["capital_call"].rolling(window=30, min_periods=1).mean()
    df["distribution_rolling"] = df["distribution"].rolling(window=30, min_periods=1).mean()
    
    log_info("Capital calls and distributions computed.")
    return df

def store_master_data(df):
    """
    Stores the updated DataFrame back into the master_data SQL table.
    Uses index_label='Date' to preserve the date index.
    """
    try:
        conn = get_connection()
        df.to_sql("master_data", conn, if_exists="replace", index=True, index_label="Date")
        conn.close()
        log_info("Master data table updated successfully in the database.")
    except Exception as e:
        log_error(f"Error storing master data: {e}")
        raise

def main():
    # Load master data.
    df_master = load_master_data()
    
    # Compute capital calls and distributions.
    df_updated = compute_capital_calls_and_distributions(df_master)
    
    # Store the updated DataFrame back into the SQL master_data table.
    store_master_data(df_updated)
    
    # Also save to a CSV file.
    output_file = "output/csv/capital_calls_master.csv"
    df_updated.to_csv(output_file)
    log_info(f"Master data with capital calls and distributions saved to {output_file}")
    
    return df_updated

if __name__ == "__main__":
    df_calls = main()
    # Display the last 10 days of computed metrics.
    print(df_calls[["capital_call", "capital_call_rolling", "distribution", "distribution_rolling"]].tail(10))
