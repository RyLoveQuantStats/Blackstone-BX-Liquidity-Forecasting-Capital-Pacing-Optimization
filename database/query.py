"""
Description: Script to inspect all column names in the three tables:
  - stock_prices
  - kkr_statements
  - macroeconomic_data
"""

import pandas as pd
import os

from utils.logging_utils import setup_logging, log_info, log_error
from utils.db_utils import get_connection, DB_PATH

# Set up logging.
setup_logging()

# Define table names.
STOCK_TABLE = "stock_prices"
FINANCIALS_TABLE = "kkr_statements"
MACRO_TABLE = "macroeconomic_data"

def inspect_table_columns(table_name):
    """
    Loads a single row from the specified table and returns the column names.
    """
    try:
        conn = get_connection()
        # Using LIMIT 1 to minimize data transfer
        query = f"SELECT * FROM {table_name} LIMIT 1"
        df = pd.read_sql(query, conn)
        conn.close()
        columns = list(df.columns)
        log_info(f"Columns in table '{table_name}': {columns}")
        return columns
    except Exception as e:
        log_error(f"Error inspecting table '{table_name}': {e}")
        raise

def main():
    os.makedirs("output", exist_ok=True)
    log_info("Output directory created or already exists.")
    
    stock_columns = inspect_table_columns(STOCK_TABLE)
    fin_columns = inspect_table_columns(FINANCIALS_TABLE)
    macro_columns = inspect_table_columns(MACRO_TABLE)
    
    # Optionally, print them to standard output as well.
    print("Stock Table Columns:", stock_columns)
    print("Financials Table Columns:", fin_columns)
    print("Macro Table Columns:", macro_columns)

if __name__ == "__main__":
    main()
