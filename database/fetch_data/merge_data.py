import pandas as pd
import os
from utils.logging_utils import setup_logging, log_info, log_error
from utils.db_utils import get_connection, DB_PATH

setup_logging()

STOCK_TABLE = "stock_prices"
FINANCIALS_TABLE = "kkr_statements"
MACRO_TABLE = "macroeconomic_data"
OUTPUT_CSV = "output/csv/master_data.csv"

STOCK_COLS = ["Date", "Volatility_30"]
FIN_COLS = [
    "Date",
    "cash_and_cash_equivalents",
    "net_debt",
    "total_assets",
    "total_liabilities_net_minority_interest",
    "operating_cash_flow",
    "investing_cash_flow",
    "financing_cash_flow",
    "free_cash_flow",
    "investments_in_property_plant_and_equipment",
    "purchase_of_investment",
    "sale_of_investment"
]
MACRO_COLS = ["Date", "10Y Treasury Yield", "CPI", "BAA Corporate Bond Spread"]

def fix_kkr_statements(df):
    """
    For the kkr_statements DataFrame:
      1. Delete columns named 'index' and 'Date' if they exist.
      2. Rename 'period_ending_balance' to 'Date'.
      3. Convert the new 'Date' column to datetime.
    """
    df = df.drop(columns=["index", "Date"], errors="ignore")
    if "period_ending_balance" in df.columns:
        df = df.rename(columns={"period_ending_balance": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def load_and_fix_dates(table_name):
    """
    Loads a single table, strips duplicates, and fixes the date column.
    For kkr_statements, applies fix_kkr_statements.
    """
    try:
        conn = get_connection()
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        
        # Strip whitespace and drop duplicate columns.
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Special fix for kkr_statements.
        if table_name == FINANCIALS_TABLE:
            df = fix_kkr_statements(df)
        else:
            if "index" in df.columns:
                df.rename(columns={"index": "Date"}, inplace=True)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        
        log_info(f"Loaded '{table_name}'. Columns now: {list(df.columns)}")
        return df
    except Exception as e:
        log_error(f"Error loading table '{table_name}': {e}")
        raise

def prepare_and_reindex(df, use_cols, global_start, global_end, freq="D"):
    """
    1. Keep only the columns in use_cols.
    2. Convert 'Date' to datetime, set as index, and sort.
    3. Reindex to a daily frequency from global_start to global_end.
    4. Forward-fill then backfill any missing values.
    """
    df = df[use_cols].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    
    # Create a daily date range.
    full_range = pd.date_range(global_start, global_end, freq=freq)
    
    # Reindex and fill missing values.
    df = df.reindex(full_range)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    return df

def main():
    os.makedirs("output", exist_ok=True)
    log_info("Starting data merge with daily reindex approach...")
    
    # Global date range for reindexing.
    global_start = "2020-01-01"
    global_end = "2024-12-31"
    
    # 1. Load each table.
    df_stock_raw = load_and_fix_dates(STOCK_TABLE)
    df_fin_raw   = load_and_fix_dates(FINANCIALS_TABLE)
    df_macro_raw = load_and_fix_dates(MACRO_TABLE)
    
    # 2. Prepare each DataFrame using the global date range.
    df_stock = prepare_and_reindex(df_stock_raw, STOCK_COLS, global_start, global_end)
    df_fin   = prepare_and_reindex(df_fin_raw, FIN_COLS, global_start, global_end)
    df_macro = prepare_and_reindex(df_macro_raw, MACRO_COLS, global_start, global_end)
    
    # 3. Combine all data by joining on the daily index.
    df_merged = df_stock.join(df_fin, how="outer").join(df_macro, how="outer")
    
    # 4. Filter final date range (redundant as we reindexed to this range already).
    df_merged = df_merged.loc[global_start:global_end]
    log_info(f"Data after final date filter: {df_merged.shape}")
    
    # 5. Replace zeros in 'purchase_of_investment' and 'sale_of_investment' with NaN,
    #    then backfill those NaNs with the next non-zero value, and finally fill remaining NaNs with 0.
    for col in ["purchase_of_investment", "sale_of_investment"]:
        if col in df_merged.columns:
            # Replace 0 with NaN only if the value is 0 (and not already NaN)
            df_merged[col] = df_merged[col].replace(0, pd.NA)
            # Backfill NaNs with the next valid value
            df_merged[col] = df_merged[col].bfill()
            # Fill any remaining NaNs with 0
            df_merged[col] = df_merged[col].fillna(0)
    
    # 6. Store in DB and CSV.
    try:
        conn = get_connection()
        df_merged.to_sql("master_data", conn, if_exists="replace", index=True)
        conn.close()
        log_info("Master data table stored successfully in the database.")
    except Exception as e:
        log_error(f"Error storing master data: {e}")
        raise

    df_merged.to_csv(OUTPUT_CSV)
    log_info(f"Final merged data saved as CSV in '{OUTPUT_CSV}'.")
    
    return df_merged

if __name__ == "__main__":
    master_df = main()
    print(master_df.head(10))
