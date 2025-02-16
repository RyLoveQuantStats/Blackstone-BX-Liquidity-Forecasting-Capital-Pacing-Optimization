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
      1. Drop any columns named 'index'.
      2. Convert the 'Date' column to datetime.
    Assumes that the table already contains the correct 'Date' column.
    """
    df = df.drop(columns=["index"], errors="ignore")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        log_error("No 'Date' column found in financial statements!")
    return df

def load_and_fix_dates(table_name):
    """
    Loads a table, strips duplicates, and fixes the date column.
    - For kkr_statements, applies fix_kkr_statements.
    - For other tables, if an 'index' column exists, renames it to 'Date' and converts 'Date' to datetime.
    """
    try:
        conn = get_connection()
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()

        # Strip whitespace and drop duplicate columns
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.duplicated()]

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
    1. Keep only columns in use_cols (including 'Date').
    2. Check if 'Date' exists; if not, return an empty DataFrame.
    3. Convert 'Date' to datetime, set as index, and sort.
    4. Reindex to a daily frequency from global_start to global_end.
    5. Forward-fill then backfill missing values.
    """
    existing_cols = [c for c in use_cols if c in df.columns]
    df = df[existing_cols].copy()

    if "Date" not in df.columns:
        log_info(f"No 'Date' column found in columns: {list(df.columns)}. Returning empty DataFrame.")
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    full_range = pd.date_range(global_start, global_end, freq=freq)
    df = df.reindex(full_range)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df

def main():
    os.makedirs("output", exist_ok=True)
    log_info("Starting data merge with daily reindex approach...")

    global_start = "2020-01-01"
    global_end   = "2024-12-31"

    # 1. Load & fix each table
    df_stock_raw = load_and_fix_dates(STOCK_TABLE)
    df_fin_raw   = load_and_fix_dates(FINANCIALS_TABLE)
    df_macro_raw = load_and_fix_dates(MACRO_TABLE)

    # 2. Prepare & reindex each DataFrame
    df_stock = prepare_and_reindex(df_stock_raw, STOCK_COLS, global_start, global_end)
    df_fin   = prepare_and_reindex(df_fin_raw, FIN_COLS, global_start, global_end)
    df_macro = prepare_and_reindex(df_macro_raw, MACRO_COLS, global_start, global_end)

    # 3. Join on daily index
    df_merged = df_stock.join(df_fin, how="outer").join(df_macro, how="outer")
    df_merged = df_merged.loc[global_start:global_end]
    log_info(f"Data after final date filter: {df_merged.shape}")

    # 4. Replace zeros in investment columns with NaN, then backfill, then fill remaining with 0
    for col in ["purchase_of_investment", "sale_of_investment"]:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].replace(0, pd.NA)
            df_merged[col] = df_merged[col].bfill()
            df_merged[col] = df_merged[col].fillna(0)

    # 5. Store final merged data in the database with the index as the 'Date' column.
    try:
        conn = get_connection()
        df_merged.to_sql("master_data", conn, if_exists="replace", index=True, index_label="Date")
        conn.close()
        log_info("master_data table stored successfully in the database.")
    except Exception as e:
        log_error(f"Error storing master_data: {e}")
        raise

    df_merged.to_csv(OUTPUT_CSV)
    log_info(f"Final merged data saved as CSV in '{OUTPUT_CSV}'.")
    return df_merged

if __name__ == "__main__":
    master_df = main()
    print(master_df.head(10))
