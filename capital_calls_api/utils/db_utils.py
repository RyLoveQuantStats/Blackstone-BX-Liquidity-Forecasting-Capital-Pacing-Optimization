import sqlite3
import pandas as pd

DB_PATH = r"C:\Users\ryanl\OneDrive\Desktop\Programming Apps\Python\python_work\CAPACE\database\data.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def store_dataframe(df, table_name, if_exists="replace"):
    conn = get_connection()
    df.to_sql(table_name, conn, if_exists=if_exists, index=True)
    conn.close()
    print(f"✅ Data stored in table '{table_name}'.")

def fetch_data(query):
    conn = get_connection()
    result = pd.read_sql(query, conn)
    conn.close()
    return result
