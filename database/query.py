## Query the database to inspect the columns and values inside the table
import sqlite3

# Connect to the SQL database
conn = sqlite3.connect('database/blackstone_data.db')  # Updated to your database path
cursor = conn.cursor()

# Inspect the bx_master_data table
cursor.execute("PRAGMA table_info(bx_master_data);")
columns = cursor.fetchall()

print("Columns in bx_master_data table:")
for col in columns:
    print(f"{col[1]} ({col[2]})")

conn.close()


## Inspect values inside the table columns
import sqlite3
import pandas as pd

DB_PATH = "database/blackstone_data.db"

# Inspect values inside the table columns
def inspect_column_values():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT Date, totalInvestments, netDebt, operatingCashFlow, retainedEarnings, CPI, [10Y Treasury Yield], Volatility_30 FROM bx_master_data LIMIT 10", conn)
    conn.close()
    print("Values in bx_master_data for selected columns:")
    print(df)

if __name__ == "__main__":
    inspect_column_values()
