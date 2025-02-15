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




#!/usr/bin/env python3
"""
print_db_schema.py

This script connects to the Blackstone data SQLite database and prints out all the table names
and their column details (name, type, not null flag, default value, primary key flag).
"""

import sqlite3


def list_tables_and_columns(db_path):
    """Prints all tables and column details from the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Retrieve all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("No tables found in the database.")
            return
        
        # Loop through each table and print its columns
        for table in tables:
            table_name = table[0]
            print(f"\nTable: {table_name}")
            print("-" * (len(table_name) + 7))
            
            # Retrieve column info using PRAGMA table_info
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            if not columns:
                print("  No columns found.")
            else:
                for column in columns:
                    col_id = column[0]
                    col_name = column[1]
                    col_type = column[2]
                    notnull = column[3]
                    default_val = column[4]
                    pk = column[5]
                    print(f"  Column: {col_name} | Type: {col_type} | NotNull: {notnull} | Default: {default_val} | Primary Key: {pk}")
        
        conn.close()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    list_tables_and_columns(DB_PATH)
