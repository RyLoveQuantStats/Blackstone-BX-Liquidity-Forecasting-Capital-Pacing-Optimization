import sqlite3
import pandas as pd

# Connect to database and verify data
conn = sqlite3.connect("database/blackstone_data.db")

# Read data from SQLite into a Pandas DataFrame
df = pd.read_sql("SELECT * FROM bx_financials LIMIT 5;", conn)

# Close connection
conn.close()

# Display first few rows
print(df)
