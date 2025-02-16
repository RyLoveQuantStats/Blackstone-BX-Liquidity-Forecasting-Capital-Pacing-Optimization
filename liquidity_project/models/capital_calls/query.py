import sqlite3

# Connect to the database
conn = sqlite3.connect('database/data.db')

# Create a cursor object
cursor = conn.cursor()

# Query to get column names from the bx_master_data table
cursor.execute("PRAGMA table_info(master_data);")

# Fetch and display the results
columns = cursor.fetchall()

for col in columns:
    print(col)

# Close the connection
conn.close()



###
# 
from utils.db_utils import fetch_data
query = """
SELECT 
    Date,
    fin_cash_and_cash_equivalents,
    fin_net_debt,
    fin_total_assets,
    fin_total_liabilities_net_minority_interest,
    fin_operating_cash_flow,
    fin_investing_cash_flow,
    fin_financing_cash_flow,
    fin_free_cash_flow,
    fin_investments_in_property_plant_and_equipment,
    fin_purchase_of_investment,
    fin_sale_of_investment,
    capital_calls,
    "macro_10Y Treasury Yield",
    macro_CPI,
    "macro_BAA Corporate Bond Spread"
FROM master_data
LIMIT 10;
"""

df = fetch_data(query)
print(df)


