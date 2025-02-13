import sqlite3

# Connect to SQLite database (creates if not exists)
conn = sqlite3.connect("database/blackstone_data.db")
cursor = conn.cursor()

# Read SQL schema from file
with open("database/db_setup.sql", "r") as sql_file:
    sql_script = sql_file.read()
    cursor.executescript(sql_script)

# Commit changes and close connection
conn.commit()
conn.close()

print("Database setup complete. Tables are ready for data insertion.")
