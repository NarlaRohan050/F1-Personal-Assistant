import sqlite3
import pandas as pd

# This is just to check if the library is working
conn = sqlite3.connect('f1_knowledge_base.db')
# We will just look at the first 5 names in your 'drivers' table
df = pd.read_sql_query("SELECT forename, surname FROM drivers LIMIT 5", conn)
print("--- Database Connection Test ---")
print(df)
conn.close()