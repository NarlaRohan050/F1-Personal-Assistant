import pandas as pd
import sqlite3
import os

# Where your CSVs live
raw_folder = 'raw_data'
# Where your new database will be saved
db_name = 'f1_knowledge_base.db'

conn = sqlite3.connect(db_name)

# This loop reads every CSV and saves it as a table in the database
for file in os.listdir(raw_folder):
    if file.endswith('.csv'):
        table_name = file.replace('.csv', '')
        # Loading CSV and converting '\N' (F1 specific nulls) to actual empty values
        df = pd.read_csv(os.path.join(raw_folder, file), na_values=r'\N')
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"✅ Successfully added {table_name} to the library.")

conn.close()
print("\n🏁 Your Offline F1 Library is ready!")