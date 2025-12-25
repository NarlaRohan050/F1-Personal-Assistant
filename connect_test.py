import ollama
import sqlite3
import pandas as pd

def ask_f1_assistant(user_question):
    # 1. The Instructions for the AI
    system_prompt = """
    You are an F1 Expert. Convert the question into SQL.
    Table: drivers 
    Columns: forename, surname, nationality
    
    IMPORTANT: 
    - Nationalities are adjectives (e.g., 'Brazilian' not 'Brazil', 'British' not 'UK').
    - Output ONLY the SQL. No text.
    """

    # 2. Get SQL from the AI
    response = ollama.chat(model='llama3', messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_question},
    ])
    sql_query = response['message']['content'].strip()

    # 3. Use that SQL to search your 'Vault'
    conn = sqlite3.connect('f1_knowledge_base.db')
    # This line runs the AI's SQL against your real data
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    
    return sql_query, df

# --- Let's Test the Connection ---
question = "Who are the drivers from Brazil?"
sql, data = ask_f1_assistant(question)

print(f"AI thought of this SQL: {sql}")
print("\nReal Data from your Vault:")
print(data)