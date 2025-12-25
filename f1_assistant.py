import ollama
import sqlite3
import pandas as pd

def ask_f1_assistant(user_query):
    # 1. The Blueprint: We tell the AI exactly how the library is organized
    system_instructions = """
    You are an F1 Data SQL Expert. Use these EXACT table and column names from the Ergast database:
    
    - drivers: driverId, forename, surname, nationality
    - races: raceId, year, name, date
    - results: resultId, raceId, driverId, constructorId, position, points
    - constructors: constructorId, name, nationality
    - status: statusId, status

    CRITICAL RULES:
    1. For 'Wins', use: results.position = 1
    2. To get driver names, JOIN 'results' and 'drivers' ON 'driverId'
    3. To filter by year, JOIN 'results' and 'races' ON 'raceId'
    4. Use 'driverId', NOT 'driver_id'. Use 'raceId', NOT 'race_id'.
    5. Output ONLY the raw SQL. No markdown, no backticks, no '```sql'.
    """

    try:
        # 2. Ask the Brain for the SQL
        response = ollama.chat(model='llama3', messages=[
            {'role': 'system', 'content': system_instructions},
            {'role': 'user', 'content': user_query},
        ])
        
        # Clean the AI output to ensure it's just raw SQL
        sql_query = response['message']['content'].strip()
        if sql_query.startswith("```"):
            sql_query = sql_query.split("\n", 1)[1].rsplit("\n", 1)[0]
        
        # 3. Connect to your 'Vault' and run the query
        conn = sqlite3.connect('f1_knowledge_base.db')
        result_df = pd.read_sql_query(sql_query, conn)
        conn.close()
        
        return sql_query, result_df

    except Exception as e:
        return "SQL Generation Error", f"Error: {e}"

# --- Interactive Test Loop ---
if __name__ == "__main__":
    print("🏎️ F1 Offline Assistant Active (Type 'exit' to quit)")
    while True:
        query = input("\nAsk an F1 question: ")
        if query.lower() == 'exit':
            break
            
        sql, data = ask_f1_assistant(query)
        
        print(f"\n[AI Thought - SQL]:\n{sql}")
        print("\n[Result]:")
        if isinstance(data, pd.DataFrame):
            if data.empty:
                print("No records found.")
            else:
                print(data.to_string(index=False))
        else:
            print(data)