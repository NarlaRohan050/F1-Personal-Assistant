import streamlit as st
import ollama
import sqlite3
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="F1 AI Assistant", page_icon="🏎️", layout="wide")
st.title("🏎️ F1 Knowledge & Insights Assistant")
st.markdown("---")

# 2. The Logic Function
def ask_f1_assistant(user_question):
    system_prompt = """
    You are an F1 Data Expert. Convert the user's question into a single SQL query.
    
    Database Tables: 
    - drivers (driverId, forename, surname, nationality)
    - races (raceId, year, name)
    - results (resultId, raceId, driverId, position, points)
    
    LOGIC RULES:
    1. A 'WIN' only counts if results.position = 1.
    2. A 'RACE START' counts every row in the results table for that driver.
    3. Output ONLY raw SQL. No conversational text. No backticks.
    """
    
    response = ollama.chat(model='llama3', messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_question},
    ])
    
    # --- POWERFUL CLEANING LOGIC ---
    raw_query = response['message']['content'].strip()
    
    # 1. Remove markdown code fences
    clean_sql = raw_query.replace('```sql', '').replace('```', '').strip()
    
    # 2. Extract ONLY the SQL (removes "Here is the query..." text)
    if "SELECT" in clean_sql.upper():
        # This finds the first 'SELECT' and keeps everything from that point forward
        start_index = clean_sql.upper().find("SELECT")
        clean_sql = clean_sql[start_index:]
    
    # 3. Final polish
    clean_sql = clean_sql.replace(';', '').strip()
    # -------------------------------
    
    conn = sqlite3.connect('f1_knowledge_base.db')
    df = pd.read_sql_query(clean_sql, conn)
    conn.close()
    
    return clean_sql, df

# 3. User Interface
user_input = st.chat_input("Ask a question (e.g., 'Top 5 drivers with most wins')")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    try:
        sql_used, result_df = ask_f1_assistant(user_input)
        
        with st.chat_message("assistant"):
            if result_df.empty:
                st.info("Query successful, but no matching data found.")
            else:
                st.success("Analysis Complete!")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**Data Table**")
                    st.dataframe(result_df, use_container_width=True)
                
                with col2:
                    numeric_cols = result_df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        st.write("**Visual Insight**")
                        st.bar_chart(result_df, x=result_df.columns[0], y=numeric_cols[0], color="#FF1801")

                with st.expander("See technical SQL query"):
                    st.code(sql_used, language='sql')
                    
    except Exception as e:
        # If there's still an error, we show the cleaned SQL to see what's wrong
        st.error(f"Logic Error: {e}")
        st.write("Cleaned SQL that failed:")
        st.code(sql_used if 'sql_used' in locals() else "N/A")