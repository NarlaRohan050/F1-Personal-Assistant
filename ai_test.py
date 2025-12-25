import ollama

# We are giving the AI the "Map" of our library so it knows what tables exist
system_prompt = """
You are an F1 Data Expert. Your job is to turn user questions into SQL queries for a database with these tables:
- drivers (driverid, forename, surname, nationality)
- results (resultid, raceid, driverid, constructorid, position)
- status (statusid, status)

Output ONLY the SQL query. No explanation.
"""

user_question = "Who are the top 5 drivers from Brazil?"

response = ollama.chat(model='llama3', messages=[
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': user_question},
])

print(f"🤖 AI Generated SQL:\n{response['message']['content']}")