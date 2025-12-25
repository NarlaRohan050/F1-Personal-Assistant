import streamlit as st
import ollama
import sqlite3
import pandas as pd
import fastf1
import fastf1.plotting
import os
import altair as alt
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import re

# ==========================================
# 1. APP CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="F1 AI Assistant", 
    page_icon="🏎️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Titillium Web', sans-serif; }
    .stApp { background-color: #0e0e12; color: #f0f0f0; }
    section[data-testid="stSidebar"] { background-color: #15151e; border-right: 2px solid #FF1801; }
    h1, h2, h3 { text-transform: uppercase; font-weight: 700; letter-spacing: 1px; }
    h1 { background: -webkit-linear-gradient(left, #FF1801, #ff5733); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    div.stButton > button { background-color: #FF1801; color: white; border: none; font-weight: 700; text-transform: uppercase; }
    div.stButton > button:hover { background-color: #cc1300; transform: scale(1.02); }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1f1f2e; color: #aaa; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #FF1801 !important; color: white !important; }
    div[data-testid="stMetric"] { background-color: #1f1f2e; border-left: 5px solid #FF1801; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    div[data-testid="stChatMessage"] { background-color: #1f1f2e; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# Setup FastF1
cache_dir = 'f1_cache'
if not os.path.exists(cache_dir):
    try: os.makedirs(cache_dir)
    except OSError: pass

try: 
    fastf1.Cache.enable_cache(cache_dir)
    fastf1.plotting.setup_mpl(color_scheme='fastf1')
except: pass

# ==========================================
# 2. SESSION STATE
# ==========================================
if "messages" not in st.session_state: st.session_state.messages = []
if "view_mode" not in st.session_state: st.session_state.view_mode = "chat"
if "manual_df" not in st.session_state: st.session_state.manual_df = pd.DataFrame()

# ==========================================
# 3. DATABASE ENGINE (Hybrid)
# ==========================================
def run_query(query):
    conn = sqlite3.connect('f1_knowledge_base.db')
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        conn.close()
        raise e
    finally:
        conn.close()

@st.cache_data
def get_sidebar_options():
    try:
        d = run_query("SELECT DISTINCT surname FROM drivers ORDER BY surname")
        y = run_query("SELECT DISTINCT year FROM races ORDER BY year DESC")
        return d['surname'].tolist(), y['year'].tolist()
    except: return [], []

# ==========================================
# 4. INTELLIGENCE LAYER (Router)
# ==========================================
def ask_f1_assistant(user_question):
    normalized_q = user_question.lower()
    
    # 1. "Most Wins" Template
    match_wins = re.search(r"(most wins|who won).*(\d{4})", normalized_q)
    if match_wins:
        year = match_wins.group(2)
        sql = f"""
        SELECT d.surname, ra.name as Race, ra.date
        FROM results r
        JOIN drivers d ON r.driverId = d.driverId
        JOIN races ra ON r.raceId = ra.raceId
        WHERE ra.year = {year} AND r.position = 1
        ORDER BY ra.date ASC
        """
        return sql, run_query(sql)

    # 2. "Most Poles" Template
    match_poles = re.search(r"(most poles|pole position).*(\d{4})", normalized_q)
    if match_poles:
        year = match_poles.group(2)
        sql = f"""
        SELECT d.surname, ra.name as Race, ra.date
        FROM results r
        JOIN drivers d ON r.driverId = d.driverId
        JOIN races ra ON r.raceId = ra.raceId
        WHERE ra.year = {year} AND r.grid = 1
        ORDER BY ra.date ASC
        """
        return sql, run_query(sql)

    # 3. Fallback AI
    def clean_filler(text):
        if "```sql" in text: text = text.split("```sql")[1].split("```")[0]
        elif "```" in text: text = text.split("```")[1].split("```")[0]
        text = text.replace(';', '')
        if "SELECT" in text.upper(): text = text[text.upper().find("SELECT"):]
        return text.strip()

    def generate_sql(question, error_context=None):
        system_prompt = """
        You are a strict F1 SQL Expert. 
        RULES: 
        1. Schema: drivers(driverId, surname), races(raceId, year, name, date), results(resultId, raceId, driverId, position).
        2. Aliases: drivers->d, races->ra, results->r.
        3. Logic: WINS = (r.position = 1). POLES = (r.grid = 1).
        4. Output: Raw SQL only.
        """
        messages = [{'role': 'system', 'content': system_prompt}]
        if error_context:
            messages.append({'role': 'user', 'content': f"Fix Error: {error_context}. Question: {question}"})
        else:
            messages.append({'role': 'user', 'content': question})
        
        response = ollama.chat(model='llama3', messages=messages)
        return clean_filler(response['message']['content'])

    try:
        sql = generate_sql(user_question)
        sql = sql.replace("driver_id", "driverId").replace("race_id", "raceId").replace("result_id", "resultId")
        sql = sql.replace("r.name", "ra.name").replace("r.date", "ra.date").replace("r.year", "ra.year")
        if "d.name" in sql: sql = sql.replace("d.name", "d.surname")
        df = run_query(sql)
        return sql, df
    except Exception as e:
        try:
            fixed_sql = generate_sql(user_question, str(e))
            df = run_query(fixed_sql)
            return fixed_sql, df
        except Exception as final_error:
            raise final_error

# ==========================================
# 5. RIVALRY & TELEMETRY
# ==========================================
def get_rivalry_stats(driver1, driver2, year):
    sql_stats = f"""
    SELECT d.surname,
        COUNT(CASE WHEN r.position = 1 THEN 1 END) as Wins,
        COUNT(CASE WHEN r.position <= 3 THEN 1 END) as Podiums,
        COUNT(CASE WHEN r.grid = 1 THEN 1 END) as Poles,
        SUM(r.points) as TotalPoints
    FROM results r
    JOIN drivers d ON r.driverId = d.driverId
    JOIN races ra ON r.raceId = ra.raceId
    WHERE ra.year = {year} AND d.surname IN ('{driver1}', '{driver2}')
    GROUP BY d.surname
    """
    df_stats = run_query(sql_stats)
    
    sql_prog = f"""
    SELECT ra.date, ra.name as Race, d.surname, r.points
    FROM results r
    JOIN drivers d ON r.driverId = d.driverId
    JOIN races ra ON r.raceId = ra.raceId
    WHERE ra.year = {year} AND d.surname IN ('{driver1}', '{driver2}')
    ORDER BY ra.date ASC
    """
    df_prog = run_query(sql_prog)
    df_prog['Cumulative Points'] = df_prog.groupby('surname')['points'].cumsum()
    return df_stats, df_prog

def plot_track_map(year, gp, session_type, driver):
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, laps=True, weather=False)
    lap = session.laps.pick_drivers(driver).pick_fastest()
    x, y, color = lap.telemetry['X'], lap.telemetry['Y'], lap.telemetry['Speed']
    points = np.array([x.values, y.values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'{gp.upper()} {year} | {driver}', size=20, color='#FF1801', weight='bold')
    ax.axis('off')
    
    norm = plt.Normalize(color.min(), color.max())
    lc = LineCollection(segments, cmap='plasma', norm=norm, linestyle='-', linewidth=5)
    lc.set_array(color)
    line = ax.add_collection(lc)
    ax.set_xlim(x.min()-500, x.max()+500)
    ax.set_ylim(y.min()-500, y.max()+500)
    cbar = fig.colorbar(line, ax=ax, shrink=0.8)
    cbar.set_label('Speed (km/h)', size=12, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    return fig

def get_telemetry_trace(year, gp, session_type, driver_code):
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, laps=True, weather=False)
    fastest_lap = session.laps.pick_drivers(driver_code).pick_fastest()
    return fastest_lap.get_car_data().add_distance()

# ==========================================
# 6. UI LAYOUT
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/1200px-F1.svg.png", width=100)
    st.title("F1 CONTROL")
    d_list, y_list = get_sidebar_options()
    st.markdown("### QUICK ACTIONS")
    if st.button("ACCESS DATA VAULT", type="primary"):
        sql = "SELECT ra.year, ra.name, d.surname, r.position FROM results r JOIN drivers d ON r.driverId = d.driverId JOIN races ra ON r.raceId = ra.raceId WHERE 1=1 ORDER BY ra.date DESC LIMIT 100"
        st.session_state.manual_df = run_query(sql)
        st.session_state.view_mode = "manual"

st.title("🏎️ F1 ASSISTANT")
st.caption("AI-POWERED STRATEGY & TELEMETRY SYSTEM")

tab_chat, tab_rivalry, tab_track, tab_telemetry = st.tabs(["💬 AI ENGINEER", "⚔️ RIVALRY BATTLE", "🗺️ TRACK DOMINANCE", "📊 TELEMETRY LAB"])

with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "df" in msg:
                df = msg["df"]
                st.dataframe(df, width=1000) 
                if not df.empty:
                    # --- FIXED VISUALIZATION LOGIC ---
                    # Priority: Check for 'surname' or 'driver' column first
                    target_col = None
                    if 'surname' in df.columns:
                        target_col = 'surname'
                    elif 'driver' in df.columns:
                        target_col = 'driver'
                    else:
                        # Fallback to the first text column that IS NOT 'date' or 'Race'
                        text_cols = df.select_dtypes(include=['object']).columns
                        for col in text_cols:
                            if col.lower() not in ['date', 'race', 'name']:
                                target_col = col
                                break
                    
                    if target_col:
                        counts = df[target_col].value_counts()
                        top_val = counts.idxmax()
                        top_count = counts.max()
                        st.metric(label=f"🏆 TOP RESULT", value=f"{top_val} ({top_count})")
                        st.bar_chart(counts, color="#FF1801")
                    else:
                        # If we really can't find a driver column, just show row count
                        st.metric(label="TOTAL RECORDS", value=len(df))

    if prompt := st.chat_input("Ask Race Control (e.g., 'Who had the most wins in 2021?')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        try:
            with st.spinner("CALCULATING STRATEGY..."):
                sql, df = ask_f1_assistant(prompt)
                if df.empty: st.warning("NO DATA FOUND.")
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "DATA RETRIEVED.", "df": df})
                    st.rerun()
        except Exception as e: st.error(f"SYSTEM ERROR: {e}")

with tab_rivalry:
    st.subheader("⚔️ DRIVER DUEL")
    c1, c2, c3 = st.columns(3)
    r_driver1 = c1.selectbox("DRIVER A", d_list, index=d_list.index('Verstappen') if 'Verstappen' in d_list else 0)
    r_driver2 = c2.selectbox("DRIVER B", d_list, index=d_list.index('Hamilton') if 'Hamilton' in d_list else 1)
    r_year = c3.selectbox("SEASON", y_list, index=y_list.index(2021) if 2021 in y_list else 0)
    if st.button("INITIATE BATTLE ANALYSIS", type="primary"):
        stats_df, prog_df = get_rivalry_stats(r_driver1, r_driver2, r_year)
        st.markdown("#### 🥊 TALE OF THE TAPE")
        d1 = stats_df[stats_df['surname'] == r_driver1]
        d2 = stats_df[stats_df['surname'] == r_driver2]
        if not d1.empty and not d2.empty:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("WINS", f"{d1['Wins'].sum()} vs {d2['Wins'].sum()}")
            m2.metric("PODIUMS", f"{d1['Podiums'].sum()} vs {d2['Podiums'].sum()}")
            m3.metric("POLES", f"{d1['Poles'].sum()} vs {d2['Poles'].sum()}")
            m4.metric("POINTS", f"{d1['TotalPoints'].sum()} vs {d2['TotalPoints'].sum()}")
            st.markdown("#### 📈 CHAMPIONSHIP GAP")
            chart = alt.Chart(prog_df).mark_line(point=True).encode(
                x='date:T', y='Cumulative Points:Q', color=alt.Color('surname:N', scale=alt.Scale(range=['#FF1801', '#00D2BE'])),
                tooltip=['Race', 'surname', 'Cumulative Points']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

with tab_track:
    st.subheader("🗺️ SPEED HEATMAP")
    t1, t2, t3 = st.columns(3)
    map_year = t1.number_input("YEAR", 2018, 2024, 2023)
    map_gp = t2.text_input("GRAND PRIX", "Monza")
    map_driver = t3.text_input("DRIVER", "VER").upper()
    if st.button("GENERATE MAP", type="primary"):
        with st.spinner(f"BUILDING CIRCUIT MODEL..."):
            try:
                fig = plot_track_map(map_year, map_gp, 'Q', map_driver)
                st.pyplot(fig)
            except Exception as e: st.error(f"MAP ERROR: {e}")

with tab_telemetry:
    st.subheader("📊 TELEMETRY TRACE")
    c1, c2, c3 = st.columns(3)
    t_year = c1.number_input("Year", 2018, 2024, 2024)
    t_gp = c2.text_input("Circuit", "Silverstone")
    t_driver = c3.text_input("Driver", "NOR").upper()
    if st.button("LOAD TRACE"):
        with st.spinner("DOWNLOADING TELEMETRY..."):
            try:
                tel_df = get_telemetry_trace(t_year, t_gp, 'Q', t_driver)
                st.line_chart(tel_df, x='Distance', y='Speed')
                st.line_chart(tel_df, x='Distance', y=['Throttle', 'Brake'])
            except Exception as e: st.error(f"Error: {e}")

if st.session_state.view_mode == "manual" and not st.session_state.manual_df.empty:
    st.divider()
    st.subheader("DATA VAULT")
    st.dataframe(st.session_state.manual_df, width=1500)
    if st.button("CLOSE VAULT"):
        st.session_state.view_mode = "chat"
        st.rerun()