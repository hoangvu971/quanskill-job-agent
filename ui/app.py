import streamlit as st

# --- Sidebar Toggle ---
st.set_page_config(page_title="AI Tools", layout="wide")
st.sidebar.title("🛠 AI Tools")
app_mode = st.sidebar.radio("Choose an app:", ["AI Job Agent", "Simple Chatbot"])

def run_script(path):
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
        exec(code, globals())

# --- Route to Selected App ---
if app_mode == "AI Job Agent":
    st.title("🤖 AI Job Agent")
    st.write("This tool lets you upload your resume, fetch matching jobs, and ask questions.")
    run_script("ui/job_agent_ui.py")

elif app_mode == "Simple Chatbot":
    st.title("💬 Simple Chatbot")
    st.write("This is a plain chatbot for free-form conversation.")
    run_script("ui/chatbot_ui.py")
