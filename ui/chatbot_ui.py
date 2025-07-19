import os
import sys

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Add project root for consistency
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# ✅ Streamlit page config
st.set_page_config(page_title="Gemini Flash 2.0 Chatbot", layout="centered")
st.title("💬 Gemini Flash 2.0 Chatbot")
st.write(
    "This chatbot uses Google's Gemini Flash 2.0 model for intelligent conversations."
)

# ✅ API Key Input Section
st.sidebar.header("🔑 API Configuration")
api_key = st.sidebar.text_input(
    "Enter your Google AI API Key:",
    type="password",
    help="Get your API key from https://makersuite.google.com/app/apikey",
)

if not api_key:
    st.warning(
        "⚠️ Please enter your Google AI API key in the sidebar to start chatting."
    )
    st.info(
        "📌 To get your API key:\n1. Go to https://makersuite.google.com/app/apikey\n2. Create a new API key\n3. Copy and paste it in the sidebar"
    )
    st.stop()


# ✅ Initialize Gemini Flash 2.0 model
@st.cache_resource
def load_gemini_model(api_key):
    try:
        # Use the latest Gemini Flash 2.0 model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=1024,
        )
        return llm
    except Exception as e:
        st.error(f"Error loading Gemini model: {str(e)}")
        return None


# Load the model
llm = load_gemini_model(api_key)

if llm is None:
    st.error("❌ Failed to load Gemini model. Please check your API key.")
    st.stop()

# ✅ Maintain chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Chat Input ---
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask me anything:", "")
    send_button = st.form_submit_button("Send")

# --- On user input ---
if send_button and user_input.strip():
    # Save user message
    st.session_state.chat_history.append(("You", user_input))

    # Generate response
    with st.spinner("Thinking..."):
        try:
            response = llm.invoke(user_input)
            bot_reply = response.content
        except Exception as e:
            bot_reply = f"Sorry, I encountered an error: {str(e)}"

    # Save bot reply
    st.session_state.chat_history.append(("Bot", bot_reply))

# --- Display chat history ---
if st.session_state.chat_history:
    st.markdown("### 🗨️ Conversation")
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑 You:** {message}")
        else:
            st.markdown(f"**🤖 Bot:** {message}")

# --- Clear chat button ---
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# --- Additional Information ---
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown(
    """
**Model:** Gemini 2.0 Flash Experimental
**Provider:** Google AI
**Features:**
- Fast response times
- Advanced reasoning
- Multimodal capabilities
"""
)

st.sidebar.markdown("### 📖 Instructions")
st.sidebar.markdown(
    """
1. Enter your Google AI API key above
2. Type your message in the chat input
3. Press Send or hit Enter
4. Use 'Clear Chat' to reset conversation
"""
)
