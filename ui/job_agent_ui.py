import os
import sys
import tempfile

import pandas as pd
import streamlit as st

# ✅ Import necessary modules from LangChain for memory management
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage

# Add project root for imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from modules.job_scraper import JobScraper
from modules.matcher import JobMatcher
from modules.qa_agent import GeminiLLM
from modules.resume_parser import ResumeParser
from modules.vector_store import Vectorizer

# --- Streamlit UI ---
st.set_page_config(page_title="AI Job Agent", layout="wide")
st.title("🤖 AI Job Agent")
st.write(
    "Upload your resume, search for jobs, and interactively ask questions about them!"
)

# --- API Key Configuration ---
st.sidebar.header("🔑 API Configuration")
api_key = st.sidebar.text_input(
    "Enter your Google AI API Key:",
    type="password",
    help="Get your API key from https://makersuite.google.com/app/apikey",
)

if not api_key:
    st.warning(
        "⚠️ Please enter your Google AI API key in the sidebar to enable Q&A functionality."
    )
    st.info(
        "📌 To get your API key:\n1. Go to https://makersuite.google.com/app/apikey\n2. Create a new API key\n3. Copy and paste it in the sidebar"
    )

# --- Sidebar Inputs ---
st.sidebar.header("Job Search Settings")
job_query = st.sidebar.text_input("Job search query", value="AI Engineer")
job_location = st.sidebar.text_input("Location", value="Ho Chi Minh City")
num_results = st.sidebar.slider("Number of jobs to fetch", 1, 20, 5)

# --- Session State Variables ---
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "ranked_jobs" not in st.session_state:
    st.session_state.ranked_jobs = None
if "selected_job_index" not in st.session_state:
    st.session_state.selected_job_index = None
# ✅ Initialize LangChain ConversationBufferMemory in session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "last_error" not in st.session_state:
    st.session_state.last_error = None

uploaded_resume = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_resume and st.session_state.resume_text is None:
    # Parse resume only once with better error handling
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_resume.read())
            tmp_path = tmp.name

        st.success("✅ Resume uploaded successfully!")
        with st.spinner("Parsing resume..."):
            resume_parser = ResumeParser(tmp_path)
            st.session_state.resume_text = resume_parser.extract_text()

        # Validate resume text was extracted
        if (
            not st.session_state.resume_text
            or st.session_state.resume_text.strip() == ""
        ):
            st.error(
                "❌ Could not extract text from the PDF. Please ensure it's a valid PDF with readable text."
            )
            st.session_state.resume_text = None
        else:
            st.success(
                f"✅ Resume parsed successfully! Extracted {len(st.session_state.resume_text)} characters."
            )

        # Remove temp file but keep text in session
        os.remove(tmp_path)

    except Exception as e:
        st.error(f"❌ Error processing resume: {str(e)}")
        st.session_state.resume_text = None
        # Clean up temp file if it exists
        try:
            if "tmp_path" in locals():
                os.remove(tmp_path)
        except:
            pass

# If resume parsed successfully, enable job search
if st.session_state.resume_text:
    if st.button("🔍 Find Matching Jobs"):
        try:
            with st.spinner("Scraping job listings..."):
                scraper = JobScraper(
                    search_term=job_query,
                    location=job_location,
                    num_results=num_results,
                )
                df_jobs = scraper.scrape()

            # Fallback if scraping returns nothing
            if df_jobs is None or df_jobs.empty:
                st.warning("⚠️ No jobs found! Using sample dummy jobs instead.")
                df_jobs = pd.DataFrame(
                    [
                        {
                            "title": "AI Engineer",
                            "company": "OpenAI",
                            "location": "Remote",
                            "description": "Work on cutting-edge AI models and deployment. Develop and optimize large language models, implement distributed training systems, and collaborate with research teams to bring AI innovations to production. Required: Python, PyTorch, distributed systems experience.",
                            "job_url": "https://example.com/job1",
                        },
                        {
                            "title": "Machine Learning Engineer",
                            "company": "Google DeepMind",
                            "location": "London, UK",
                            "description": "Build ML pipelines, optimize deep learning models. Design and implement scalable machine learning infrastructure, develop model training pipelines, and work on computer vision and NLP applications. Required: TensorFlow, Kubernetes, MLOps experience.",
                            "job_url": "https://example.com/job2",
                        },
                        {
                            "title": "Data Scientist",
                            "company": "Meta AI",
                            "location": "Remote",
                            "description": "Analyze large datasets, create AI-powered insights. Develop predictive models, perform statistical analysis, and build recommendation systems. Work with product teams to implement data-driven solutions. Required: Python, SQL, statistics, A/B testing.",
                            "job_url": "https://example.com/job3",
                        },
                    ]
                )

            with st.spinner("Generating embeddings & ranking matches..."):
                vectorizer = Vectorizer()
                resume_emb = vectorizer.embed_text(st.session_state.resume_text)
                job_descriptions = df_jobs["description"].tolist()

                # Better validation of job descriptions
                job_descriptions = [
                    desc for desc in job_descriptions if desc and str(desc).strip()
                ]

                if not job_descriptions:
                    st.error("❌ No valid job descriptions found. Cannot rank jobs.")
                else:
                    job_embs = vectorizer.embed_texts(job_descriptions)

                    if len(job_embs) == 0:
                        st.error("❌ No job embeddings generated. Cannot rank jobs.")
                    else:
                        matcher = JobMatcher()
                        ranked_jobs = matcher.rank_jobs(
                            resume_emb, job_embs, df_jobs, top_k=min(5, len(df_jobs))
                        )
                        st.session_state.ranked_jobs = ranked_jobs
                        st.success(f"✅ Found and ranked {len(ranked_jobs)} jobs!")

        except Exception as e:
            st.error(f"❌ Error during job search: {str(e)}")
            st.session_state.last_error = str(e)

# Display ranked jobs if available
if st.session_state.ranked_jobs is not None and not st.session_state.ranked_jobs.empty:
    st.subheader("📌 Top Matching Jobs")

    # Ensure required columns exist before displaying
    display_columns = ["title", "company", "location", "similarity"]
    if "job_url" in st.session_state.ranked_jobs.columns:
        display_columns.append("job_url")

    st.dataframe(st.session_state.ranked_jobs[display_columns])

    job_index = st.selectbox(
        "Select a job for Q&A",
        st.session_state.ranked_jobs.index,
        format_func=lambda i: f"{st.session_state.ranked_jobs.loc[i, 'title']} at {st.session_state.ranked_jobs.loc[i, 'company']} (Similarity: {st.session_state.ranked_jobs.loc[i, 'similarity']:.2%})",
    )

    # Store selected job index for chat context
    if job_index is not None and job_index != st.session_state.selected_job_index:
        st.session_state.selected_job_index = job_index
        # ✅ Clear memory when switching jobs
        st.session_state.memory.clear()

    if job_index is not None:
        selected_job_desc = st.session_state.ranked_jobs.loc[job_index, "description"]
        selected_job_title = st.session_state.ranked_jobs.loc[job_index, "title"]
        selected_job_company = st.session_state.ranked_jobs.loc[job_index, "company"]

        st.subheader(f"💬 Chat about: {selected_job_title} at {selected_job_company}")

        # Example questions to copy
        st.markdown("**💡 Example questions you can ask:**")
        example_questions = [
            "What are the key responsibilities for this role?",
            "How well does my resume match this job?",
            "What skills should I highlight in my application?",
            "What experience gaps do I have for this position?",
            "Compare this job with the other scraped positions",
            "What questions should I prepare for the interview?",
        ]

        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(
                    f"📋 {question[:30]}...",
                    key=f"copy_{i}",
                    help=f"Click to copy: {question}",
                ):
                    st.session_state.temp_question = question

        # Chat interface
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask anything about this job, your resume match, or compare with other jobs:",
                value=st.session_state.get("temp_question", ""),
                placeholder="e.g., How well does my background match this role?",
            )
            send_button = st.form_submit_button("Send 💬")

        # Clear temp question after displaying
        if "temp_question" in st.session_state:
            del st.session_state.temp_question

        if send_button and user_input.strip():
            if not api_key:
                st.error(
                    "❌ Please enter your Google AI API key in the sidebar to use chat functionality."
                )
            else:
                with st.spinner("Thinking..."):
                    try:
                        llm = GeminiLLM(api_key=api_key)

                        # ✅ Load conversation history from memory
                        try:
                            memory_context = (
                                st.session_state.memory.load_memory_variables({})[
                                    "history"
                                ]
                            )
                        except KeyError:
                            memory_context = []

                        # Build context with better formatting
                        context = f"""
CONTEXT INFORMATION:

USER'S RESUME:
{st.session_state.resume_text[:2000]}{'...' if len(st.session_state.resume_text) > 2000 else ''}

SELECTED JOB:
Title: {selected_job_title}
Company: {selected_job_company}
Description: {selected_job_desc}

ALL SCRAPED JOBS:
{chr(10).join([f'{i+1}. {row["title"]} at {row["company"]} - {str(row["description"])[:200]}...' 
              for i, row in st.session_state.ranked_jobs.iterrows()])}

CONVERSATION HISTORY:
{memory_context if memory_context else 'No previous conversation.'}

CURRENT QUESTION: {user_input}

Instructions: Answer the user's question using the context provided. You can reference the resume, the selected job, other scraped jobs, and previous conversation. Be specific and helpful.
"""

                        answer = llm.answer_question(context)

                        # Validate answer before saving
                        if answer and answer.strip():
                            # ✅ Save the current exchange to memory
                            st.session_state.memory.save_context(
                                {"input": user_input}, {"output": answer}
                            )
                            # Rerun to display the updated conversation
                            st.rerun()
                        else:
                            st.error(
                                "❌ Received empty response from AI. Please try again."
                            )

                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.last_error = str(e)

        # ✅ Simple chat display - show conversation if it exists
        if st.session_state.memory.chat_memory.messages:
            st.markdown("### 💬 Conversation")

            # Display messages in chronological order (oldest first)
            for message in st.session_state.memory.chat_memory.messages:
                if isinstance(message, HumanMessage):
                    st.markdown(f"**🧑 You:** {message.content}")
                elif isinstance(message, AIMessage):
                    st.markdown(f"**🤖 AI:** {message.content}")
                    st.markdown("---")

            # ✅ Simple clear chat button
            if st.button("🧹 Clear Conversation"):
                st.session_state.memory.clear()
                st.rerun()

# Show helpful messages based on state
elif st.session_state.ranked_jobs is not None and st.session_state.ranked_jobs.empty:
    st.warning("⚠️ No jobs were found or ranked. Please try different search terms.")
else:
    if not uploaded_resume:
        st.info("📄 Please upload your resume to start finding matching jobs.")
    elif not st.session_state.resume_text:
        st.info(
            "🔄 Please wait for resume processing to complete, then search for jobs."
        )

# Display error information if needed
if st.session_state.last_error:
    with st.expander("🔧 Debug Information"):
        st.code(st.session_state.last_error)
        if st.button("Clear Error"):
            st.session_state.last_error = None
            st.rerun()

# --- Additional Information ---
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown(
    """
**AI Job Agent** helps you:
- Parse your resume  
- Find matching job opportunities
- Chat about specific jobs with AI
- Compare jobs and get match analysis
"""
)

st.sidebar.markdown("### 🔧 Model Info")
st.sidebar.markdown(
    """
**LLM:** Gemini Pro
**Provider:** Google AI
**Features:** Advanced reasoning, job analysis
"""
)

# Add system status in sidebar
st.sidebar.markdown("### 📊 System Status")
status_items = []
if st.session_state.resume_text:
    status_items.append("✅ Resume loaded")
else:
    status_items.append("❌ No resume")

if st.session_state.ranked_jobs is not None and not st.session_state.ranked_jobs.empty:
    status_items.append(f"✅ {len(st.session_state.ranked_jobs)} jobs ranked")
else:
    status_items.append("❌ No jobs found")

if api_key:
    status_items.append("✅ API key configured")
else:
    status_items.append("❌ No API key")

for item in status_items:
    st.sidebar.markdown(item)
