import streamlit as st
import plotly.express as px
import pandas as pd
import time
from integrated_system import AdaptiveLLMSystem

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Adaptive AI Platform",
    page_icon="🤖",
    layout="wide"
)

# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #111827);
    color: white;
}

/* Titles */
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #60A5FA;
}

.sub-title {
    font-size: 18px;
    color: #9CA3AF;
    margin-bottom: 20px;
}

/* Response Box */
.response-box {
    background: #1E293B;
    padding: 25px;
    border-radius: 15px;
    border-left: 5px solid #3B82F6;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    font-size: 16px;
}

/* Success */
.success-box {
    background: #065F46;
    padding: 12px;
    border-radius: 10px;
    font-weight: 500;
}

/* Warning */
.warning-box {
    background: #7F1D1D;
    padding: 12px;
    border-radius: 10px;
    font-weight: 500;
}

/* Buttons */
.stButton>button {
    background: #2563EB;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
}

.stButton>button:hover {
    background: #1D4ED8;
    color: white;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: 24px;
    font-weight: 700;
    color: #38BDF8;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD SYSTEM ----------------
@st.cache_resource
def load_system():
    documents = [
        "Retrieval-Augmented Generation (RAG) combines retrieval systems with large language models.",
        "BM25 is a ranking function used in information retrieval.",
        "Vector embeddings convert text into numerical representations.",
        "Cross-encoders score query-document pairs jointly."
    ]
    return AdaptiveLLMSystem(documents)

system = load_system()

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🚀 Adaptive AI Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hybrid Retrieval + Fine-Tuned LLM + Hallucination Detection</div>', unsafe_allow_html=True)

st.divider()

# ---------------- INPUT ----------------
query = st.text_input("Ask your question:")

if st.button("Run System") and query:

    start_time = time.time()
    result = system.run(query)
    total_time = round(time.time() - start_time, 3)

    st.divider()

    col1, col2 = st.columns([3,1])

    # ---------------- RESPONSE ----------------
    with col1:
        st.subheader("🤖 AI Response")
        st.markdown(f'<div class="response-box">{result["response"]}</div>', unsafe_allow_html=True)

    # ---------------- PERFORMANCE ----------------
    with col2:
        st.subheader("⚡ Performance")
        st.metric("Latency (sec)", total_time)

    st.divider()

    # ---------------- HALLUCINATION STATUS ----------------
    st.subheader("🧠 Hallucination Analysis")

    if result["hallucination"] == 1:
        st.markdown('<div class="warning-box">⚠️ Potential Hallucination Detected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">✅ Response Appears Grounded</div>', unsafe_allow_html=True)

    st.divider()

    # ---------------- RETRIEVAL SCORES ----------------
    st.subheader("📊 Retrieval Scores")

    docs = [doc for doc, score in result["retrieved_docs"]]
    scores = [score for doc, score in result["retrieved_docs"]]

    df = pd.DataFrame({
        "Document": docs,
        "Score": scores
    })

    fig = px.bar(
        df,
        x="Score",
        y="Document",
        orientation="h",
        title="Top Retrieved Documents",
        color="Score",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()