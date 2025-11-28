import streamlit as st
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="IELTS Writing RAG Assistant",
    page_icon="📘",
    layout="wide"
)

# ----------------------------
# Load Index, Texts, Metadata
# ----------------------------
@st.cache_resource
def load_index_and_texts():
    index = faiss.read_index("index.faiss")
    with open("texts.pkl", "rb") as f:
        texts = pickle.load(f)
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, texts, metadata

@st.cache_resource
def load_encoder():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

index, texts, metadata = load_index_and_texts()
sbert_model = load_encoder()

# ----------------------------
# GEMINI CONFIG
# ----------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.0-flash")


# ----------------------------
# Helper Functions
# ----------------------------
def cosine_sim(a, b):
    return float(np.dot(a, b)) / (float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8)


def retrieve_context(query, k_final: int = 7):
    if not query.strip():
        return []

    query_vec = sbert_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, max(k_final * 2, 15))

    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(texts) and texts[idx].strip():
            candidates.append((dist, idx, texts[idx]))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0])

    selected, embeddings = [], []
    for dist, idx, text in candidates:
        emb = sbert_model.encode([text], convert_to_numpy=True).astype("float32")[0]

        if any(cosine_sim(emb, prev) > 0.9 for prev in embeddings):
            continue

        selected.append(text)
        embeddings.append(emb)

        if len(selected) >= k_final:
            break

    return selected


def build_prompt(query, retrieved):
    if not retrieved:
        return f"""
I can only provide a partial answer based on the available context.
QUESTION: {query}
"""

    context = "\n".join([f"- {item}" for item in retrieved])

    return f"""
You are an IELTS Writing Tutor. Use ONLY the context.

========================================
### CONTEXT
{context}
========================================

### QUESTION
{query}

========================================
STRICT FORMAT:
1. Introduction — brief explanation of problem (NO title "Introduction")
2. 4–6 bullet points (each 1–2 full sentences, ONE idea each, spacing exactly as format)
3. Summary (NO repeated idea or synonym reuse)
"""


def rag_answer(query):
    retrieved = retrieve_context(query)
    prompt = build_prompt(query, retrieved)
    response = gemini_model.generate_content(prompt)
    return response.text


# ----------------------------
# CSS
# ----------------------------
st.markdown("""
<style>
html, body {
    font-family: 'Inter', sans-serif;
}

.header-gradient {
    background: white;
    padding: 30px;
    border-radius: 12px;
    color: black !important;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.12);
}

.user-bubble {
    background: #e8f1ff;
    padding: 12px 18px;
    color: #003366;
    border-radius: 14px;
    margin: 10px 0;
}

.bot-bubble {
    background: white;
    padding: 15px 18px;
    color: black;
    border-radius: 14px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# SIDEBAR PRESET QUESTIONS
# ----------------------------
with st.sidebar:
    st.header("📘 IELTS Writing Helper")

    categories = {
        "Coherence": [
            "How can I improve coherence?",
            "How do I ensure sentences connect smoothly?",
            "How do I avoid abrupt topic shifts?"
        ],
        "Cohesion": [
            "What linking words improve cohesion?",
            "How do I use cohesive devices properly?"
        ]
    }

    for cat, questions in categories.items():
        st.subheader(cat)
        for q in questions:
            if st.button(q, key=q):
                st.session_state["selected_question"] = q
                st.experimental_rerun()


# ----------------------------
# Chat Title
# ----------------------------
st.markdown("""
<div class='header-gradient'>
    <h1>IELTS Writing RAG Assistant</h1>
    <p>Ask about writing structure, coherence, grammar, or clarity.</p>
</div>
""", unsafe_allow_html=True)


# ----------------------------
# CHAT HISTORY
# ----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []


for msg in st.session_state["messages"]:
    role_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
    st.markdown(f"<div class='{role_class}'>{msg['content']}</div>", unsafe_allow_html=True)


# ----------------------------
# USER INPUT
# ----------------------------
default_value = st.session_state.get("selected_question", "")

user_query = st.text_input("", placeholder="Type your question here...", value=default_value)

send = st.button("Send")


if send and user_query.strip():
    st.session_state["messages"].append({"role": "user", "content": user_query})

    with st.spinner("Thinking..."):
        answer = rag_answer(user_query)

    st.session_state["messages"].append({"role": "assistant", "content": answer})

    st.session_state["selected_question"] = ""

    st.experimental_rerun()
