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
# LOAD INDEX + MODEL
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
# CONFIGURE GEMINI
# ----------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.0-flash")


# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def cosine_sim(a, b):
    return float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def retrieve_context(query, k_final=7):
    if not query.strip():
        return []

    query_vec = sbert_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, max(k_final * 2, 15))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(texts):
            continue
        t = texts[idx].strip()
        if t:
            results.append((dist, t))
    results.sort(key=lambda x: x[0])

    selected, used = [], []
    for _, t in results:
        emb = sbert_model.encode([t], convert_to_numpy=True).astype("float32")[0]
        if any(cosine_sim(emb, u) > 0.9 for u in used):
            continue
        used.append(emb)
        selected.append(t)
        if len(selected) >= k_final:
            break

    return selected


def build_prompt(query, retrieved):
    if not retrieved:
        return f"""
No retrievable context found. 
Question: {query}

Respond:

"I can only provide a partial answer based on the available context."
"""

    context_text = "\n".join([f"- {c}" for c in retrieved])

    return f"""
You are an IELTS Writing Tutor. Use ONLY the information below.

========================================
CONTEXT:
{context_text}
========================================

QUESTION:
{query}


FORMAT TO FOLLOW:

1. (No word "Introduction") — Explain the relevant idea based ONLY on context. Do not repeat the conclusion. Do not rephrase the same idea twice.

2. Bullet points (STRICT FORMAT):
• Exactly 4–6 bullets.
• One space after bullet.
• Each bullet: 1–2 full sentences.
• One empty line between bullets.
• No repeated ideas.

Example format:

• First idea sentence.

• Second idea sentence.

3. (No word "Conclusion") — Summarize using NEW wording, no overlaps with introduction, and explain impact on writing quality.
"""


def rag_answer(query):
    retrieved = retrieve_context(query)
    response = gemini_model.generate_content(build_prompt(query, retrieved))
    return response.text


# ----------------------------
# UI STYLING
# ----------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.chat-container {
    max-width: 800px;
    margin: auto;
}

.user-bubble {
    background: #e8f1ff;
    padding: 12px 18px;
    border-radius: 14px;
    margin-bottom: 10px;
    text-align: right;
    color: #003366;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}

.bot-bubble {
    background: #ffffff;
    padding: 15px 18px;
    border-radius: 14px;
    margin-bottom: 10px;
    color: #1a1a1a;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}

.header-gradient {
    background: white;
    padding: 30px 10px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.12);
    margin-bottom: 25px;
}

.header-gradient * {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# SIDEBAR PRESETS
# ----------------------------
with st.sidebar:
    st.header("📘 IELTS Writing Helper")
    st.write("Click a question to auto-fill the chat.")

    categories = {
        "🧩 Coherence": [
            "How can I improve coherence?",
            "How do I ensure each sentence follows logically from the previous one?",
            "How can I improve logical flow of ideas?"
        ],
        "🔗 Cohesion": [
            "How can I use cohesive devices effectively?",
            "What linking words improve cohesion?"
        ],
        "📑 Paragraph Structure": [
            "How do I structure a paragraph?",
            "How can I organize a paragraph clearly?"
        ],
        "📝 Task Response": [
            "How do I choose good main ideas?",
            "How do I avoid technical arguments?"
        ],
        "📚 Vocabulary": [
            "How can I improve vocabulary?",
            "How do I avoid repeating words?"
        ],
        "⚙️ Grammar Range": [
            "How can I improve my grammatical range?",
            "How can I vary my sentence structure?"
        ]
    }

    for section, questions in categories.items():
        st.subheader(section)
        for q in questions:
            if st.button(q):
                st.session_state["selected_question"] = q


# ----------------------------
# CHAT HEADER
# ----------------------------
st.markdown("""
<div class="header-gradient">
    <h1>IELTS Writing RAG Assistant</h1>
    <p>Ask a question or pick one from the sidebar.</p>
</div>
""", unsafe_allow_html=True)


# ----------------------------
# CHAT STATE
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "draft" not in st.session_state:
    st.session_state.draft = ""

if "selected_question" in st.session_state and st.session_state.selected_question:
    st.session_state.draft = st.session_state.selected_question
    st.session_state.selected_question = ""


# ----------------------------
# SHOW CHAT HISTORY
# ----------------------------
for msg in st.session_state.messages:
    bubble = "user-bubble" if msg["role"] == "user" else "bot-bubble"
    st.markdown(f"<div class='{bubble}'>{msg['content']}</div>", unsafe_allow_html=True)


# ----------------------------
# INPUT
# ----------------------------
query = st.text_input("", placeholder="Type here...", value=st.session_state.draft)
send = st.button("Send")


if send and query.strip():
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.draft = ""

    with st.spinner("Thinking..."):
        reply = rag_answer(query)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

