import streamlit as st
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ----------------------------
# PAGE CONFIG (must be first)
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
# Gemini configuration
# ----------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.0-flash")


# ----------------------------
# Helper Functions
# ----------------------------
def cosine_sim(a, b):
    num = float(np.dot(a, b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return num / denom

def retrieve_context(query, k_final: int = 7):
    if not query.strip():
        return []

    query_vec = sbert_model.encode([query], convert_to_numpy=True).astype("float32")
    base_k = max(k_final * 2, 15)
    distances, indices = index.search(query_vec, base_k)

    candidates = [
        (dist, idx, texts[idx])
        for dist, idx in zip(distances[0], indices[0])
        if 0 <= idx < len(texts) and texts[idx].strip()
    ]

    candidates.sort(key=lambda x: x[0])
    selected_texts, selected_embs = [], []

    for _, idx, t in candidates:
        emb = sbert_model.encode([t], convert_to_numpy=True).astype("float32")[0]

        if any(cosine_sim(emb, prev) > 0.90 for prev in selected_embs):
            continue

        selected_texts.append(t)
        selected_embs.append(emb)

        if len(selected_texts) >= k_final:
            break

    return selected_texts

def build_prompt(query, retrieved):
    if not retrieved:
        return f"""
I can only provide a partial answer based on the available context.
"""

    retrieved_text = "\n".join([f"- {item}" for item in retrieved])

    return f"""
You are an IELTS Writing Tutor. Your answer MUST use ONLY the information provided in the CONTEXT.

========================================
### CONTEXT
{retrieved_text}
========================================

### QUESTION
{query}

========================================
FORMAT:
- Short introduction (no title "Introduction")
- 4–6 bullets (strict rules)
- Short conclusion sentence (no title "Conclusion")

No outside knowledge. No repeated sentences.
"""

def rag_answer(query):
    retrieved = retrieve_context(query)
    prompt = build_prompt(query, retrieved)
    response = gemini_model.generate_content(prompt)
    return response.text, retrieved


# ----------------------------
# CSS Style
# ----------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.header-gradient {
    background: white;
    padding: 30px 10px;
    border-radius: 12px;
    color: black !important;
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
# Sidebar Preset Questions
# ----------------------------
with st.sidebar:
    st.header("📘 IELTS Writing Helper")
    st.write("Click a question to auto-fill the chat box.")

    categories = {
        "🧩 Coherence": [
            "How can I improve coherence?",
            "How do I ensure each sentence follows logically from the previous one?",
            "How do I avoid abrupt changes in ideas?"
        ]
    }

    for label, items in categories.items():
        st.subheader(label)
        for q in items:
            if st.button(q):
                st.session_state["chat_input"] = q
                st.rerun()

# ----------------------------
# Chat UI
# ----------------------------
st.markdown("""
<div class='header-gradient'>
    <h1>IELTS Writing RAG Assistant</h1>
    <p>Chat with an AI tutor trained on your IELTS writing knowledge base</p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.write(f"**{msg['role'].capitalize()}:** {msg['content']}")

user_query = st.text_input("Ask something", key="chat_input")
if st.button("Send") and user_query.strip():
    st.session_state.messages.append({"role": "user", "content": user_query})
    answer, _ = rag_answer(user_query)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_input = ""
    st.rerun()
