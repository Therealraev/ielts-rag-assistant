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
# HELPERS
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

    candidates.sort(key=lambda x: x[0])

    selected, embeddings = [], []
    for dist, idx, text in candidates:
        emb = sbert_model.encode([text], convert_to_numpy=True).astype("float32")[0]

        if any(cosine_sim(emb, prev) > 0.90 for prev in embeddings):
            continue

        selected.append(text)
        embeddings.append(emb)

        if len(selected) >= k_final:
            break

    return selected

def build_prompt(query, retrieved):
    if not retrieved:
        return f"I can only provide a partial answer based on the available context.\nQUESTION: {query}"

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
1. Short explanation of problem (NO title word)
2. 4–6 bullet points (exact spacing required)
3. Summary (different wording, no repeated idea)
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
html, body {font-family: 'Inter', sans-serif;}

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
# SIDEBAR WITH ALL QUESTIONS
# ----------------------------
with st.sidebar:
    st.header("📘 IELTS Writing Helper")

    categories = {
        "🧩 Coherence": [
            "How can I improve coherence?",
            "How do I ensure each sentence follows logically from the previous one?",
            "How do I avoid abrupt changes in ideas?",
            "How can I improve the logical flow of my writing?",
            "How do I maintain a consistent line of reasoning?"
        ],
        "🔗 Cohesion": [
            "How can I use cohesive devices effectively?",
            "What linking words improve cohesion?",
            "How do reference words improve cohesion?",
            "How do I show relationships between ideas clearly?",
            "How do I avoid repeating the same cohesive devices?"
        ],
        "📑 Paragraph Structure": [
            "How do I structure a paragraph?",
            "How can I organize a paragraph clearly?",
            "How can I sequence ideas inside a paragraph?",
            "How do I keep a paragraph focused on one idea?",
            "How do I avoid mixing unrelated ideas in a paragraph?"
        ],
        "🎯 Topic Sentences": [
            "What is a topic sentence?",
            "How do I write an effective topic sentence?",
            "How do topic sentences improve coherence?"
        ],
        "📝 Task Response (Task 2)": [
            "How do I choose good main ideas for Task 2?",
            "How can I avoid complex ideas in Task 2?",
            "How do I make sure my Task 2 ideas are easy to develop?",
            "How can I avoid technical arguments in Task 2?"
        ],
        "📚 Vocabulary": [
            "How can I improve vocabulary?",
            "How can I improve vocabulary for Task 2?",
            "Which formal verbs are useful for IELTS writing?",
            "How can I avoid repeating words?",
            "How do I use topic-specific vocabulary effectively?"
        ],
        "⚙️ Grammar Range": [
            "How can I improve my grammatical range?",
            "How do I avoid sentence fragments?",
            "How do I fix run-on sentences?",
            "How can I vary my sentence structure?",
            "What are common subject–verb agreement mistakes?"
        ],
        "✨ Writing Quality": [
            "How do I avoid list-like writing?",
            "How can I add explanation sentences?",
            "How do I keep my writing clear and coherent?",
            "How do I keep my writing formal?",
            "How can I organize my ideas more effectively?"
        ]
    }

    for cat_label, questions in categories.items():
        st.subheader(cat_label)
        for q in questions:
            if st.button(q, key=q):
                st.session_state["selected_question"] = q
                st.experimental_rerun()


# ----------------------------
# HEADER
# ----------------------------
st.markdown("""
<div class='header-gradient'>
    <h1>IELTS Writing RAG Assistant</h1>
    <p>Ask about structure, grammar, clarity, coherence, vocabulary & scoring.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# CHAT SESSION
# ----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []


for msg in st.session_state["messages"]:
    role = "user-bubble" if msg["role"] == "user" else "bot-bubble"
    st.markdown(f"<div class='{role}'>{msg['content']}</div>", unsafe_allow_html=True)

# ----------------------------
# INPUT + SEND BUTTON
# ----------------------------
default_value = st.session_state.get("selected_question", "")
user_query = st.text_input("", placeholder="Type your question here...", value=default_value)
send = st.button("Send")

if send and user_query.strip():
    st.session_state["messages"].append({"role": "user", "content": user_query})

    with st.spinner("Thinking..."):
        bot_reply = rag_answer(user_query)

    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})

    st.session_state["selected_question"] = ""
    st.experimental_rerun()
