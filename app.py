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
    # MUST match the model used to build the FAISS index
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

index, texts, metadata = load_index_and_texts()
sbert_model = load_encoder()

# ----------------------------
# Gemini configuration
# ----------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.0-flash")


# ----------------------------
# Helper: semantic deduplication
# ----------------------------
def cosine_sim(a, b):
    num = float(np.dot(a, b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return num / denom


def retrieve_context(query, k_final: int = 7):
    """
    1) Search FAISS for more candidates (base_k)
    2) Sort by distance
    3) Remove near-duplicates using cosine similarity
    4) Return up to k_final diverse chunks
    """
    if not query.strip():
        return []

    # Encode query (must be float32 for FAISS)
    query_vec = sbert_model.encode([query], convert_to_numpy=True).astype("float32")

    # We search more than we need, then filter (diversity)
    base_k = max(k_final * 2, 15)
    distances, indices = index.search(query_vec, base_k)

    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(texts):
            continue
        t = texts[idx]
        if not t or not t.strip():
            continue
        candidates.append((dist, idx, t))

    if not candidates:
        return []

    # Sort by distance (smaller = closer)
    candidates.sort(key=lambda x: x[0])

    # Semantic deduplication
    selected_texts = []
    selected_embs = []

    for dist, idx, t in candidates:
        emb = sbert_model.encode([t], convert_to_numpy=True).astype("float32")[0]

        too_similar = False
        for prev_emb in selected_embs:
            if cosine_sim(emb, prev_emb) > 0.90:  # threshold
                too_similar = True
                break

        if too_similar:
            continue

        selected_texts.append(t)
        selected_embs.append(emb)

        if len(selected_texts) >= k_final:
            break

    return selected_texts


# ----------------------------
# Prompt builder
# ----------------------------
def build_prompt(query, retrieved):
    if not retrieved:
        # Fallback when FAISS returns nothing
        return f"""
You are an IELTS Writing Tutor.

The knowledge base did not return any relevant context for this query:

QUESTION:
{query}

You MUST answer:

"I can only provide a partial answer based on the available context."

Do NOT invent any IELTS rules or advice beyond that sentence.
"""

    retrieved_text = "\n".join([f"- {item}" for item in retrieved])

    return f"""
You are an IELTS Writing Tutor. Your answer MUST use ONLY the information provided in the CONTEXT.

❌ You MUST NOT:
- Add new ideas not present in the context
- Invent new IELTS rules
- Use outside knowledge
- Rephrase the same idea twice
- Use synonyms to disguise repeated ideas
- Introduce unrelated explanations

If the context does NOT contain enough information to answer the question fully, you MUST respond:
"I can only provide a partial answer based on the available context."

========================================
### CONTEXT (use ONLY this)
{retrieved_text}
========================================

### QUESTION
{query}

========================================
✅ STRICT FORMAT (FOLLOW EXACTLY)
========================================
1. Introduction Requirements ,do not rite Introduction just start with answer 

Provide a brief explanation of the issue using ONLY the retrieved context.

Do NOT reuse the same ideas, structure, or synonyms from the conclusion.

Do NOT repeat any sentence pattern or phrasing that appears in the conclusion.

2. Main Explanation Requirements (Bullet Points) , here also just write answer

You MUST follow ALL bullet-point rules below:

Provide exactly 4–6 bullet points.

Every bullet MUST begin with: •
(bullet symbol + one space)

Each bullet must contain 1–2 complete sentences.

Each bullet must express ONE unique idea only.

Insert ONE EMPTY LINE between bullets (no text, no spaces).

If the context contains similar ideas, you MUST merge them into one bullet.

You MUST NOT repeat ideas using different vocabulary.

Bullet example (spacing MUST look exactly like this):

• First idea sentence(s).

• Second idea sentence(s).

• Third idea sentence(s).

3. Conclusion Requirements, here too just write answer do not write Conclusion just start with answer

Provide a summary using completely different wording from the introduction.

MUST NOT repeat the same meaning, ideas, or synonyms from the introduction.

Explain how the listed strategies improve writing quality.

Tone Requirement

Writing must be academic, concise, and strictly grounded in the retrieved context.

No outside knowledge. No hallucinations.
"""



def rag_answer(query):
    retrieved = retrieve_context(query, k_final=7)
    prompt = build_prompt(query, retrieved)
    response = gemini_model.generate_content(prompt)
    return response.text, retrieved


# ----------------------------
# Chat UI CSS
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
    background: #ffffff; /* same as bot bubble */
    padding: 30px 10px;
    border-radius: 12px;
    color: #ffffff;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.12);
    margin-bottom: 25px;
}
/* Force all text inside header to black */
.header-gradient h1, 
.header-gradient h2,
.header-gradient h3,
.header-gradient p,
.header-gradient span {
    color: #000000 !important;
}
.input-bar {
    background: #ffffff;
    padding: 12px 15px;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Sidebar with Examples
# ----------------------------
with st.sidebar:
    st.header("📘 IELTS Writing Helper")
    st.write("Improve your writing using:")
    st.write("• Coherence & cohesion")
    st.write("• Logical progression")
    st.write("• Paragraph structure")
    st.write("• Transitions")
    st.write("• Grammar rules")
    
    st.write("Select a category to explore useful writing questions.")

    # === Categories and Questions ===
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

    # === Render categories ===
    for cat_label, questions in categories.items():
        st.subheader(cat_label)
        for q in questions:
            if st.button(q):
                st.session_state["preset_question"] = q


   

# ----------------------------
# Header Section
# ----------------------------
st.markdown("""
<div class='header-gradient'>
    <h1 style='margin-bottom:5px;'>IELTS Writing RAG Assistant</h1>
    <p style='margin-top:0px;font-size:18px;'>Chat with an AI tutor trained on your IELTS writing knowledge base</p>
</div>
""", unsafe_allow_html=True)


# ----------------------------
# Initialize chat history
# ----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ----------------------------
# Display chat history
# ----------------------------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Input Bar (Bottom Chat Box)
# ----------------------------
user_query = st.text_input(
    "",
    placeholder="Type your question here...",
    value=st.session_state.get("preset_question", ""),
    key="chat_input"
)

send = st.button("Send")


# ----------------------------
# Handle User Message
# ----------------------------
if send and user_query.strip() != "":
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": user_query})

    # Generate RAG response
    with st.spinner("Thinking..."):
        answer, retrieved_items = rag_answer(user_query)

    # Save assistant message
    st.session_state["messages"].append({"role": "assistant", "content": answer})

    # Clear preset so next input is empty unless user clicks example again
    st.session_state["preset_question"] = ""

    st.rerun()
