# IELTS Writing RAG Assistant

An interactive **Streamlit web app** that helps IELTS Writing learners ask focused questions about coherence, cohesion, paragraph structure, vocabulary, grammar, and task response. The app uses a **retrieval-augmented generation (RAG)** pipeline: it retrieves relevant guidance from a local vector index (FAISS + SentenceTransformer) and then calls **Google Gemini** to generate structured, contextual answers. [file:114][file:115]

## Features

- Streamlit chat-style interface with user and assistant message bubbles.
- Sidebar with preset IELTS Writing questions grouped by:
  - Coherence, Cohesion, Paragraph Structure, Topic Sentences
  - Task Response (Task 2), Vocabulary, Grammar Range, Writing Quality
- Vector search over pre-embedded teaching content using:
  - `faiss` index (`index.faiss`)
  - `sentence-transformers/all-mpnet-base-v2` encoder
- RAG generation using **Gemini 2.0 Flash** (`google-generativeai`):
  - Retrieves top-k diverse passages from the index.
  - Builds a strict prompt with bullet-point format constraints.
  - Returns a well-structured explanation + summary.

## How it works

1. On startup, the app loads:
   - `index.faiss` – FAISS index of sentence embeddings.
   - `texts.pkl` – list of text chunks used as RAG context.
   - `metadata.pkl` – optional metadata associated with each chunk.
   These three files must be present in the working directory. [file:114]

2. User types a question (or clicks a preset question in the sidebar).
3. The question is encoded with `sentence-transformers/all-mpnet-base-v2` and searched in FAISS.
4. Retrieved passages are de-duplicated using cosine similarity to keep diverse context.
5. A prompt is built that:
   - injects the retrieved context,
   - enforces a strict answer format (intro, 4–6 bullets, short conclusion),
   - instructs Gemini to answer **only based on context**.
6. Gemini generates the response via `google.generativeai.GenerativeModel("gemini-2.0-flash")`.
7. Streamlit displays the conversation in styled chat bubbles.

## Requirements

Install dependencies (for example in a virtual environment):


`requirements.txt`:

streamlit==1.32.0
sentence-transformers==2.7.0
faiss-cpu==1.12.0
google-generativeai==0.8.5
protobuf==4.25.3
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.6.1
scipy==1.11.4
torch==2.2.2
transformers==4.41.1

## Environment setup

1. Create a `.streamlit/secrets.toml` file (not committed to Git) with your Gemini API key:
GEMINI_API_KEY = "your-gemini-api-key-here"

2. Place the following RAG assets in the project root (same directory as `app.py`):

- `index.faiss`
- `texts.pkl`
- `metadata.pkl`

These are generated offline by your own preprocessing pipeline (not included in this repo).

## Running the app

From the project directory:

streamlit run app.py

Then open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

- Type any IELTS Writing question in the input box, or click a preset question in the left sidebar.
- Wait for the “Thinking...” spinner while the app retrieves context and queries Gemini.
- Read the structured feedback and iterate with follow-up questions as needed.

## Project files

- `app.py` – main Streamlit application with:
  - UI layout and styling,
  - FAISS + SBERT retrieval logic,
  - Gemini RAG answer generation.
- `requirements.txt` – Python dependencies for running the app.
- `index.faiss`, `texts.pkl`, `metadata.pkl` – local vector index and texts (not tracked here; expected to be present when running the app).
