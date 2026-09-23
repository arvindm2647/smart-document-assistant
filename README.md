# Smart Document Assistant

A small RAG (Retrieval-Augmented Generation) application: upload PDF, TXT, DOCX, or CSV
documents, ask questions about them in plain English, and get answers grounded in the
uploaded content — with sources and a confidence indicator, and a clear "I don't know"
when the answer isn't in the documents.

## 1. Problem Understanding

Users often have several documents (policies, handbooks, reports, data files) and want a
direct answer to a specific question without reading through all of them manually. A plain
LLM chatbot can't answer questions about private documents it was never trained on, and if
pushed, it may confidently invent an answer. This project solves both problems with a RAG
pipeline: retrieve the most relevant passages from the user's own documents first, then ask
the LLM to answer strictly from that retrieved context — and explicitly refuse when nothing
relevant was found.

## 2. Architecture

See [`architecture.md`](./architecture.md) for the full diagram (Mermaid, renders natively
on GitHub). In short:

```
User → Streamlit UI → Document Parser (pypdf / python-docx / csv) → Chunker → Embeddings
(sentence-transformers) → ChromaDB (vector store) → Retrieval → Confidence
check → Prompt with context → Groq API (openai/gpt-oss-120b) → Answer + Sources → UI
```

- **Document Parser** (`backend/document_processor.py`) extracts text per page from PDFs
  (pypdf), reads TXT files directly, extracts paragraphs and tables from DOCX files
  (python-docx), and turns each CSV row into a readable "column: value" sentence.
- **Chunker** splits text into ~800-character chunks with 150-character overlap, so an
  answer-relevant sentence near a chunk boundary isn't lost.
- **Vector Store** (`backend/vector_store.py`) embeds each chunk with a local
  sentence-transformers model and stores it in a persistent local ChromaDB collection.
- **QA Orchestrator** (`backend/qa.py`) embeds the question, retrieves the top-k most
  similar chunks, checks a similarity threshold (hallucination gate), builds a grounded
  prompt (including recent conversation turns for follow-up questions), and calls the LLM.
- **LLM Client** (`backend/llm_client.py`) is a thin wrapper around the Groq API
  (OpenAI-compatible chat completions, running `openai/gpt-oss-120b` by default).

## 3. Technology Choices

| Choice | Reason |
|---|---|
| Streamlit | Fastest way to build an upload + chat UI within the time budget |
| pypdf / python-docx / csv | Lightweight, no external service, good enough text extraction for typical PDF, Word, and CSV files |
| sentence-transformers (all-MiniLM-L6-v2) | Free, runs locally — indexing documents doesn't cost API calls or require network access |
| ChromaDB (persistent, local) | Zero-setup embedded vector database, matches the assignment's suggested options |
| Groq API (`openai/gpt-oss-120b`) | Free API key, very fast inference (Groq's custom LPU hardware), OpenAI-compatible SDK — easy to explain and demo without a paid account |

## 4. Hallucination / Unknown-Question Handling

Two layers, on purpose — relying on prompting alone isn't reliable enough:

1. **Retrieval gate (before the LLM is even called):** every retrieved chunk has a
   similarity score. If the best match is below a threshold (`SIMILARITY_FLOOR = 0.25` in
   `backend/qa.py`), the app returns *"I couldn't find information about that in the
   uploaded documents"* directly — it never asks the LLM to answer from weak/irrelevant
   context in the first place.
2. **Prompt-level grounding:** the system prompt in `backend/qa.py` explicitly instructs
   the LLM to answer only from the given context, to say the exact "not found" sentence
   when the context is insufficient, and to cite which document the information came from.

A **confidence indicator** (🟢 High / 🟡 Medium / 🔴 Low, shown under every answer) is
derived from the top retrieval similarity score, so the user can see at a glance how well
the documents actually supported the answer — even when an answer is given.

## 5. Creative Feature(s)

- **Conversation memory:** the last 3 Q&A turns are included in the prompt, so users can
  ask natural follow-ups like "and what about sick leave?" without repeating context.
- **Confidence indicator:** every answer is labeled High/Medium/Low based on retrieval
  similarity, giving users an honest signal about answer reliability beyond just the text.

## 6. How to Run

```bash
# 1. Clone/unzip the project, then create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# then edit .env and paste your GROQ_API_KEY (get a free one at https://console.groq.com/keys)

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Sample documents are provided in `sample_docs/`
— upload `leave_policy.txt`, `insurance_policy.txt`, and `working_hours_policy.txt` to try
the example from the assignment ("How many annual leaves does an employee get?") and an
unanswerable question ("What is the company's stock price?"). PDF, DOCX, and CSV samples
can be uploaded alongside these to test the additional supported formats.

### Environment variables
| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Your free Groq API key ([console.groq.com/keys](https://console.groq.com/keys)) |
| `GROQ_MODEL` | No | Overrides the default model (`openai/gpt-oss-120b`). Available models vary by account — check yours by listing `client.models.list()` with the Groq SDK. |

No secrets are committed to this repository — `.env` is gitignored, and `.env.example` is
provided as a template only.

## 7. AI Tools Used

This solution was built with the help of Claude (Anthropic) for: scaffolding the RAG
pipeline structure, reviewing the chunking/overlap logic, adding DOCX/CSV support, and
drafting this README. The LLM-answering component itself calls the Groq API
(`openai/gpt-oss-120b`) at runtime. All generated code was read, tested against sample
documents (including PDF, DOCX, and CSV files), and adjusted by hand — e.g. the
retrieval-gate threshold and prompt wording were tuned manually after testing, and the
default Groq model was corrected after discovering the initially chosen model wasn't
available on the account being used.

## 8. Known Limitations

- **PDF extraction quality** depends on pypdf; scanned/image-only PDFs (no embedded text
  layer) won't extract any text — OCR is not implemented.
- **Fixed-size chunking** is simple and works well for prose-style documents, but doesn't
  respect semantic boundaries like tables or nested headings.
- **DOCX pagination** isn't preserved — Word files are treated as a single continuous page
  during chunking, so citations show the document name but not a page number.
- **Single local vector store**: documents persist in a local `chroma_db/` folder; this
  isn't a multi-user or production-scale setup.
- **Confidence score** is a heuristic based on embedding similarity, not a calibrated
  probability — it's a useful signal, not a guarantee.
- **No authentication/access control** — out of scope for this assignment.
- **Conversation memory** is a simple sliding window of the last 3 turns, not a full
  summarization-based memory system.
- **Model availability varies by Groq account** — the default model is set to one
  confirmed to work, but Groq's available model list can change or differ per account.

## 9. Time Log (approx. 8 hours)

| Phase | Time |
|---|---|
| Reading assignment & planning architecture | 30 min |
| Document processing + chunking (PDF, TXT, DOCX, CSV) | 1 hr |
| Vector store integration (ChromaDB + embeddings) | 1.5 hr |
| QA orchestration + hallucination handling | 1.5 hr |
| Streamlit UI + conversation memory + confidence feature | 1.5 hr |
| Testing & debugging with sample documents (incl. switching LLM provider) | 1 hr |
| README, architecture diagram, video | 1 hr |

## Project Structure

```
smart-doc-assistant/
├── app.py                       # Streamlit UI
├── backend/
│   ├── document_processor.py    # PDF/TXT/DOCX/CSV parsing + chunking
│   ├── vector_store.py          # Embeddings + ChromaDB
│   ├── qa.py                    # Retrieval, grounding, hallucination gate
│   └── llm_client.py            # Groq API wrapper
├── sample_docs/                 # Sample policy documents for testing
├── architecture.md              # Architecture diagram (Mermaid)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
