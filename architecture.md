# Architecture Diagram

```mermaid
flowchart TD
    A[User] --> B[Streamlit UI - app.py]
    B -->|Upload PDF/TXT/DOCX/CSV| C[Document Parser<br/>pypdf, python-docx, csv]
    C --> D[Chunker<br/>800 chars, 150 overlap]
    D --> E[Embedding Model<br/>sentence-transformers MiniLM]
    E --> F[(ChromaDB<br/>persistent vector store, cosine distance)]

    B -->|Ask question| G[Embed Question]
    G --> F
    F -->|top-k similar chunks + cosine similarity scores| H{Similarity above<br/>threshold?}
    H -->|No| I["Return: 'not found in documents'<br/>(no LLM call)"]
    H -->|Yes| J[Build grounded prompt<br/>context + conversation history]
    J --> K[Groq API<br/>openai/gpt-oss-120b]
    K --> L[Answer]
    L --> B
    I --> B
    F -->|source doc + page + score| B
```

## Component responsibilities

| Component | File | Responsibility |
|---|---|---|
| UI | `app.py` | Upload flow, chat interface, displays answer/sources/confidence |
| Document Parser | `backend/document_processor.py` | Extracts text from PDF (pypdf), DOCX (python-docx), CSV, and TXT; splits into overlapping chunks |
| Vector Store | `backend/vector_store.py` | Embeds chunks, stores/retrieves them via ChromaDB using cosine distance |
| QA Orchestrator | `backend/qa.py` | Retrieval gate, prompt construction, confidence scoring |
| LLM Client | `backend/llm_client.py` | Thin wrapper around the Groq API (`openai/gpt-oss-120b`) |

## Data flow summary
`User → UI → Parser (PDF/DOCX/CSV/TXT) → Chunker → Embeddings → ChromaDB (cosine) → Retrieval → (grounding check) → Groq LLM → Response`

## Notes on key fixes made during development

- **Distance metric:** ChromaDB defaults to raw Euclidean (`l2`) distance, which produced distance values routinely above 1 even for correct matches, causing the confidence gate to incorrectly refuse to answer. The collection is now explicitly created with `metadata={"hnsw:space": "cosine"}` in `vector_store.py`, so distances stay in a predictable 0–2 range and the `similarity = 1 - distance` conversion works as intended.
- **LLM model:** the Groq model was switched from `llama-3.3-70b-versatile` (not available on all accounts) to `openai/gpt-oss-120b`, confirmed available and the strongest chat model on the account used for testing.
