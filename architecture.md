# Architecture Diagram

```mermaid
flowchart TD
    A[User] --> B[Streamlit UI - app.py]
    B -->|Upload PDF/TXT| C[Document Parser<br/>pypdf]
    C --> D[Chunker<br/>800 chars, 150 overlap]
    D --> E[Embedding Model<br/>sentence-transformers MiniLM]
    E --> F[(ChromaDB<br/>persistent vector store)]

    B -->|Ask question| G[Embed Question]
    G --> F
    F -->|top-k similar chunks + similarity scores| H{Similarity above<br/>threshold?}
    H -->|No| I["Return: 'not found in documents'<br/>(no LLM call)"]
    H -->|Yes| J[Build grounded prompt<br/>context + conversation history]
    J --> K[Groq API<br/>Llama 3.3 70B]
    K --> L[Answer]
    L --> B
    I --> B
    F -->|source doc + page + score| B
```

## Component responsibilities

| Component | File | Responsibility |
|---|---|---|
| UI | `app.py` | Upload flow, chat interface, displays answer/sources/confidence |
| Document Parser | `backend/document_processor.py` | Extracts text from PDF/TXT, splits into overlapping chunks |
| Vector Store | `backend/vector_store.py` | Embeds chunks, stores/retrieves them via ChromaDB |
| QA Orchestrator | `backend/qa.py` | Retrieval gate, prompt construction, confidence scoring |
| LLM Client | `backend/llm_client.py` | Thin wrapper around the Groq API |

## Data flow summary
`User → UI → Parser → Chunker → Embeddings → Vector Store → Retrieval → (grounding check) → LLM → Response`
