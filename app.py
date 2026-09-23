import os
import streamlit as st
from dotenv import load_dotenv

from backend.document_processor import process_document
from backend.vector_store import VectorStore
from backend.llm_client import LLMClient
from backend.qa import answer_question

load_dotenv()

st.set_page_config(page_title="Smart Document Assistant", layout="wide")


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore()


def get_llm() -> LLMClient | None:
    try:
        return LLMClient()
    except ValueError:
        return None


store = get_store()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {question, answer, sources, confidence}
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

st.title("Smart Document Assistant")
st.caption("Upload documents, then ask questions grounded in their contents.")

#Step 1: Upload
st.header("1. Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF, TXT, DOCX, or CSV files", type=["pdf", "txt", "docx", "csv"], accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        if f.name in st.session_state.processed_files:
            continue
        with st.spinner(f"Processing {f.name}..."):
            try:
                chunks = process_document(f.name, f.read())
                store.add_chunks(chunks)
                st.session_state.processed_files.add(f.name)
                st.success(f"Indexed {f.name} ({len(chunks)} chunks)")
            except Exception as e:
                st.error(f"Failed to process {f.name}: {e}")

#Step 2: Show uploaded docs
st.header("2. Uploaded Documents")
docs = store.list_documents()
if docs:
    for d in docs:
        st.write(f"- {d}")
else:
    st.info("No documents uploaded yet.")

if docs and st.button("Clear all documents"):
    store.reset()
    st.session_state.processed_files.clear()
    st.session_state.chat_history.clear()
    st.rerun()

st.divider()

#Step 3: Ask questions
st.header("3. Ask a Question")

llm = get_llm()
if llm is None:
    st.warning("Set GROQ_API_KEY in your environment or a .env file to enable question answering.")

question = st.text_input("Your question", placeholder="e.g. How many annual leaves does an employee get?")
ask_clicked = st.button("Ask", type="primary", disabled=(llm is None or not docs))

if ask_clicked and question.strip():
    with st.spinner("Searching documents and generating answer..."):
        result = answer_question(
            question=question,
            store=store,
            llm=llm,
            history=st.session_state.chat_history,
        )
    st.session_state.chat_history.append({
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"],
    })

#Step 4 & 5: Show answer + sources (most recent first)
st.header("4. Conversation")
for turn in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        conf_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[turn["confidence"]]
        st.write(turn["answer"])
        st.caption(f"{conf_color} Confidence: {turn['confidence']}")
        if turn["sources"]:
            with st.expander(f"Sources ({len(turn['sources'])})"):
                for s in turn["sources"]:                
                    page = f", page {s['page_number']}" if s["page_number"] != -1 else ""
                    st.markdown(f"**{s['doc_name']}{page}** — similarity {s['similarity']:.2f}")
                    st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))
