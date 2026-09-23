from typing import List, Dict, Optional
from .vector_store import VectorStore
from .llm_client import LLMClient

SIMILARITY_FLOOR = 0.25  

SYSTEM_PROMPT = """You are a Smart Document Assistant. You answer questions using ONLY the
document excerpts provided in the user message as context.

Rules you must follow strictly:
1. Base your answer only on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer, respond exactly with:
   "I couldn't find information about that in the uploaded documents."
   Do not guess or partially answer in that case.
3. When you do answer, mention which document(s) the information came from
   (use the document names given in the context, e.g. "According to leave_policy.pdf...").
4. Be concise and direct. Do not repeat the question back.
"""


def _confidence_label(top_similarity: float) -> str:
    if top_similarity >= 0.55:
        return "High"
    elif top_similarity >= SIMILARITY_FLOOR:
        return "Medium"
    else:
        return "Low"


def _build_context(results: List[Dict]) -> str:
    blocks = []
    for r in results:
        page_info = f", page {r['page_number']}" if r["page_number"] != -1 else ""
        blocks.append(f"[Source: {r['doc_name']}{page_info}]\n{r['text']}")
    return "\n\n---\n\n".join(blocks)


def _build_user_prompt(question: str, context: str, history: Optional[List[Dict]]) -> str:
    history_block = ""
    if history:
        turns = [f"Q: {h['question']}\nA: {h['answer']}" for h in history[-3:]]
        history_block = "Previous conversation (for follow-up context only):\n" + "\n\n".join(turns) + "\n\n"

    return (
        f"{history_block}"
        f"Document context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above."
    )


def answer_question(
    question: str,
    store: VectorStore,
    llm: LLMClient,
    history: Optional[List[Dict]] = None,
    n_results: int = 4,
) -> Dict:
    results = store.query(question, n_results=n_results)

    if not results or results[0]["similarity"] < SIMILARITY_FLOOR:
        return {
            "answer": "I couldn't find information about that in the uploaded documents.",
            "sources": [],
            "confidence": "Low",
        }

    context = _build_context(results)
    user_prompt = _build_user_prompt(question, context, history)
    answer = llm.generate(SYSTEM_PROMPT, user_prompt)
    seen = set()
    sources = []
    for r in results:
        key = (r["doc_name"], r["page_number"])
        if key not in seen:
            seen.add(key)
            sources.append(r)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": _confidence_label(results[0]["similarity"]),
    }
