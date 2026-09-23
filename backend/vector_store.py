from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from .document_processor import Chunk

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
)

    def add_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        ids = [f"{c.doc_name}::{c.chunk_id}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {"doc_name": c.doc_name, "page_number": c.page_number, "chunk_id": c.chunk_id}
            for c in chunks
        ]

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, question: str, n_results: int = 4) -> List[Dict]:
        results = self.collection.query(query_texts=[question], n_results=n_results)
        if not results["documents"] or not results["documents"][0]:
            return []

        out = []
        for text, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            
            similarity = max(0.0, 1.0 - distance)
            out.append({
                "text": text,
                "doc_name": meta["doc_name"],
                "page_number": meta["page_number"],
                "similarity": similarity,
            })
        return out

    def list_documents(self) -> List[str]:
        data = self.collection.get()
        return sorted({m["doc_name"] for m in data["metadatas"]}) if data["metadatas"] else []

    def reset(self) -> None:
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=self.embedding_fn
        )
