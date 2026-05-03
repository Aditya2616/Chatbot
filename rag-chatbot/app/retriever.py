import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app import config

INDEX_FILES = ["index.faiss", "index.pkl"]


def _vector_store_exists(path: Path) -> bool:
    return all((path / filename).exists() for filename in INDEX_FILES)


class VectorStoreManager:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vector_store: Optional[FAISS] = None
        self.version = 0

    def load(self) -> Optional[FAISS]:
        if _vector_store_exists(config.VECTOR_STORE_DIR):
            self.vector_store = FAISS.load_local(
                str(config.VECTOR_STORE_DIR),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logging.info("Loaded vector store from %s", config.VECTOR_STORE_DIR)
        return self.vector_store

    def create_or_update(self, documents: List[Document]) -> int:
        if not documents:
            return 0
        if self.vector_store is None and _vector_store_exists(config.VECTOR_STORE_DIR):
            self.load()
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        self._persist()
        self.version += 1
        return len(documents)

    def _persist(self) -> None:
        config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(config.VECTOR_STORE_DIR))

    def similarity_search(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        if self.vector_store is None:
            self.load()
        if self.vector_store is None:
            return []
        k = top_k or config.TOP_K
        return self.vector_store.similarity_search(query, k=k)

    def has_index(self) -> bool:
        return _vector_store_exists(config.VECTOR_STORE_DIR)
