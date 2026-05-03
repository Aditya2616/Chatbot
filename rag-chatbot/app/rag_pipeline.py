import logging
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app import config
from app.retriever import VectorStoreManager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a domain-specific assistant. Use ONLY the provided context to answer. "
    "If the answer is not in the context, respond with exactly: \"I don't know\". "
    "Do not use outside knowledge."
)


class QueryCache:
    def __init__(self, maxsize: int, ttl_seconds: int) -> None:
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._store: "OrderedDict[Tuple, Tuple[Dict, float]]" = OrderedDict()

    def get(self, key: Tuple) -> Optional[Dict]:
        now = time.time()
        if key not in self._store:
            return None
        value, ts = self._store[key]
        if now - ts > self.ttl_seconds:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: Tuple, value: Dict) -> None:
        self._store[key] = (value, time.time())
        self._store.move_to_end(key)
        if len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()


class RAGPipeline:
    def __init__(self, vector_store_manager: VectorStoreManager) -> None:
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required to initialize the LLM")
        self.vector_store_manager = vector_store_manager
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=config.TEMPERATURE,
            openai_api_key=config.OPENAI_API_KEY,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Context:\n{context}\n\nConversation history:\n{history}\n\n"
                    "Question: {question}\nAnswer:",
                ),
            ]
        )
        self.cache = QueryCache(config.CACHE_MAXSIZE, config.CACHE_TTL_SECONDS)
        self.history: Dict[str, List[Tuple[str, str]]] = {}

    def clear_cache(self) -> None:
        self.cache.clear()

    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        session_id: Optional[str] = None,
        use_history: bool = True,
    ) -> Tuple[str, List[Dict], bool]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question cannot be empty")

        history_items = self._get_history(session_id) if use_history else []
        history_text = self._format_history(history_items)
        cache_key = (
            self.vector_store_manager.version,
            normalized_question,
            top_k or config.TOP_K,
            history_text,
        )
        cached = self.cache.get(cache_key)
        if cached:
            return cached["answer"], cached["sources"], True

        documents = self.vector_store_manager.similarity_search(
            normalized_question,
            top_k=top_k,
        )
        if not documents:
            answer = "I don't know"
            sources: List[Dict] = []
        else:
            context = self._format_context(documents)
            prompt_messages = self.prompt.format_messages(
                context=context,
                history=history_text or "None",
                question=normalized_question,
            )
            response = self.llm.invoke(prompt_messages)
            answer = response.content.strip()
            sources = self._extract_sources(documents)

        logger.info("Query: %s | Answer: %s", normalized_question, answer)
        if session_id and use_history:
            self._append_history(session_id, normalized_question, answer)

        payload = {"answer": answer, "sources": sources}
        self.cache.set(cache_key, payload)
        return answer, sources, False

    def _format_context(self, documents: List) -> str:
        return "\n\n".join(doc.page_content for doc in documents)

    def _format_history(self, history_items: List[Tuple[str, str]]) -> str:
        if not history_items:
            return ""
        history_lines = [f"User: {q}\nAssistant: {a}" for q, a in history_items]
        return "\n\n".join(history_lines)

    def _append_history(self, session_id: str, question: str, answer: str) -> None:
        session_history = self.history.get(session_id, [])
        session_history.append((question, answer))
        session_history = session_history[-config.MAX_HISTORY_TURNS :]
        self.history[session_id] = session_history

    def _get_history(self, session_id: Optional[str]) -> List[Tuple[str, str]]:
        if not session_id:
            return []
        return self.history.get(session_id, [])

    def _extract_sources(self, documents: List) -> List[Dict]:
        sources = []
        for doc in documents:
            metadata = doc.metadata or {}
            sources.append(
                {
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "snippet": doc.page_content[:200],
                }
            )
        return sources
