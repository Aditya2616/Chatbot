import logging
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

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
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required to initialize the LLM")
        self.vector_store_manager = vector_store_manager
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=config.TEMPERATURE,
            google_api_key=config.GEMINI_API_KEY,
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
        self.session_last_seen: Dict[str, float] = {}

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

        self._prune_sessions()
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
        self.session_last_seen[session_id] = time.time()

    def _get_history(self, session_id: Optional[str]) -> List[Tuple[str, str]]:
        if not session_id:
            return []
        if self._is_session_expired(session_id):
            self.history.pop(session_id, None)
            self.session_last_seen.pop(session_id, None)
            return []
        return self.history.get(session_id, [])

    def _is_session_expired(self, session_id: str) -> bool:
        last_seen = self.session_last_seen.get(session_id)
        if last_seen is None:
            return False
        return time.time() - last_seen > config.SESSION_TTL_SECONDS

    def _prune_sessions(self) -> None:
        if not self.session_last_seen:
            return
        now = time.time()
        expired = [
            session_id
            for session_id, last_seen in self.session_last_seen.items()
            if now - last_seen > config.SESSION_TTL_SECONDS
        ]
        for session_id in expired:
            self.history.pop(session_id, None)
            self.session_last_seen.pop(session_id, None)

    def _extract_sources(self, documents: List) -> List[Dict]:
        sources = []
        for doc in documents:
            metadata = doc.metadata or {}
            sources.append(
                {
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "snippet": self._truncate_text(doc.page_content),
                }
            )
        return sources

    def _truncate_text(self, text: str) -> str:
        max_length = config.SNIPPET_LENGTH
        if len(text) <= max_length:
            return text
        prefix = text[:max_length]
        split = prefix.rsplit(" ", 1)
        return split[0] if len(split) > 1 else prefix
