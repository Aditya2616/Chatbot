from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app import config


def get_embeddings() -> Embeddings:
    """Return a configured embeddings provider (Gemini or Hugging Face)."""
    provider = config.EMBEDDINGS_PROVIDER.lower()
    if provider not in {"huggingface", "gemini"}:
        raise ValueError(
            "Unsupported embeddings provider; use 'gemini' or 'huggingface'"
        )
    if provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=config.HUGGINGFACE_MODEL)
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required for Gemini embeddings")
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GEMINI_API_KEY,
    )
