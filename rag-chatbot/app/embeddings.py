from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from app import config


def get_embeddings():
    provider = config.EMBEDDINGS_PROVIDER.lower()
    if provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=config.HUGGINGFACE_MODEL)
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )
