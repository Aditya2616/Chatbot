import logging
from pathlib import Path
from typing import Iterable, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

from app import config

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def load_documents_from_files(paths: Iterable[Path]) -> List[Document]:
    documents: List[Document] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            logging.warning("Unsupported file type: %s", path)
            continue
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            loader = Docx2txtLoader(str(path))
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata.setdefault("source", path.name)
        documents.extend(loaded_docs)
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)
