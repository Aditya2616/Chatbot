import logging
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app import config
from app.document_loader import load_documents_from_files, split_documents
from app.embeddings import get_embeddings
from app.rag_pipeline import RAGPipeline
from app.retriever import VectorStoreManager

logger = logging.getLogger(__name__)
router = APIRouter()

embeddings = get_embeddings()
vector_store_manager = VectorStoreManager(embeddings)
vector_store_manager.load()
rag_pipeline = RAGPipeline(vector_store_manager)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1)
    session_id: Optional[str] = None
    use_history: bool = True
    return_sources: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    cached: bool


class UploadResponse(BaseModel):
    message: str
    files: List[str]
    chunks_added: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    try:
        answer, sources, cached = rag_pipeline.answer_question(
            question=payload.question,
            top_k=payload.top_k,
            session_id=payload.session_id,
            use_history=payload.use_history,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return QueryResponse(
        answer=answer,
        sources=sources if payload.return_sources else [],
        cached=cached,
    )


@router.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    config.RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for file in files:
        filename = Path(file.filename or "").name
        if not filename:
            continue
        destination = config.RAW_DOCS_DIR / filename
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(destination)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid files uploaded")

    documents = load_documents_from_files(saved_paths)
    if not documents:
        raise HTTPException(
            status_code=400,
            detail="No supported documents were uploaded",
        )

    chunks = split_documents(documents)
    chunks_added = vector_store_manager.create_or_update(chunks)
    rag_pipeline.clear_cache()

    return UploadResponse(
        message="Documents ingested",
        files=[path.name for path in saved_paths],
        chunks_added=chunks_added,
        total_chunks=len(chunks),
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        vector_store_ready=vector_store_manager.has_index(),
    )
