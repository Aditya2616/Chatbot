import logging

from fastapi import FastAPI

from app import config
from app.routes import router

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

app = FastAPI(title="RAG Chatbot API", version="1.0.0")
app.include_router(router)


@app.get("/")
def root() -> dict:
    return {"message": "RAG Chatbot API is running"}
