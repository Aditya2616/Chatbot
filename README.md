# LLM-Powered RAG Chatbot

A production-ready, domain-specific Q&A chatbot using Retrieval-Augmented Generation (RAG). It ingests PDF/TXT/DOCX files, builds embeddings, stores them in FAISS, and serves answers via a FastAPI REST API.

## Features
- Document ingestion for PDF, TXT, DOCX
- Chunking with `RecursiveCharacterTextSplitter`
- Embeddings via OpenAI or Hugging Face
- FAISS vector store with local persistence
- Strict, context-grounded RAG prompt ("I don't know" fallback)
- FastAPI endpoints: `/query`, `/upload`, `/health`
- Optional Streamlit UI
- Query/response logging + in-memory caching
- Dockerfile for deployment

## Project Structure
```
rag-chatbot/
├── app/
│   ├── main.py
│   ├── routes.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   ├── embeddings.py
│   ├── document_loader.py
│   └── config.py
├── data/
│   ├── raw_docs/
│   └── vector_store/
├── frontend/
│   └── streamlit_app.py
├── .env (create from .env.example)
├── .env.example
├── requirements.txt
└── Dockerfile
```

## Setup (Local)
1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```
4. **Run the API**
   ```bash
   uvicorn app.main:app --reload
   ```

## API Usage
### Upload documents
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@data/raw_docs/your-document.pdf"
```

### Ask a question
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the refund policy?"}'
```

### Health check
```bash
curl http://localhost:8000/health
```

## Streamlit UI (Optional)
```bash
streamlit run frontend/streamlit_app.py
```

## Example Test Cases
1. **Answer found in documents**
   - Query: "What is the warranty period for the product?"
   - Expected: A precise answer sourced from the uploaded documents.

2. **Answer NOT found in documents**
   - Query: "What is the capital of France?"
   - Expected: `I don't know`

3. **Another context-grounded question**
   - Query: "List the onboarding steps described in the handbook."
   - Expected: Steps listed from the document content.

## Hallucination Reduction
- Strict system prompt that prohibits outside knowledge
- RAG retrieval ensures answers are grounded in relevant context
- Fallback response when context is missing

## Evaluation Method (Suggested)
- Create a small set of Q&A pairs from your documents.
- Measure **answerability rate** (percentage of questions answered from context).
- Track **"I don't know" precision** (how often the model correctly abstains).
- Optionally compare baseline answers without retrieval.

## Caching & Logging
- In-memory LRU cache for repeated queries (configurable TTL).
- Session history expires automatically (configurable TTL).
- Logs each query and response via standard logging output.

## Docker
```bash
docker build -t rag-chatbot .
docker run -p 8000:8000 --env-file .env rag-chatbot
```
