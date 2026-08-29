import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.ingestion import DocumentIngestionEngine
from src.rag_service import EnterpriseRAGService

app = FastAPI(
    title="Enterprise RAG Service",
    description="Multimodal Ingestion, Hybrid Search, and Grounded Multi-Turn Conversational Q&A API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = EnterpriseRAGService()
ingestion_engine = DocumentIngestionEngine()
UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    strategy: Optional[str] = "hybrid"


class ResetRequest(BaseModel):
    session_id: str


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "enterprise-rag-api"}


@app.post("/api/rag/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    parsed_docs = ingestion_engine.process_file(file_path, file.filename)
    if not parsed_docs:
        raise HTTPException(status_code=400, detail="No readable text could be extracted from the file.")

    rag_service.strategy_mgr.index_documents(parsed_docs)

    return {
        "status": "success",
        "filename": file.filename,
        "indexed_chunks": len(parsed_docs),
        "message": f"Successfully indexed {file.filename} into RAG strategy indices."
    }


@app.post("/api/rag/chat")
def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")
    try:
        response = rag_service.chat(
            session_id=request.session_id,
            message=request.message,
            strategy=request.strategy or "hybrid"
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/sources")
def get_sources():
    sources = {}
    for doc in rag_service.strategy_mgr.corpus_docs:
        src = doc.metadata.get("source", "unknown")
        pg = doc.metadata.get("page", 1)
        if src not in sources:
            sources[src] = {"filename": src, "total_chunks": 0, "pages": set()}
        sources[src]["total_chunks"] += 1
        sources[src]["pages"].add(pg)

    return {
        "sources": [
            {
                "filename": data["filename"],
                "total_chunks": data["total_chunks"],
                "pages_indexed": sorted(list(data["pages"]))
            }
            for data in sources.values()
        ]
    }


@app.post("/api/rag/session/reset")
def reset_session(request: ResetRequest):
    rag_service.reset_session(request.session_id)
    return {"status": "success", "message": f"Session {request.session_id} history cleared."}