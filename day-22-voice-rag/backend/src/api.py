import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from src.rag_service import EnterpriseRAGService
from src.stt_service import stt_service  # <-- Day 22: New STT Import

app = FastAPI(title="Enterprise Voice RAG Control Plane", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = EnterpriseRAGService()

class ChatRequest(BaseModel):
    session_id: str
    message: str
    strategy: Optional[str] = "hybrid"

class ResetRequest(BaseModel):
    session_id: str

@app.get("/api/rag/sources")
async def get_sources():
    return {"sources": rag_service.strategy_mgr.get_indexed_sources()}

@app.post("/api/rag/ingest")
async def ingest_document(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename
    count = rag_service.strategy_mgr.ingest_file(filename, contents)
    return {"filename": filename, "indexed_chunks": count, "status": "success"}

@app.post("/api/rag/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    result = rag_service.chat(
        session_id=req.session_id,
        message=req.message,
        strategy=req.strategy or "hybrid"
    )
    return result

@app.post("/api/rag/session/reset")
async def reset_session(req: ResetRequest):
    rag_service.reset_session(req.session_id)
    return {"status": "reset", "session_id": req.session_id}

# ---------------------------------------------------------
# Day 22: Voice Integration Endpoint
# ---------------------------------------------------------
@app.post("/api/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    """Receives recorded audio binary from React MediaRecorder and returns transcription."""
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file received.")
            
        # Pass the raw audio bytes to Whisper
        transcript = stt_service.transcribe_audio(audio_bytes, filename=file.filename)
        
        return {
            "text": transcript,
            "filename": file.filename,
            "status": "success"
        }
    except Exception as e:
        print(f"[Transcription Error]: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")