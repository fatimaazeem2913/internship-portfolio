import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.conversational_rag import ConversationalRAGPipeline

app = FastAPI(
    title="Day 20: LangChain Conversational RAG API",
    description="Multi-turn Conversational RAG with Contextual Compression and Source Attribution",
    version="1.0.0"
)

# Global pipeline instance
rag_pipeline = ConversationalRAGPipeline(use_compression=False)


class ChatRequest(BaseModel):
    session_id: str = Field(..., example="session_user_01", description="Unique session ID for conversation memory")
    message: str = Field(..., example="What are the components of a neuron in Figure 4.1?", description="User query")
    use_compression: Optional[bool] = Field(default=False, description="Enable LLM contextual compression retrieval")


class ChatResponse(BaseModel):
    session_id: str
    standalone_query: str
    answer: str
    citations: List[str]
    retrieved_chunks_count: int
    compression_used: bool


class SessionClearResponse(BaseModel):
    session_id: str
    status: str


@app.post("/api/rag/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    try:
        rag_pipeline.use_compression = payload.use_compression
        result = rag_pipeline.ask(session_id=payload.session_id, query=payload.message)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/session/clear", response_model=SessionClearResponse)
async def clear_session_endpoint(session_id: str):
    rag_pipeline.clear_session(session_id)
    return SessionClearResponse(session_id=session_id, status="Memory cleared successfully")


@app.get("/health")
async def health():
    return {"status": "healthy", "engine": "LangChain Conversational RAG"}