import time
from typing import Dict, Any, Optional
from src.retriever import VectorRetriever
from src.prompt_builder import PromptConstructor
from src.llm_client import LLMClient

class SimpleRAGPipeline:
    """Complete End-to-End Grounded Retrieval-Augmented Generation Pipeline."""
    def __init__(
        self,
        retriever: Optional[VectorRetriever] = None,
        llm_client: Optional[LLMClient] = None
    ):
        self.retriever = retriever or VectorRetriever()
        self.llm_client = llm_client or LLMClient()

    def run(
        self,
        query: str,
        top_k: int = 3,
        confidence_threshold: float = 0.20
    ) -> Dict[str, Any]:
        """Executes End-to-End: Query -> Retrieval -> Prompt Augmentation -> Grounded Synthesis."""
        start_time = time.perf_counter()
        
        # 1. Retrieval
        retrieval_start = time.perf_counter()
        retrieved_chunks = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=confidence_threshold
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # 2. Augmented Prompt Construction
        prompt_data = PromptConstructor.build(query=query, retrieved_chunks=retrieved_chunks)

        # 3. LLM Generation
        generation_start = time.perf_counter()
        answer = self.llm_client.generate_response(
            system_instruction=prompt_data["system_instruction"],
            user_prompt=prompt_data["user_prompt"]
        )
        generation_ms = (time.perf_counter() - generation_start) * 1000.0
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        citations = []
        for c in retrieved_chunks:
            meta = c.get("metadata", {})
            src = meta.get("source", "Unknown")
            pg = meta.get("page_number", "N/A")
            tag = f"[Source: {src}, Page: {pg}]"
            if tag not in citations:
                citations.append(tag)

        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "citations": citations,
            "metrics": {
                "retrieval_latency_ms": round(retrieval_ms, 2),
                "generation_latency_ms": round(generation_ms, 2),
                "total_pipeline_latency_ms": round(total_latency_ms, 2),
                "chunks_retrieved": len(retrieved_chunks)
            }
        }