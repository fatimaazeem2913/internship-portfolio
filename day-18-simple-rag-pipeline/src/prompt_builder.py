from typing import List, Dict, Any

class PromptConstructor:
    """Constructs structured, citation-enforced augmented prompts for the LLM."""
    
    SYSTEM_INSTRUCTION = """You are an expert, honest AI Assistant.
Answer the user query strictly using ONLY the provided verified context passages.
Rules:
1. Cite the exact source document and page number in brackets [Source: <doc_name>, Page: <p_num>] for every factual assertion.
2. If the context describes a figure, plot, or diagram, clearly reference the visual details and formulas indicated.
3. If the context does not contain sufficient facts to answer the question, explicitly state:
   "I am sorry, but the provided documentation does not contain sufficient information to answer this question."
4. Do NOT hallucinate, extrapolate, or bring in external outside facts."""

    @classmethod
    def build(cls, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, str]:
        """Formats retrieved text and figure chunks into tagged context blocks."""
        if not retrieved_chunks:
            formatted_context = "NO RELEVANT CONTEXT FOUND."
        else:
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks, start=1):
                meta = chunk.get("metadata", {})
                source = meta.get("source", "Unknown Document")
                page = meta.get("page_number", "N/A")
                heading = meta.get("section_heading", "General")
                doc_type = meta.get("doc_type", "text")
                conf = chunk.get("confidence_score", 0.0)
                
                # Robust text/content extraction
                chunk_text = chunk.get("text") or chunk.get("content", "")
                
                type_tag = "[Figure / Visual Diagram]" if "figure" in doc_type or "fig_" in str(chunk.get("id", "")) else "[Document Passage]"
                
                header = f"--- [Passage {i}] {type_tag} Source: {source} (Page {page}) | Section: {heading} | Confidence: {conf:.2f} ---"
                context_parts.append(f"{header}\n{chunk_text}\n")
            formatted_context = "\n".join(context_parts)

        user_prompt = f"""CONTEXT INFORMATION:
=====================================================
{formatted_context}
=====================================================

USER QUESTION:
{query}

GROUNDED ANSWER (with bracketed citations):"""

        return {
            "system_instruction": cls.SYSTEM_INSTRUCTION,
            "user_prompt": user_prompt,
            "context_str": formatted_context
        }