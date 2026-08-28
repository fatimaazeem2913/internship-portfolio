import os
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from google import genai
from src.retriever import LangChainRetrieverManager


class GeminiLangChainLLM(LLM):
    """LangChain LLM wrapper using official Google GenAI SDK with gemini-3.6-flash."""
    api_key: str
    model_name: str = "gemini-3.6-flash"

    @property
    def _llm_type(self) -> str:
        return "gemini_genai"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        client = genai.Client(api_key=self.api_key)
        candidate_models = [self.model_name, "gemini-3.6-flash", "gemini-3.5-flash"]
        
        for model in candidate_models:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                continue
                
        return ""


class ConversationalRAGPipeline:
    """Native LangChain Multi-Turn Conversational RAG with Session Memory and Citations."""
    def __init__(self, use_compression: bool = False):
        self.retriever_manager = LangChainRetrieverManager()
        self.use_compression = use_compression
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        
        if self.api_key:
            self.llm = GeminiLangChainLLM(api_key=self.api_key, model_name="gemini-3.6-flash")
        else:
            self.llm = None
            
        self.session_histories: Dict[str, List[BaseMessage]] = {}

    def get_session_history(self, session_id: str) -> List[BaseMessage]:
        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
        return self.session_histories[session_id]

    def clear_session(self, session_id: str):
        if session_id in self.session_histories:
            self.session_histories[session_id] = []

    def contextualize_query(self, session_id: str, query: str) -> str:
        """LangChain chain to reformulate follow-ups into standalone queries."""
        history = self.get_session_history(session_id)
        if not history or self.llm is None:
            return query

        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone search query which can be understood without the chat history. Do NOT answer the question, return ONLY the reformulated search query string."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        try:
            chain = contextualize_prompt | self.llm
            res = chain.invoke({
                "chat_history": history[-6:],
                "input": query
            })
            rewritten = res.strip()
            if rewritten and len(rewritten) > 3 and "provided documentation" not in rewritten:
                return rewritten
            return query
        except Exception:
            return query

    def ask(self, session_id: str, query: str) -> Dict[str, Any]:
        """Executes the full LangChain Conversational RAG chain."""
        standalone_query = self.contextualize_query(session_id, query)

        # Retrieve documents via LangChain Retriever Manager
        docs = self.retriever_manager.retrieve(
            query=standalone_query,
            top_k=3,
            use_compression=self.use_compression
        )

        context_parts = []
        citations = []
        for d in docs:
            src = d.metadata.get("source", d.metadata.get("source_doc", "SupportcoursesM-DLearning.pdf"))
            pg = d.metadata.get("page", d.metadata.get("page_number", 1))
            chunk_txt = d.page_content.strip()
            context_parts.append(f"[Source: {src}, Page: {pg}]\n{chunk_txt}")
            citations.append(f"[Source: {src}, Page: {pg}]")

        formatted_context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert technical assistant answering questions using retrieved documentation.
Rules:
1. Answer the user question accurately, clearly, and thoroughly based ONLY on the retrieved context below.
2. For every key fact, formula, or concept explanation, include an inline source citation in the format: [Source: <filename>, Page: <page>].
3. If the context does not contain enough information to answer the question, state: "The provided documentation does not contain information to answer this question."
4. Do not speculate or extrapolate beyond the provided context passages.

Retrieved Context:
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        history = self.get_session_history(session_id)
        answer_text = ""

        if self.llm is not None:
            try:
                rag_chain = qa_prompt | self.llm
                answer_text = rag_chain.invoke({
                    "context": formatted_context,
                    "chat_history": history[-4:],
                    "input": query
                }).strip()
            except Exception:
                answer_text = ""

        if not answer_text or len(answer_text) < 5:
            answer_text = "The provided documentation does not contain information to answer this question."

        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=answer_text))

        return {
            "session_id": session_id,
            "raw_query": query,
            "standalone_query": standalone_query,
            "answer": answer_text,
            "citations": sorted(list(set(citations))),
            "retrieved_chunks_count": len(docs),
            "compression_used": self.use_compression
        }