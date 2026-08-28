import os
import json
from typing import List, Sequence, Optional, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
except ImportError:
    try:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain_core.documents.compressor import BaseDocumentCompressor
    except ImportError:
        ContextualCompressionRetriever = None
        BaseDocumentCompressor = object


class SentenceRelevanceCompressor(BaseDocumentCompressor):
    """LangChain document compressor that filters out non-relevant sentences."""
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Any] = None
    ) -> Sequence[Document]:
        query_terms = set(query.lower().split())
        compressed_docs = []
        for doc in documents:
            sentences = [s.strip() for s in doc.page_content.split(".") if s.strip()]
            relevant = []
            for s in sentences:
                s_lower = s.lower()
                if any(term in s_lower for term in query_terms if len(term) > 3) or len(relevant) < 2:
                    relevant.append(s)
            
            content = ". ".join(relevant) + "." if relevant else doc.page_content
            compressed_docs.append(Document(page_content=content, metadata=doc.metadata))
        return compressed_docs


class LangChainRetrieverManager:
    """Manages LangChain Chroma retriever and Contextual Compression Retriever."""
    def __init__(
        self,
        persist_directory: str = "./outputs/chroma_db",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = None
        self._init_vectorstore()

    def _init_vectorstore(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="rag_day20_langchain"
            )
            if self.vectorstore._collection.count() == 0:
                self.build_from_corpus()
        except Exception:
            self.build_from_corpus()

    def build_from_corpus(self):
        corpus_candidates = [
            "../day-19-hybrid-search-advanced-retrieval/data/sample_corpus.json",
            "../day-16-chunking-strategies/outputs/chunks_hierarchical.json",
            "../day-18-simple-rag-pipeline/data/chunks_hierarchical.json",
            "data/chunks_hierarchical.json"
        ]
        
        chunks = []
        for p in corpus_candidates:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        chunks = data
                        break

        docs = []
        for c in chunks:
            text = c.get("text", c.get("content", ""))
            meta = c.get("metadata", {})
            clean_meta = {}
            for k, v in meta.items():
                clean_meta[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
            docs.append(Document(page_content=text, metadata=clean_meta))

        if docs:
            self.vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name="rag_day20_langchain"
            )

    def retrieve(self, query: str, top_k: int = 3, use_compression: bool = False) -> List[Document]:
        if self.vectorstore is None or self.vectorstore._collection.count() == 0:
            self.build_from_corpus()
            
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        
        if use_compression:
            compressor = SentenceRelevanceCompressor()
            if ContextualCompressionRetriever is not None:
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever
                )
                return compression_retriever.invoke(query)
            else:
                docs = base_retriever.invoke(query)
                return list(compressor.compress_documents(docs, query))
            
        return base_retriever.invoke(query)