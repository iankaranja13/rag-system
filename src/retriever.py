# src/retriever.py

from typing import List, Dict, Any, Tuple
from langchain.schema import Document
import logging
import faiss
from src.embeddings import EmbeddingManager
from src.reranker import HybridReranker
from config.settings import Config
from utils.async_utils import ensure_event_loop


class MedicalRetriever:
    def __init__(self):
        ensure_event_loop()
        self.embedding_manager = EmbeddingManager()
        self.reranker = HybridReranker()

        # Detect available rerankers
        self.has_cross_encoder = self.reranker.cross_encoder is not None
        self.has_cohere = self.reranker.cohere_client is not None

        if self.has_cross_encoder:
            logging.info("✅ Using CrossEncoder for reranking")
        elif self.has_cohere:
            logging.info("✅ Using Cohere for reranking")
        else:
            logging.warning("⚠️ No reranker available, using only FAISS similarity search")

    def retrieve_documents(self, query: str, collection_name: str,
                          retrieval_method: str = "dense",
                          rerank_method: str = "auto") -> List[Document]:
        """Main retrieval method"""

        try:
            # Step 1: Initial retrieval
            if retrieval_method == "dense":
                initial_docs = self._dense_retrieval(query, collection_name)
            elif retrieval_method == "hybrid":
                initial_docs = self._hybrid_retrieval(query, collection_name)
            else:
                initial_docs = self._dense_retrieval(query, collection_name)

            if not initial_docs:
                return []

            # Step 2: Decide reranking strategy
            if rerank_method == "none":
                return initial_docs[:Config.TOP_K_RERANK]

            if rerank_method == "auto":
                if self.has_cross_encoder:
                    rerank_method = "cross_encoder"
                elif self.has_cohere:
                    rerank_method = "cohere"
                else:
                    rerank_method = "none"

            # Step 3: Apply reranking
            if rerank_method != "none":
                reranked_docs = self.reranker.get_top_k_reranked(
                    query, initial_docs, rerank_method
                )
                return reranked_docs

            return initial_docs[:Config.TOP_K_RERANK]

        except Exception as e:
            logging.error(f"Error in document retrieval: {str(e)}")
            return []

    def _dense_retrieval(self, query: str, collection_name: str) -> List[Document]:
        """Perform dense vector retrieval"""
        try:
            docs = self.embedding_manager.similarity_search(
                collection_name, query, Config.TOP_K_RETRIEVAL
            )
            return docs
        except Exception as e:
            logging.error(f"Error in dense retrieval: {str(e)}")
            return []

    def _hybrid_retrieval(self, query: str, collection_name: str) -> List[Document]:
        """Perform hybrid retrieval combining dense and sparse methods"""
        try:
            dense_docs = self._dense_retrieval(query, collection_name)
            if not dense_docs:
                return []

            hybrid_results = self.reranker.hybrid_retrieval(query, dense_docs)
            return [doc for doc, score in hybrid_results[:Config.TOP_K_RETRIEVAL]]

        except Exception as e:
            logging.error(f"Error in hybrid retrieval: {str(e)}")
            return self._dense_retrieval(query, collection_name)

    def retrieve_with_scores(self, query: str, collection_name: str) -> List[Tuple[Document, float]]:
        """Retrieve documents with similarity scores"""
        try:
            docs_with_scores = self.embedding_manager.similarity_search_with_score(
                collection_name, query, Config.TOP_K_RETRIEVAL
            )
            return docs_with_scores
        except Exception as e:
            logging.error(f"Error retrieving documents with scores: {str(e)}")
            return []

    def get_relevant_context(self, query: str, collection_name: str,
                           max_context_length: int = 4000) -> str:
        """Get relevant context for generation, respecting token limits"""

        docs = self.retrieve_documents(query, collection_name)

        if not docs:
            return "No relevant documents found."

        context_parts = []
        current_length = 0

        for i, doc in enumerate(docs):
            source_info = self._format_source_info(doc.metadata)
            doc_text = f"Source {i+1}: {source_info}\n{doc.page_content}\n"

            if current_length + len(doc_text) > max_context_length:
                remaining_space = max_context_length - current_length - 100
                if remaining_space > 200:
                    truncated_content = doc.page_content[:remaining_space] + "..."
                    doc_text = f"Source {i+1}: {source_info}\n{truncated_content}\n"
                    context_parts.append(doc_text)
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n---\n".join(context_parts)

    def _format_source_info(self, metadata: Dict[str, Any]) -> str:
        source_type = metadata.get("source_type", "unknown")

        if source_type == "pdf":
            return f"PDF: {metadata.get('file_name', 'Unknown file')}"
        elif source_type == "text":
            return f"Text file: {metadata.get('file_name', 'Unknown file')}"
        elif source_type == "youtube":
            return f"YouTube: {metadata.get('title', 'Unknown video')}"
        elif source_type == "website":
            return f"Website: {metadata.get('title', 'Unknown page')}"
        else:
            return f"Document: {metadata.get('file_name', 'Unknown source')}"

    def search_specific_document(self, query: str, collection_name: str,
                               document_filter: Dict[str, Any]) -> List[Document]:
        """Search within specific documents based on metadata filter"""

        all_docs = self.retrieve_documents(query, collection_name)

        filtered_docs = []
        for doc in all_docs:
            match = all(doc.metadata.get(key) == value for key, value in document_filter.items())
            if match:
                filtered_docs.append(doc)

        return filtered_docs

    def get_document_statistics(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        try:
            vector_store_info = self.embedding_manager.get_vector_store_info(collection_name)
            return {
                "collection_name": collection_name,
                "total_chunks": vector_store_info.get("document_count", 0),
                "vector_store_exists": vector_store_info.get("exists", False),
                "last_updated": vector_store_info.get("last_updated"),
                "collection_path": vector_store_info.get("path")
            }
        except Exception as e:
            return {
                "collection_name": collection_name,
                "total_chunks": 0,
                "vector_store_exists": False,
                "error": str(e)
            }
