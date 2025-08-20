# src/reranker.py

from typing import List, Tuple, Dict, Any
from langchain.schema import Document
import logging
import numpy as np
from rank_bm25 import BM25Okapi

from config.settings import Config

# Try loading CrossEncoder safely
try:
    from sentence_transformers import CrossEncoder
    _torch_ok = True
except Exception as e:
    logging.warning(f"CrossEncoder not available: {e}")
    _torch_ok = False

# Try loading Cohere safely
try:
    import cohere
    _cohere_ok = True
except Exception as e:
    logging.warning(f"Cohere not available: {e}")
    _cohere_ok = False


class HybridReranker:
    def __init__(self):
        # CrossEncoder (PyTorch)
        self.cross_encoder = None
        if _torch_ok:
            try:
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logging.info("✅ CrossEncoder initialized")
            except Exception as e:
                logging.error(f"Failed to load CrossEncoder: {e}")
                self.cross_encoder = None

        # Cohere
        self.cohere_client = None
        if _cohere_ok and Config.COHERE_API_KEY:
            try:
                self.cohere_client = cohere.Client(Config.COHERE_API_KEY)
                logging.info("✅ Cohere client initialized")
            except Exception as e:
                logging.error(f"Failed to initialize Cohere client: {e}")

        # BM25
        self.bm25_index = None
        self.bm25_documents = None

    # -------------------------------
    # Rerank Methods
    # -------------------------------
    def rerank_with_cross_encoder(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        if not self.cross_encoder or not documents:
            return [(doc, 1.0) for doc in documents]  # fallback

        try:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = self.cross_encoder.predict(pairs)
            doc_score_pairs = list(zip(documents, scores))
            return sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logging.error(f"CrossEncoder reranking failed: {e}")
            return [(doc, 1.0) for doc in documents]

    def rerank_with_cohere(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        if not self.cohere_client or not documents:
            return [(doc, 1.0) for doc in documents]

        try:
            doc_texts = [doc.page_content for doc in documents]
            results = self.cohere_client.rerank(
                model="rerank-english-v2.0",
                query=query,
                documents=doc_texts,
                top_k=min(len(documents), Config.TOP_K_RERANK)
            )
            reranked_docs = [(documents[result.index], result.relevance_score) for result in results.results]
            return reranked_docs
        except Exception as e:
            logging.error(f"Cohere reranking failed: {e}")
            return [(doc, 1.0) for doc in documents]

    # -------------------------------
    # BM25 (sparse retrieval)
    # -------------------------------
    def create_bm25_index(self, documents: List[Document]):
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_docs)
        self.bm25_documents = documents

    def bm25_search(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        if not self.bm25_index:
            return []

        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL

        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.bm25_documents[idx], float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def hybrid_retrieval(self, query: str, dense_results: List[Document], alpha: float = 0.7) -> List[Tuple[Document, float]]:
        if not dense_results:
            return []

        self.create_bm25_index(dense_results)
        bm25_results = self.bm25_search(query, len(dense_results))
        bm25_scores = {doc.page_content: score for doc, score in bm25_results}

        dense_scores = {doc.page_content: 1.0 for doc in dense_results}  # equal weights
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0

        hybrid_results = []
        for doc in dense_results:
            dense_score = dense_scores.get(doc.page_content, 0.0)
            sparse_score = bm25_scores.get(doc.page_content, 0.0)
            normalized_sparse = sparse_score / max_bm25 if max_bm25 > 0 else 0.0
            combined_score = alpha * dense_score + (1 - alpha) * normalized_sparse
            hybrid_results.append((doc, combined_score))

        return sorted(hybrid_results, key=lambda x: x[1], reverse=True)

    # -------------------------------
    # Main API
    # -------------------------------
    def rerank_documents(self, query: str, documents: List[Document], method: str = "cross_encoder") -> List[Tuple[Document, float]]:
        if not documents:
            return []

        if method == "cross_encoder":
            return self.rerank_with_cross_encoder(query, documents)
        elif method == "cohere":
            return self.rerank_with_cohere(query, documents)
        elif method == "hybrid":
            return self.hybrid_retrieval(query, documents)
        else:
            return [(doc, 1.0) for doc in documents]

    def get_top_k_reranked(self, query: str, documents: List[Document], method: str = "cross_encoder", k: int = None) -> List[Document]:
        if k is None:
            k = Config.TOP_K_RERANK
        reranked = self.rerank_documents(query, documents, method)
        return [doc for doc, score in reranked[:k]]
