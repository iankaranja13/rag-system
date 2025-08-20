# src/embeddings.py

import os
import pickle
from typing import List, Dict, Any

import faiss
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.async_utils import ensure_event_loop

from config.settings import Config

class EmbeddingManager:
    def __init__(self):
        ensure_event_loop()
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=Config.GEMINI_API_KEY
        )
        self.vector_stores = {}

    def create_vector_store(self, documents: List[Document], collection_name: str) -> FAISS:
        """Create FAISS vector store from documents"""
        if not documents:
            raise ValueError("No documents provided for vector store creation")

        print(f"Creating embeddings for {len(documents)} documents...")

        # Create FAISS vector store
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        # Save vector store
        vector_store_path = Config.DOCUMENT_COLLECTIONS[collection_name]["vector_store_path"]
        os.makedirs(vector_store_path, exist_ok=True)
        vector_store.save_local(vector_store_path)

        # Save document metadata separately
        metadata_path = os.path.join(vector_store_path, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump([doc.metadata for doc in documents], f)

        print(f"Vector store saved to {vector_store_path}")
        self.vector_stores[collection_name] = vector_store

        return vector_store

    def load_vector_store(self, collection_name: str) -> FAISS:
        """Load existing vector store"""
        if collection_name in self.vector_stores:
            return self.vector_stores[collection_name]

        vector_store_path = Config.DOCUMENT_COLLECTIONS[collection_name]["vector_store_path"]

        if not os.path.exists(vector_store_path):
            raise FileNotFoundError(f"Vector store not found at {vector_store_path}")

        try:
            vector_store = FAISS.load_local(
                vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.vector_stores[collection_name] = vector_store
            print(f"Loaded vector store for {collection_name}")
            return vector_store
        except Exception as e:
            raise Exception(f"Error loading vector store: {str(e)}")

    def get_vector_store_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a vector store"""
        vector_store_path = Config.DOCUMENT_COLLECTIONS[collection_name]["vector_store_path"]

        if not os.path.exists(vector_store_path):
            return {
                "exists": False,
                "document_count": 0,
                "last_updated": None
            }

        try:
            vector_store = self.load_vector_store(collection_name)
            doc_count = vector_store.index.ntotal
            index_file = os.path.join(vector_store_path, "index.faiss")
            last_updated = os.path.getmtime(index_file) if os.path.exists(index_file) else None

            return {
                "exists": True,
                "document_count": doc_count,
                "last_updated": last_updated,
                "path": vector_store_path
            }
        except Exception as e:
            return {
                "exists": False,
                "document_count": 0,
                "last_updated": None,
                "error": str(e)
            }

    def delete_vector_store(self, collection_name: str):
        """Delete a vector store"""
        vector_store_path = Config.DOCUMENT_COLLECTIONS[collection_name]["vector_store_path"]
        if os.path.exists(vector_store_path):
            import shutil
            shutil.rmtree(vector_store_path)
            print(f"Deleted vector store for {collection_name}")

        if collection_name in self.vector_stores:
            del self.vector_stores[collection_name]

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query"""
        return self.embeddings.embed_query(query)

    def similarity_search(self, collection_name: str, query: str, k: int = None) -> List[Document]:
        """Perform similarity search"""
        if k is None:
            k = Config.TOP_K_RETRIEVAL

        vector_store = self.load_vector_store(collection_name)
        return vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, collection_name: str, query: str, k: int = None) -> List[tuple]:
        """Perform similarity search with scores"""
        if k is None:
            k = Config.TOP_K_RETRIEVAL

        vector_store = self.load_vector_store(collection_name)
        return vector_store.similarity_search_with_score(query, k=k)
