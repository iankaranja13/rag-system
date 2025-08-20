# config/settings.py

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    
    
    # Document Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_CHUNKS_PER_DOC = 100
    
    # Embedding Configuration
    EMBEDDING_MODEL = "models/embedding-001"
    EMBEDDING_MODEL_NAME = "GoogleGenerativeAIEmbeddings"
    # Use the embedding dimension for the model
    EMBEDDING_DIMENSION = 1536
    
    # Retrieval Configuration
    TOP_K_RETRIEVAL = 10
    TOP_K_RERANK = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Generation Configuration
    LLM_MODEL = "gemini-1.5-flash"
    MAX_TOKENS = 2048
    TEMPERATURE = 0.1
    
    # File Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
    VECTOR_STORES_DIR = os.path.join(DATA_DIR, "vector_stores")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    
    # Document Collections
    DOCUMENT_COLLECTIONS = {
        "cardiology": {
            "name": "Cardiology Research",
            "description": "Heart disease, cardiovascular research, and treatment guidelines",
            "vector_store_path": os.path.join(VECTOR_STORES_DIR, "cardiology")
        },
        "oncology": {
            "name": "Oncology Research", 
            "description": "Cancer research, treatment protocols, and clinical trials",
            "vector_store_path": os.path.join(VECTOR_STORES_DIR, "oncology")
        },
        "general_medicine": {
            "name": "General Medicine",
            "description": "General medical guidelines and clinical practices",
            "vector_store_path": os.path.join(VECTOR_STORES_DIR, "general_medicine")
        }
    }
    
    @classmethod
    def validate_api_keys(cls):
        """Validate that required API keys are present"""
        missing_keys = []
        
        if not cls.GEMINI_API_KEY:
            missing_keys.append("GEMINI_API_KEY")
            
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        return True
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.DOCUMENTS_DIR,
            cls.VECTOR_STORES_DIR,
            cls.PROCESSED_DIR
        ]
        
        for collection in cls.DOCUMENT_COLLECTIONS.values():
            directories.append(collection["vector_store_path"])
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)