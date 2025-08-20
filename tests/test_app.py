import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager
from config.settings import Config

class TestDocumentProcessor:
    
    def setup_method(self):
        self.processor = DocumentProcessor()
    
    def test_chunk_documents(self):
        """Test document chunking"""
        from langchain.schema import Document
        
        test_doc = Document(
            page_content="This is a test document. " * 100,
            metadata={"source": "test"}
        )
        
        chunks = self.processor.chunk_documents([test_doc])
        
        assert len(chunks) > 1
        assert all(len(chunk.page_content) <= Config.CHUNK_SIZE + Config.CHUNK_OVERLAP for chunk in chunks)
    
    def test_extract_video_id(self):
        """Test YouTube video ID extraction"""
        test_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ"
        ]
        
        for url in test_urls:
            video_id = self.processor.extract_video_id(url)
            assert video_id == "dQw4w9WgXcQ"

if __name__ == "__main__":
    pytest.main([__file__])