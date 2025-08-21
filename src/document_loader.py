import os
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import tempfile

# LangChain community loaders
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    WebBaseLoader,
    UnstructuredFileLoader
)

# LangChain text splitting & schema
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # ✅ use schema Document instead of langchain_community.documents

# Additional imports
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
from config import Config


class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_youtube_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        if 'youtube.com/watch?v=' in url:
            return url.split('v=')[1].split('&')[0]
        elif 'youtu.be/' in url:
            return url.split('youtu.be/')[1].split('?')[0]
        return None
    
    def load_youtube_transcript(self, url: str) -> List[Document]:
        """Load and process YouTube transcript"""
        try:
            video_id = self.extract_youtube_video_id(url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")
            
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = " ".join([entry['text'] for entry in transcript])
            
            document = Document(
                page_content=full_text,
                metadata={
                    'source': url,
                    'type': 'youtube_transcript',
                    'video_id': video_id,
                    'duration': max([entry['start'] + entry['duration'] for entry in transcript])
                }
            )
            return self.text_splitter.split_documents([document])
        except Exception as e:
            st.error(f"Error loading YouTube transcript: {str(e)}")
            return []
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and process PDF file"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata.update({'type': 'pdf', 'source': os.path.basename(file_path)})
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return []
    
    def load_text_file(self, file_path: str) -> List[Document]:
        """Load and process text file"""
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata.update({'type': 'text', 'source': os.path.basename(file_path)})
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            st.error(f"Error loading text file: {str(e)}")
            return []
    
    def load_website(self, url: str) -> List[Document]:
        """Load and process website content"""
        try:
            loader = WebBaseLoader([url])
            documents = loader.load()
            for doc in documents:
                doc.metadata.update({'type': 'website', 'source': url, 'domain': urlparse(url).netloc})
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            st.error(f"Error loading website: {str(e)}")
            return []
    
    def process_uploaded_file(self, uploaded_file) -> List[Document]:
        """Process uploaded file from Streamlit"""
        if uploaded_file is None:
            return []
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'pdf':
                documents = self.load_pdf(tmp_path)
            elif file_extension in ['txt', 'md']:
                documents = self.load_text_file(tmp_path)
            else:
                loader = UnstructuredFileLoader(tmp_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata.update({'type': file_extension, 'source': uploaded_file.name})
                documents = self.text_splitter.split_documents(docs)
            return documents
        finally:
            os.unlink(tmp_path)
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        if not documents:
            return {}
        
        total_chunks = len(documents)
        total_chars = sum(len(doc.page_content) for doc in documents)
        
        type_counts = {}
        for doc in documents:
            doc_type = doc.metadata.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return {
            'total_chunks': total_chunks,
            'total_characters': total_chars,
            'average_chunk_size': total_chars // total_chunks if total_chunks > 0 else 0,
            'type_distribution': type_counts
        }
