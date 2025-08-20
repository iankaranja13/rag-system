import os
import json
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredURLLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from bs4 import BeautifulSoup

from config.settings import Config

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and process PDF documents"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": os.path.basename(file_path),
                    "file_path": file_path
                })
            
            return self.chunk_documents(documents)
        except Exception as e:
            print(f"Error loading PDF {file_path}: {str(e)}")
            return []
    
    def load_text_file(self, file_path: str) -> List[Document]:
        """Load and process text files"""
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source_type": "text",
                    "file_name": os.path.basename(file_path),
                    "file_path": file_path
                })
            
            return self.chunk_documents(documents)
        except Exception as e:
            print(f"Error loading text file {file_path}: {str(e)}")
            return []
    
    def load_youtube_transcript(self, video_url: str) -> List[Document]:
        """Load YouTube transcript"""
        try:
            # Extract video ID from URL
            video_id = self.extract_video_id(video_url)
            if not video_id:
                return []
            
            # Get transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Combine transcript into text
            full_text = " ".join([entry['text'] for entry in transcript])
            
            # Create document
            doc = Document(
                page_content=full_text,
                metadata={
                    "source_type": "youtube",
                    "video_id": video_id,
                    "video_url": video_url,
                    "title": f"YouTube Video {video_id}"
                }
            )
            
            return self.chunk_documents([doc])
        except Exception as e:
            print(f"Error loading YouTube transcript {video_url}: {str(e)}")
            return []
    
    def scrape_website(self, url: str) -> List[Document]:
        """Scrape website content"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Create document
            doc = Document(
                page_content=text,
                metadata={
                    "source_type": "website",
                    "url": url,
                    "title": soup.title.string if soup.title else "Unknown"
                }
            )
            
            return self.chunk_documents([doc])
        except Exception as e:
            print(f"Error scraping website {url}: {str(e)}")
            return []
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunked_docs = []
        
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk.page_content)
                })
            
            chunked_docs.extend(chunks)
        
        return chunked_docs
    
    def extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from URL"""
        patterns = [
            r'youtube\.com/watch\?v=([^&]+)',
            r'youtu\.be/([^?]+)',
            r'youtube\.com/embed/([^?]+)'
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def process_documents_from_directory(self, directory_path: str, collection_name: str) -> List[Document]:
        """Process all documents in a directory"""
        all_documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            print(f"Directory {directory_path} does not exist")
            return []
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                file_extension = file_path.suffix.lower()
                
                if file_extension == '.pdf':
                    docs = self.load_pdf(str(file_path))
                elif file_extension in ['.txt', '.md']:
                    docs = self.load_text_file(str(file_path))
                else:
                    continue
                
                # Add collection metadata
                for doc in docs:
                    doc.metadata["collection"] = collection_name
                
                all_documents.extend(docs)
        
        print(f"Processed {len(all_documents)} chunks from {collection_name}")
        return all_documents
    
    def save_processed_documents(self, documents: List[Document], collection_name: str):
        """Save processed documents to disk"""
        processed_path = os.path.join(Config.PROCESSED_DIR, f"{collection_name}_processed.json")
        
        # Convert documents to serializable format
        docs_data = []
        for doc in documents:
            docs_data.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        
        with open(processed_path, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(documents)} processed documents to {processed_path}")
    
    def load_processed_documents(self, collection_name: str) -> List[Document]:
        """Load processed documents from disk"""
        processed_path = os.path.join(Config.PROCESSED_DIR, f"{collection_name}_processed.json")
        
        if not os.path.exists(processed_path):
            return []
        
        with open(processed_path, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
        
        documents = []
        for doc_data in docs_data:
            documents.append(Document(
                page_content=doc_data["page_content"],
                metadata=doc_data["metadata"]
            ))
        
        return documents