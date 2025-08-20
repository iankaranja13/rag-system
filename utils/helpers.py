import base64
import json
import os
from typing import List, Dict, Any
from datetime import datetime
import streamlit as st

from config.settings import Config  # ✅ correct import

# -------------------------------------------------------------------
def format_sources(sources: List[Dict[str, Any]]) -> str:
    """Format source information for display"""
    if not sources:
        return "No sources available."
    
    formatted = []
    for i, source in enumerate(sources, 1):
        source_text = f"{i}. **{source.get('title', 'Unknown Title')}**"
        
        if source.get('type'):
            source_text += f" ({source['type'].title()})"
        
        if source.get('file_name'):
            source_text += f"\n   📄 File: {source['file_name']}"
        
        if source.get('url'):
            source_text += f"\n   🔗 URL: {source['url']}"
        
        if source.get('page'):
            source_text += f"\n   📖 Page: {source['page']}"
        
        formatted.append(source_text)
    
    return "\n\n".join(formatted)

# -------------------------------------------------------------------
def create_download_link(data: str, filename: str, link_text: str) -> str:
    """Create a download link for text data"""
    b64_data = base64.b64encode(data.encode()).decode()
    return f'<a href="data:text/plain;base64,{b64_data}" download="{filename}">{link_text}</a>'

# -------------------------------------------------------------------
def export_chat_history(messages: List[Dict[str, Any]]) -> str:
    """Export chat history to JSON format"""
    export_data = {
        "export_date": datetime.now().isoformat(),
        "total_messages": len(messages),
        "messages": []
    }
    
    for msg in messages:
        export_data["messages"].append({
            "query": msg["query"],
            "response": msg["response"],
            "collection": msg["collection"],
            "timestamp": msg["timestamp"].isoformat(),
            "sources_count": len(msg["sources"])
        })
    
    return json.dumps(export_data, indent=2)

# -------------------------------------------------------------------
def validate_file_type(file_name: str, allowed_types: List[str]) -> bool:
    """Validate if file type is allowed"""
    ext = os.path.splitext(file_name)[1].lower()
    return ext in [f".{ext}" for ext in allowed_types]

# -------------------------------------------------------------------
def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

# -------------------------------------------------------------------
def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display"""
    diff = datetime.now() - timestamp
    
    if diff.days > 0:
        return f"{diff.days} days ago"
    elif diff.seconds > 3600:
        return f"{diff.seconds // 3600} hours ago"
    elif diff.seconds > 60:
        return f"{diff.seconds // 60} minutes ago"
    else:
        return "Just now"

# -------------------------------------------------------------------
def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    for char in ['<', '>', ':', '"', '|', '?', '*', '/', '\\']:
        filename = filename.replace(char, "_")
    return filename

# -------------------------------------------------------------------
def calculate_text_statistics(text: str) -> Dict[str, int]:
    """Calculate basic text statistics"""
    words = text.split()
    sentences = [s for s in text.split('.') if s.strip()]
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    
    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "paragraphs": len(paragraphs)
    }

# -------------------------------------------------------------------
def check_system_health() -> Dict[str, Any]:
    """Check system health and configuration"""
    return {
        "api_keys": {
            "google": bool(Config.GEMINI_API_KEY),
            "cohere": bool(Config.COHERE_API_KEY)
        },
        "directories": {
            "data": os.path.exists(Config.DATA_DIR),
            "documents": os.path.exists(Config.DOCUMENTS_DIR),
            "vector_stores": os.path.exists(Config.VECTOR_STORE_DIR),
            "processed": os.path.exists(Config.PROCESSED_DIR)
        }
    }

# -------------------------------------------------------------------
def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else 0.0

# -------------------------------------------------------------------
def estimate_processing_time(num_documents: int, avg_doc_size: int) -> str:
    """Estimate processing time based on document count and size"""
    base_time = num_documents * 2
    size_factor = (avg_doc_size / 1000) * 0.5
    total_seconds = base_time + (num_documents * size_factor)
    
    if total_seconds < 60:
        return f"~{int(total_seconds)} seconds"
    elif total_seconds < 3600:
        return f"~{int(total_seconds // 60)} minutes"
    else:
        return f"~{int(total_seconds // 3600)} hours"

# -------------------------------------------------------------------
class MedicalTermExtractor:
    """Extract and highlight medical terms from text"""
    
    def __init__(self):
        self.prefixes = ['cardio', 'neuro', 'gastro', 'pulmo', 'nephro', 'hepato',
                         'osteo', 'arthro', 'dermato', 'ophthalmo', 'oto', 'rhino']
        self.suffixes = ['itis', 'osis', 'emia', 'pathy', 'therapy', 'scopy',
                         'tomy', 'ectomy', 'plasty', 'graphy', 'metry']
    
    def extract_medical_terms(self, text: str) -> List[str]:
        terms = []
        for word in text.split():
            w = word.lower().strip('.,!?;:"()[]{}')
            if any(w.startswith(p) for p in self.prefixes) or any(w.endswith(s) for s in self.suffixes):
                terms.append(w)
            elif len(w) > 6 and any(p in w for p in ['syndrome', 'disease', 'disorder', 'condition']):
                terms.append(w)
        return list(set(terms))

# -------------------------------------------------------------------
def create_medical_glossary(documents: List[str]) -> Dict[str, int]:
    """Create a glossary of medical terms from documents"""
    extractor = MedicalTermExtractor()
    term_frequency = {}
    
    for doc in documents:
        for term in extractor.extract_medical_terms(doc):
            term_frequency[term] = term_frequency.get(term, 0) + 1
    
    return dict(sorted(term_frequency.items(), key=lambda x: x[1], reverse=True))
