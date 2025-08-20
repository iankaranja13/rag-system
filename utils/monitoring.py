import logging
import streamlit as st
from datetime import datetime
import json

class AppMonitoring:
    
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        """Setup application logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_query(self, query: str, collection: str, response_time: float):
        """Log user queries for analytics"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],  # Truncate for privacy
            "collection": collection,
            "response_time": response_time,
            "user_id": st.session_state.get("user_id", "anonymous")
        }
        
        self.logger.info(f"QUERY: {json.dumps(log_data)}")
    
    def log_error(self, error: str, context: dict = None):
        """Log application errors"""
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "context": context or {}
        }
        
        self.logger.error(f"ERROR: {json.dumps(error_data)}")