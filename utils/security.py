import streamlit as st
import hashlib
import hmac
from datetime import datetime

class SecurityManager:
    
    @staticmethod
    def validate_api_key(api_key: str, service: str) -> bool:
        """Validate API key format"""
        if service == "openai":
            return api_key.startswith("sk-") and len(api_key) > 20
        elif service == "google":
            return len(api_key) > 30
        elif service == "cohere":
            return len(api_key) > 20
        return False
    
    @staticmethod
    def sanitize_input(user_input: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        # Remove potentially harmful characters
        dangerous_chars = ['<', '>', '"', "'", '&', '%', ';']
        sanitized = user_input
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized.strip()
    
    @staticmethod
    def rate_limit_check(user_id: str, max_requests: int = 100) -> bool:
        """Simple rate limiting implementation"""
        # In production, use Redis or database for rate limiting
        if "rate_limit" not in st.session_state:
            st.session_state.rate_limit = {}
        
        current_hour = datetime.now().hour
        key = f"{user_id}_{current_hour}"
        
        current_count = st.session_state.rate_limit.get(key, 0)
        
        if current_count >= max_requests:
            return False
        
        st.session_state.rate_limit[key] = current_count + 1
        return True