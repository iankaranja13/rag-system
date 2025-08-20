# src/utils/async_utils.py
import asyncio

def ensure_event_loop():
    """Ensure there's an asyncio event loop in the current thread (for Streamlit compatibility)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
