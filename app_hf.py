import os
import streamlit as st
from app import main  # assumes app.py has a main() function

if "SPACE_ID" in os.environ:
    st.set_page_config(
        page_title="Medical Research Assistant",
        page_icon="🏥",
        layout="wide"
    )

# Run the main app
main()
