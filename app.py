# =========================
# Top of app.py - Apple Silicon Safe Setup
# =========================

import os

# Limit threads to prevent segmentation fault on Apple Silicon
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Apple Silicon GPU fallback

# =========================
# Standard imports
# =========================
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go

# =========================
# Local imports
# =========================
from config.settings import Config
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager
from src.generator import MedicalResponseGenerator
from src.retriever import MedicalRetriever
from utils.helpers import format_sources, create_download_link


# Page configuration
st.set_page_config(
    page_title="Medical Research Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .collection-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .source-box {
        background-color: #e8f4fd;
        padding: 0.8rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_collection" not in st.session_state:
        st.session_state.current_collection = "general_medicine"
    if "document_processor" not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager()
    if "response_generator" not in st.session_state:
        st.session_state.response_generator = MedicalResponseGenerator()
    if "retriever" not in st.session_state:
        st.session_state.retriever = MedicalRetriever()

# Sidebar functions
def render_sidebar():
    """Render sidebar with navigation and controls"""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/1f77b4/ffffff?text=Medical+RAG", width=200)
        
        st.markdown("### 🏥 Medical Research Assistant")
        st.markdown("*AI-powered medical literature analysis*")
        
        # Navigation
        page = st.selectbox(
            "Navigate to:",
            ["🔍 Query Interface", "📚 Document Management", "📊 Analytics", "⚙️ Settings"],
            key="navigation"
        )
        
        st.markdown("---")
        
        # Collection selector
        st.markdown("### Document Collections")
        collection_options = {}
        for key, value in Config.DOCUMENT_COLLECTIONS.items():
            collection_options[f"{value['name']} - {value['description'][:30]}..."] = key
        
        selected_collection = st.selectbox(
            "Select Collection:",
            list(collection_options.keys()),
            key="collection_selector"
        )
        
        st.session_state.current_collection = collection_options[selected_collection]
        
        # Collection info
        collection_info = st.session_state.retriever.get_document_statistics(
            st.session_state.current_collection
        )
        
        with st.expander("📋 Collection Info"):
            st.write(f"**Documents:** {collection_info['total_chunks']}")
            st.write(f"**Status:** {'✅ Ready' if collection_info['vector_store_exists'] else '❌ Not Ready'}")
            if collection_info.get('last_updated'):
                last_updated = datetime.fromtimestamp(collection_info['last_updated'])
                st.write(f"**Updated:** {last_updated.strftime('%Y-%m-%d %H:%M')}")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### Quick Actions")
        if st.button("🔄 Refresh Collections"):
            st.rerun()
        
        if st.button("🧹 Clear Chat History"):
            st.session_state.messages = []
            st.success("Chat history cleared!")
        
        return page

# Main pages
def render_query_interface():
    """Render the main query interface"""
    st.markdown('<h1 class="main-header">🏥 Medical Research Assistant</h1>', unsafe_allow_html=True)
    
    # Warning about medical advice
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for research and educational purposes only. 
        Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment.
    </div>
    """, unsafe_allow_html=True)
    
    # Query modes
    col1, col2, col3 = st.columns(3)
    with col1:
        query_mode = st.selectbox("Query Mode:", ["❓ Q&A", "📄 Summary", "🔬 Analysis"])
    with col2:
        retrieval_method = st.selectbox("Retrieval:", ["Dense", "Hybrid"])
    with col3:
        rerank_method = st.selectbox("Reranking:", ["Cross-Encoder", "Cohere", "None"])
    
    # Main query interface
    st.markdown("### Ask Your Medical Research Question")
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What are the latest findings on cardiovascular disease prevention?",
        height=100,
        key="main_query"
    )
    
    # Submit button and options
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        submit_query = st.button("🔍 Search & Analyze", type="primary", use_container_width=True)
    with col2:
        include_sources = st.checkbox("Show Sources", value=True)
    with col3:
        show_context = st.checkbox("Show Context", value=False)
    
    # Process query
    if submit_query and query.strip():
        with st.spinner("Analyzing medical literature..."):
            # Determine response type
            if query_mode == "📄 Summary":
                response_type = "summary"
            elif query_mode == "🔬 Analysis":
                response_type = "analysis"
            else:
                response_type = "qa"
            
            # Generate response
            result = st.session_state.response_generator.generate_response(
                query,
                st.session_state.current_collection,
                response_type
            )
            
            # Display results
            if result["success"]:
                st.markdown("### 📋 Analysis Results")
                st.markdown(result["answer"])
                
                # Display sources
                if include_sources and result["sources"]:
                    st.markdown("### 📚 Sources")
                    for i, source in enumerate(result["sources"][:5]):  # Limit to 5 sources
                        with st.expander(f"Source {i+1}: {source['title'][:50]}..."):
                            st.write(f"**Type:** {source['type'].title()}")
                            if source['file_name']:
                                st.write(f"**File:** {source['file_name']}")
                            if source['url']:
                                st.write(f"**URL:** {source['url']}")
                
                # Display context if requested
                if show_context:
                    with st.expander("🔍 Retrieved Context"):
                        st.text(result["context_used"][:2000] + "..." if len(result["context_used"]) > 2000 else result["context_used"])
                
                # Add to chat history
                st.session_state.messages.append({
                    "query": query,
                    "response": result["answer"],
                    "sources": result["sources"],
                    "timestamp": datetime.now(),
                    "collection": st.session_state.current_collection
                })
                
            else:
                st.error(f"❌ Error: {result['answer']}")
    
    # Chat history
    if st.session_state.messages:
        st.markdown("### 💬 Recent Queries")
        for i, message in enumerate(reversed(st.session_state.messages[-5:])):  # Show last 5
            with st.expander(f"Query: {message['query'][:60]}..." if len(message['query']) > 60 else message['query']):
                st.write(f"**Collection:** {message['collection']}")
                st.write(f"**Time:** {message['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Response:** {message['response'][:300]}..." if len(message['response']) > 300 else message['response'])

def render_document_management():
    """Render document management interface"""
    st.markdown("# 📚 Document Management")
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "🗂️ Manage Collections", "🔧 Process Documents"])
    
    with tab1:
        st.markdown("### Upload New Documents")
        
        # Collection selector for upload
        upload_collection = st.selectbox(
            "Select target collection:",
            list(Config.DOCUMENT_COLLECTIONS.keys()),
            format_func=lambda x: Config.DOCUMENT_COLLECTIONS[x]["name"]
        )
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose files to upload:",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'md']
        )
        
        # URL input for web scraping
        st.markdown("### Or Add Web Sources")
        urls = st.text_area(
            "Enter URLs (one per line):",
            placeholder="https://example.com/medical-article\nhttps://another-source.com/research"
        )
        
        # YouTube URLs
        youtube_urls = st.text_area(
            "Enter YouTube URLs (one per line):",
            placeholder="https://www.youtube.com/watch?v=example"
        )
        
        if st.button("📥 Process All Sources"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_documents = []
            
            # Process uploaded files
            if uploaded_files:
                for i, file in enumerate(uploaded_files):
                    progress_bar.progress((i + 1) / (len(uploaded_files) + len(urls.split('\n')) + len(youtube_urls.split('\n'))))
                    status_text.text(f"Processing {file.name}...")
                    
                    # Save file temporarily
                    temp_path = os.path.join(Config.DOCUMENTS_DIR, file.name)
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Process file
                    if file.name.endswith('.pdf'):
                        docs = st.session_state.document_processor.load_pdf(temp_path)
                    else:
                        docs = st.session_state.document_processor.load_text_file(temp_path)
                    
                    all_documents.extend(docs)
            
            # Process URLs
            if urls.strip():
                url_list = [url.strip() for url in urls.split('\n') if url.strip()]
                for url in url_list:
                    status_text.text(f"Scraping {url}...")
                    docs = st.session_state.document_processor.scrape_website(url)
                    all_documents.extend(docs)
            
            # Process YouTube URLs
            if youtube_urls.strip():
                youtube_list = [url.strip() for url in youtube_urls.split('\n') if url.strip()]
                for url in youtube_list:
                    status_text.text(f"Processing YouTube: {url}")
                    docs = st.session_state.document_processor.load_youtube_transcript(url)
                    all_documents.extend(docs)
            
            if all_documents:
                # Create vector store
                status_text.text("Creating embeddings...")
                st.session_state.embedding_manager.create_vector_store(all_documents, upload_collection)
                
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                st.success(f"Successfully processed {len(all_documents)} document chunks!")
            else:
                st.warning("No documents were processed. Please check your inputs.")
    
    with tab2:
        st.markdown("### Collection Overview")
        
        # Display collection statistics
        for collection_key, collection_info in Config.DOCUMENT_COLLECTIONS.items():
            stats = st.session_state.retriever.get_document_statistics(collection_key)
            
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    st.markdown(f"**{collection_info['name']}**")
                    st.caption(collection_info['description'])
                
                with col2:
                    st.metric("Documents", stats['total_chunks'])
                
                with col3:
                    status = "✅ Ready" if stats['vector_store_exists'] else "❌ Empty"
                    st.write(f"**Status:** {status}")
                
                with col4:
                    if st.button("🗑️", key=f"delete_{collection_key}"):
                        st.session_state.embedding_manager.delete_vector_store(collection_key)
                        st.success(f"Deleted {collection_key} collection")
                        st.rerun()
                
                st.markdown("---")
    
    with tab3:
        st.markdown("### Batch Processing")
        
        # Directory processing
        st.markdown("#### Process Directory")
        directory_path = st.text_input("Directory path:")
        target_collection = st.selectbox(
            "Target collection:",
            list(Config.DOCUMENT_COLLECTIONS.keys()),
            format_func=lambda x: Config.DOCUMENT_COLLECTIONS[x]["name"],
            key="batch_collection"
        )
        
        if st.button("🔄 Process Directory"):
            if os.path.exists(directory_path):
                with st.spinner("Processing directory..."):
                    documents = st.session_state.document_processor.process_documents_from_directory(
                        directory_path, target_collection
                    )
                    
                    if documents:
                        st.session_state.embedding_manager.create_vector_store(documents, target_collection)
                        st.success(f"Successfully processed {len(documents)} documents!")
                    else:
                        st.warning("No documents found or processed.")
            else:
                st.error("Directory does not exist!")
        
        # Sample data download
        st.markdown("#### 📋 Sample Medical Documents")
        st.info("To get started quickly, you can download sample medical documents.")
        
        if st.button("📥 Download Sample Dataset"):
            st.markdown("""
            **Suggested Medical Document Sources:**
            
            1. **PubMed Central (Open Access)**
               - URL: https://www.ncbi.nlm.nih.gov/pmc/
               - Free full-text research articles
            
            2. **WHO Publications**
               - URL: https://www.who.int/publications
               - Guidelines and reports
            
            3. **Medical YouTube Channels**
               - Osmosis: Educational medical content
               - Khan Academy Medicine
               - MedCram Medical Lectures
            
            4. **Clinical Guidelines**
               - American Heart Association guidelines
               - CDC recommendations
               - NIH clinical protocols
            """)

# Enhanced query interface with medical specialization
def render_enhanced_query_interface():
    """Enhanced query interface with medical specializations"""
    
    # Medical specialty quick queries
    st.markdown("### 🏥 Quick Medical Queries")
    
    specialty_queries = {
        "Cardiology": [
            "Latest guidelines for heart failure management",
            "Risk factors for coronary artery disease",
            "Blood pressure management protocols"
        ],
        "Oncology": [
            "Immunotherapy treatment options",
            "Cancer screening guidelines",
            "Palliative care protocols"
        ],
        "General Medicine": [
            "Diabetes management guidelines",
            "Hypertension treatment protocols",
            "Preventive care recommendations"
        ]
    }
    
    specialty = st.selectbox("Medical Specialty:", list(specialty_queries.keys()))
    
    col1, col2 = st.columns([3, 1])
    with col1:
        sample_query = st.selectbox(
            "Sample Queries:",
            specialty_queries[specialty],
            key="sample_query"
        )
    with col2:
        if st.button("🔍 Use Sample Query"):
            st.session_state.main_query = sample_query
            st.rerun()

# Add this function to the main app
def render_query_interface():
    """Render the main query interface"""
    st.markdown('<h1 class="main-header">🏥 Medical Research Assistant</h1>', unsafe_allow_html=True)
    
    # Warning about medical advice
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for research and educational purposes only. 
        Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment.
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced query interface
    render_enhanced_query_interface()
    
    # Query modes
    col1, col2, col3 = st.columns(3)
    with col1:
        query_mode = st.selectbox("Query Mode:", ["❓ Q&A", "📄 Summary", "🔬 Analysis"])
    with col2:
        retrieval_method = st.selectbox("Retrieval:", ["Dense", "Hybrid"])
    with col3:
        rerank_method = st.selectbox("Reranking:", ["Cross-Encoder", "Cohere", "None"])
    
    # Main query interface
    st.markdown("### Ask Your Medical Research Question")
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What are the latest findings on cardiovascular disease prevention?",
        height=100,
        key="main_query"
    )
    
    # Submit button and options
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        submit_query = st.button("🔍 Search & Analyze", type="primary", use_container_width=True)
    with col2:
        include_sources = st.checkbox("Show Sources", value=True)
    with col3:
        show_context = st.checkbox("Show Context", value=False)
    
    # Process query
    if submit_query and query.strip():
        with st.spinner("Analyzing medical literature..."):
            # Determine response type
            if query_mode == "📄 Summary":
                response_type = "summary"
            elif query_mode == "🔬 Analysis":
                response_type = "analysis"
            else:
                response_type = "qa"
            
            # Generate response
            result = st.session_state.response_generator.generate_response(
                query,
                st.session_state.current_collection,
                response_type
            )
            
            # Display results
            if result["success"]:
                st.markdown("### 📋 Analysis Results")
                st.markdown(result["answer"])
                
                # Display sources
                if include_sources and result["sources"]:
                    st.markdown("### 📚 Sources")
                    formatted_sources = format_sources(result["sources"][:5])
                    st.markdown(formatted_sources)
                
                # Display context if requested
                if show_context:
                    with st.expander("🔍 Retrieved Context"):
                        st.text(result["context_used"][:2000] + "..." if len(result["context_used"]) > 2000 else result["context_used"])
                
                # Download results
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save Results"):
                        save_data = {
                            "query": query,
                            "answer": result["answer"],
                            "sources": result["sources"],
                            "timestamp": datetime.now().isoformat(),
                            "collection": st.session_state.current_collection
                        }
                        
                        filename = f"medical_rag_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        download_link = create_download_link(
                            json.dumps(save_data, indent=2),
                            filename,
                            "📥 Download Results"
                        )
                        st.markdown(download_link, unsafe_allow_html=True)
                
                # Add to chat history
                st.session_state.messages.append({
                    "query": query,
                    "response": result["answer"],
                    "sources": result["sources"],
                    "timestamp": datetime.now(),
                    "collection": st.session_state.current_collection
                })
                
            else:
                st.error(f"❌ Error: {result['answer']}")
    
    # Chat history with enhanced display
    if st.session_state.messages:
        st.markdown("### 💬 Recent Queries")
        
        # Export chat history option
        if len(st.session_state.messages) > 0:
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("📤 Export History"):
                    export_data = export_chat_history(st.session_state.messages)
                    filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    download_link = create_download_link(export_data, filename, "📥 Download History")
                    st.markdown(download_link, unsafe_allow_html=True)
        
        # Display recent messages
        for i, message in enumerate(reversed(st.session_state.messages[-5:])):  # Show last 5
            with st.expander(f"Query: {truncate_text(message['query'], 60)}"):
                st.write(f"**Collection:** {message['collection']}")
                st.write(f"**Time:** {format_timestamp(message['timestamp'])}")
                st.write(f"**Query:** {message['query']}")
                st.write(f"**Response:** {truncate_text(message['response'], 300)}")
                st.write(f"**Sources:** {len(message['sources'])} documents")
                
                if st.button("🔄 Rerun Query", key=f"rerun_{i}"):
                    st.session_state.main_query = message['query']
                    st.rerun()

def render_analytics():
    """Render analytics and insights page"""
    st.markdown("# 📊 Analytics & Insights")
    
    # Collection comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Collection Statistics")
        
        # Gather statistics for all collections
        stats_data = []
        for collection_key, collection_info in Config.DOCUMENT_COLLECTIONS.items():
            stats = st.session_state.retriever.get_document_statistics(collection_key)
            stats_data.append({
                "Collection": collection_info["name"],
                "Documents": stats["total_chunks"],
                "Status": "Ready" if stats["vector_store_exists"] else "Empty"
            })
        
        if stats_data:
            df = pd.DataFrame(stats_data)
            
            # Create bar chart
            fig = px.bar(
                df, 
                x="Collection", 
                y="Documents", 
                color="Status",
                title="Documents per Collection"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.dataframe(df, use_container_width=True)
    
    with col2:
        st.markdown("### Query History Analysis")
        
        if st.session_state.messages:
            # Analyze query history
            query_data = []
            for msg in st.session_state.messages:
                query_data.append({
                    "Collection": msg["collection"],
                    "Query Length": len(msg["query"]),
                    "Response Length": len(msg["response"]),
                    "Sources Count": len(msg["sources"]),
                    "Timestamp": msg["timestamp"]
                })
            
            query_df = pd.DataFrame(query_data)
            
            # Queries per collection
            queries_per_collection = query_df["Collection"].value_counts()
            fig_pie = px.pie(
                values=queries_per_collection.values,
                names=queries_per_collection.index,
                title="Queries by Collection"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Query trends over time
            query_df["Hour"] = query_df["Timestamp"].dt.hour
            hourly_queries = query_df.groupby("Hour").size()
            
            fig_line = px.line(
                x=hourly_queries.index,
                y=hourly_queries.values,
                title="Query Activity by Hour"
            )
            fig_line.update_xaxis(title="Hour of Day")
            fig_line.update_yaxis(title="Number of Queries")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No query history available yet. Start asking questions to see analytics!")
    
    # Comparative analysis section
    st.markdown("### 🔍 Comparative Collection Analysis")
    
    comparison_query = st.text_input(
        "Enter a query to compare across collections:",
        placeholder="e.g., treatment guidelines for diabetes"
    )
    
    selected_collections = st.multiselect(
        "Select collections to compare:",
        list(Config.DOCUMENT_COLLECTIONS.keys()),
        format_func=lambda x: Config.DOCUMENT_COLLECTIONS[x]["name"],
        default=list(Config.DOCUMENT_COLLECTIONS.keys())[:2]
    )
    
    if st.button("🔄 Compare Collections") and comparison_query and selected_collections:
        with st.spinner("Comparing across collections..."):
            comparison_result = st.session_state.response_generator.compare_documents(
                comparison_query, selected_collections
            )
            
            if comparison_result["success"]:
                st.markdown("### 📋 Comparative Analysis")
                st.markdown(comparison_result["synthesis"])
                
                # Individual results
                with st.expander("📊 Individual Collection Results"):
                    for collection, result in comparison_result["individual_results"].items():
                        if result.get("success"):
                            st.markdown(f"#### {Config.DOCUMENT_COLLECTIONS[collection]['name']}")
                            st.markdown(result["answer"])
                            st.markdown("---")

def render_settings():
    """Render settings and configuration page"""
    st.markdown("# ⚙️ Settings & Configuration")
    
    tab1, tab2, tab3 = st.tabs(["🔧 System Settings", "🔑 API Configuration", "📋 About"])
    
    with tab1:
        st.markdown("### Retrieval Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            new_chunk_size = st.slider("Chunk Size", 500, 2000, Config.CHUNK_SIZE)
            new_chunk_overlap = st.slider("Chunk Overlap", 50, 500, Config.CHUNK_OVERLAP)
            
        with col2:
            new_top_k = st.slider("Top-K Retrieval", 5, 20, Config.TOP_K_RETRIEVAL)
            new_top_k_rerank = st.slider("Top-K Rerank", 3, 10, Config.TOP_K_RERANK)
        
        st.markdown("### Generation Settings")
        new_temperature = st.slider("Temperature", 0.0, 1.0, Config.TEMPERATURE, 0.1)
        new_max_tokens = st.slider("Max Tokens", 500, 4000, Config.MAX_TOKENS)
        
        if st.button("💾 Save Settings"):
            # Note: In a real application, you'd save these to a config file
            st.success("Settings saved! (Note: Restart app to apply changes)")
    
    with tab2:
        st.markdown("### API Key Status")
        
        # Check API key status
        api_status = []
        
        if Config.GOOGLE_API_KEY:
            api_status.append({"Service": "Google Gemini", "Status": "✅ Configured"})
        else:
            api_status.append({"Service": "Google Gemini", "Status": "❌ Missing"})
        
        if Config.OPENAI_API_KEY:
            api_status.append({"Service": "OpenAI Embeddings", "Status": "✅ Configured"})
        else:
            api_status.append({"Service": "OpenAI Embeddings", "Status": "❌ Missing"})
        
        if Config.COHERE_API_KEY:
            api_status.append({"Service": "Cohere Rerank", "Status": "✅ Configured"})
        else:
            api_status.append({"Service": "Cohere Rerank", "Status": "⚠️ Optional"})
        
        df_api = pd.DataFrame(api_status)
        st.dataframe(df_api, use_container_width=True)
        
        st.markdown("### Environment Setup")
        st.code("""
# Create .env file with your API keys:
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here  # Optional
        """, language="bash")
    
    with tab3:
        st.markdown("### 🏥 Medical Research Assistant")
        st.markdown("""
        This application helps healthcare professionals and researchers quickly find and analyze 
        information from medical literature, research papers, and clinical guidelines.
        
        **Features:**
        - 🔍 Intelligent document search and retrieval
        - 🤖 AI-powered response generation using Google Gemini
        - 📊 Multiple reranking strategies for improved accuracy
        - 📚 Support for PDFs, text files, websites, and YouTube transcripts
        - 🏥 Specialized for medical and healthcare content
        - 📋 Document summarization and comparative analysis
        
        **Technology Stack:**
        - **LLM:** Google Gemini 1.5 Flash
        - **Embeddings:** OpenAI text-embedding-3-small
        - **Vector Store:** FAISS
        - **Framework:** LangChain + Streamlit
        - **Reranking:** Cross-Encoder, Cohere, BM25+Dense
        
        **Version:** 1.0.0
        """)

import json

def truncate_text(text, max_length):
    """Truncate text to a maximum length, adding ellipsis if needed."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def export_chat_history(messages):
    """Export chat history as a JSON string."""
    def serialize_message(msg):
        # Convert datetime to isoformat string for JSON serialization
        msg_copy = msg.copy()
        if isinstance(msg_copy.get("timestamp"), datetime):
            msg_copy["timestamp"] = msg_copy["timestamp"].isoformat()
        return msg_copy
    return json.dumps([serialize_message(m) for m in messages], indent=2)

def format_timestamp(ts):
    """Format a timestamp (datetime or string) for display."""
    if isinstance(ts, datetime):
        return ts.strftime('%Y-%m-%d %H:%M:%S')
    try:
        # Try parsing ISO format string
        return datetime.fromisoformat(ts).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return str(ts)

# Main app function
def main():
    """Main application function"""
    
    # Initialize
    initialize_session_state()
    
    # Validate API keys
    try:
        Config.validate_api_keys()
        Config.create_directories()
    except ValueError as e:
        st.error(f"Configuration Error: {str(e)}")
        st.info("Please check your .env file and ensure all required API keys are set.")
        st.stop()
    
    # Render sidebar and get current page
    current_page = render_sidebar()
    
    # Render appropriate page
    if current_page == "🔍 Query Interface":
        render_query_interface()
    elif current_page == "📚 Document Management":
        render_document_management()
    elif current_page == "📊 Analytics":
        render_analytics()
    elif current_page == "⚙️ Settings":
        render_settings()

if __name__ == "__main__":
    main()


