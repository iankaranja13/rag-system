# src/generator.py

from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import logging
from utils.async_utils import ensure_event_loop 

from src.retriever import MedicalRetriever
from config.settings import Config

class MedicalResponseGenerator:
    def __init__(self):
        ensure_event_loop()
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=Config.TEMPERATURE,
            max_output_tokens=Config.MAX_TOKENS
        )
        
        self.retriever = MedicalRetriever()
        
        # Define prompt templates
        self._setup_prompt_templates()
    
    def _setup_prompt_templates(self):
        """Setup prompt templates for different tasks"""
        
        # Q&A Template
        self.qa_template = PromptTemplate(
            template="""You are a knowledgeable medical research assistant. Answer the user's question based on the provided medical documents and research papers.

**Important Guidelines:**
- Base your answer primarily on the provided context
- If the context doesn't contain enough information, clearly state this
- Always cite your sources using the source numbers provided
- Use medical terminology appropriately but explain complex terms when necessary
- Be precise and evidence-based in your responses
- If you're uncertain about medical advice, recommend consulting healthcare professionals

**Context from Medical Literature:**
{context}

**Question:** {question}

**Answer:**""",
            input_variables=["context", "question"]
        )
        
        # Summarization Template
        self.summary_template = PromptTemplate(
            template="""You are a medical research assistant. Provide a comprehensive summary of the medical documents provided below.

**Instructions:**
- Create a structured summary covering key findings, methodologies, and conclusions
- Highlight important medical insights and research outcomes
- Organize the information in a logical, easy-to-follow format
- Include relevant statistics, study parameters, and clinical implications
- Use appropriate medical terminology with explanations where needed

**Medical Documents:**
{context}

**Summary:**""",
            input_variables=["context"]
        )
        
        # Medical Analysis Template
        self.analysis_template = PromptTemplate(
            template="""As a medical research assistant, analyze the provided medical literature to answer the specific research question.

**Analysis Guidelines:**
- Provide evidence-based analysis using the medical literature
- Compare findings across different studies if applicable
- Identify gaps in research or conflicting findings
- Discuss clinical implications and significance
- Suggest areas for further research if relevant

**Medical Literature:**
{context}

**Research Question:** {question}

**Analysis:**""",
            input_variables=["context", "question"]
        )
    
    def generate_response(self, query: str, collection_name: str, 
                         response_type: str = "qa") -> Dict[str, Any]:
        """Generate response based on query and collection"""
        
        try:
            # Get relevant context
            context = self.retriever.get_relevant_context(query, collection_name)
            
            if context == "No relevant documents found.":
                return {
                    "answer": "I couldn't find any relevant medical documents to answer your question. Please try rephrasing your query or ensure the document collection contains relevant information.",
                    "sources": [],
                    "context_used": "",
                    "success": False
                }
            
            # Select appropriate template
            if response_type == "summary":
                prompt_template = self.summary_template
                chain_input = {"context": context}
            elif response_type == "analysis":
                prompt_template = self.analysis_template
                chain_input = {"context": context, "question": query}
            else:  # Default to Q&A
                prompt_template = self.qa_template
                chain_input = {"context": context, "question": query}
            
            # Create and run the chain
            chain = (
                RunnablePassthrough() 
                | prompt_template 
                | self.llm 
                | StrOutputParser()
            )
            
            response = chain.invoke(chain_input)
            
            # Extract source information
            source_docs = self.retriever.retrieve_documents(query, collection_name)
            sources = self._extract_source_info(source_docs)
            
            return {
                "answer": response.strip(),
                "sources": sources,
                "context_used": context,
                "success": True
            }
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "sources": [],
                "context_used": "",
                "success": False
            }
    
    def generate_document_summary(self, collection_name: str, 
                                 document_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate summary of documents in collection"""
        
        try:
            # Get documents (filtered if specified)
            if document_filter:
                docs = self.retriever.search_specific_document("", collection_name, document_filter)
            else:
                # Get a representative sample of documents
                docs = self.retriever.retrieve_documents("medical research summary overview", collection_name)
            
            if not docs:
                return {
                    "summary": "No documents found to summarize.",
                    "document_count": 0,
                    "success": False
                }
            
            # Prepare context from documents
            context_parts = []
            for i, doc in enumerate(docs[:10]):  # Limit to 10 docs for summary
                source_info = self.retriever._format_source_info(doc.metadata)
                context_parts.append(f"Document {i+1} ({source_info}):\n{doc.page_content}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Generate summary
            chain = (
                RunnablePassthrough() 
                | self.summary_template 
                | self.llm 
                | StrOutputParser()
            )
            
            summary = chain.invoke({"context": context})
            
            return {
                "summary": summary.strip(),
                "document_count": len(docs),
                "sources": self._extract_source_info(docs),
                "success": True
            }
            
        except Exception as e:
            logging.error(f"Error generating document summary: {str(e)}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "document_count": 0,
                "success": False
            }
    
    def _extract_source_info(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents"""
        sources = []
        seen_sources = set()
        
        for doc in documents:
            metadata = doc.metadata
            source_key = f"{metadata.get('source_type', '')}_{metadata.get('file_name', '')}_{metadata.get('url', '')}"
            
            if source_key not in seen_sources:
                source_info = {
                    "type": metadata.get("source_type", "unknown"),
                    "title": metadata.get("title", "Unknown"),
                    "file_name": metadata.get("file_name", ""),
                    "url": metadata.get("url", ""),
                    "page": metadata.get("page", ""),
                    "chunk_id": metadata.get("chunk_id", "")
                }
                sources.append(source_info)
                seen_sources.add(source_key)
        
        return sources
    
    def compare_documents(self, query: str, collection_names: List[str]) -> Dict[str, Any]:
        """Compare responses across multiple document collections"""
        
        results = {}
        all_sources = []
        
        for collection in collection_names:
            try:
                result = self.generate_response(query, collection, "analysis")
                results[collection] = result
                if result["success"]:
                    all_sources.extend(result["sources"])
            except Exception as e:
                results[collection] = {
                    "answer": f"Error processing collection {collection}: {str(e)}",
                    "success": False
                }
        
        # Generate comparative analysis
        comparison_context = ""
        for collection, result in results.items():
            if result.get("success"):
                comparison_context += f"\n\n**{collection.upper()} COLLECTION:**\n{result['answer']}"
        
        if comparison_context:
            comparative_prompt = PromptTemplate(
                template="""Based on the analysis from different medical document collections, provide a comprehensive comparative analysis addressing the research question.

**Research Question:** {question}

**Analysis from Different Collections:**
{context}

**Comparative Analysis:**
- Synthesize findings across collections
- Identify common themes and contradictions
- Highlight unique insights from each collection
- Provide an integrated conclusion

**Synthesis:**""",
                input_variables=["question", "context"]
            )
            
            try:
                chain = (
                    RunnablePassthrough() 
                    | comparative_prompt 
                    | self.llm 
                    | StrOutputParser()
                )
                
                synthesis = chain.invoke({
                    "question": query,
                    "context": comparison_context
                })
                
                return {
                    "individual_results": results,
                    "synthesis": synthesis.strip(),
                    "all_sources": all_sources,
                    "success": True
                }
            except Exception as e:
                return {
                    "individual_results": results,
                    "synthesis": f"Error generating synthesis: {str(e)}",
                    "all_sources": all_sources,
                    "success": False
                }
        
        return {
            "individual_results": results,
            "synthesis": "No successful analyses to synthesize.",
            "all_sources": all_sources,
            "success": False
        }