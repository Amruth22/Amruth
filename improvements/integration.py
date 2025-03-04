"""
Integration module for all RAG system improvements
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Import improvement modules
from improvements.advanced_retrieval import get_advanced_retriever
from improvements.multimodal_rag import ingest_multimodal_content
from improvements.evaluation import RAGEvaluator, FeedbackLoop
from improvements.structured_output import StructuredOutputGenerator, ToolUseManager

class EnhancedRAGSystem:
    """
    Enhanced RAG system that integrates all improvements
    """
    
    def __init__(
        self,
        vector_store_path: str = "faiss_index",
        embedding_model: str = "text-embedding-3-large",
        llm_model: str = "gpt-4o",
        retriever_type: str = "hybrid"
    ):
        """
        Initialize the enhanced RAG system
        
        Args:
            vector_store_path: Path to the vector store
            embedding_model: Name of the embedding model to use
            llm_model: Name of the LLM to use
            retriever_type: Type of retriever to use
        """
        # Initialize models
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        
        # Initialize vector store
        self.vector_store_path = vector_store_path
        self.vector_store = self._initialize_vector_store()
        
        # Initialize retriever
        self.retriever_type = retriever_type
        self.retriever = get_advanced_retriever(self.vector_store, retriever_type)
        
        # Initialize components
        self.evaluator = RAGEvaluator()
        self.feedback_loop = FeedbackLoop(self.vector_store)
        self.structured_output_generator = StructuredOutputGenerator()
        self.tool_manager = ToolUseManager()
        
        # Configuration
        self.config = {
            "auto_evaluation": True,
            "use_tools": True,
            "structured_output": True,
            "feedback_collection": True,
            "retriever_type": retriever_type,
            "num_documents": 5
        }
    
    def _initialize_vector_store(self) -> FAISS:
        """
        Initialize the vector store
        
        Returns:
            Initialized vector store
        """
        try:
            # Try to load from disk if it exists
            vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)
            print(f"Loaded existing vector store from {self.vector_store_path}")
        except:
            # Create a new one if it doesn't exist
            vector_store = FAISS.from_documents(
                [Document(page_content="Initial document to create the vector store.")], 
                self.embeddings
            )
            vector_store.save_local(self.vector_store_path)
            print(f"Created new vector store at {self.vector_store_path}")
        
        return vector_store
    
    def ingest_content(self, urls: List[str]) -> Dict[str, Any]:
        """
        Ingest content from URLs into the vector store
        
        Args:
            urls: List of URLs to ingest
            
        Returns:
            Ingestion results
        """
        results = []
        total_documents = 0
        
        for url in urls:
            try:
                # Use multimodal ingestion
                documents = ingest_multimodal_content(url)
                
                # Add to vector store
                self.vector_store.add_documents(documents)
                
                # Save updated vector store
                self.vector_store.save_local(self.vector_store_path)
                
                results.append({
                    "url": url,
                    "status": "success",
                    "documents_count": len(documents),
                    "file_type": documents[0].metadata.get("file_type", "unknown") if documents else "unknown"
                })
                
                total_documents += len(documents)
            
            except Exception as e:
                results.append({
                    "url": url,
                    "status": "error",
                    "error": str(e)
                })
        
        # Reinitialize retriever after ingestion
        self.retriever = get_advanced_retriever(self.vector_store, self.retriever_type)
        
        return {
            "total_urls": len(urls),
            "successful_urls": sum(1 for r in results if r["status"] == "success"),
            "failed_urls": sum(1 for r in results if r["status"] == "error"),
            "total_documents_ingested": total_documents,
            "details": results
        }
    
    def process_query(
        self,
        query: str,
        chat_history: List[Dict[str, str]] = None,
        output_format: str = "text",
        num_documents: int = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the enhanced RAG system
        
        Args:
            query: The user's query
            chat_history: Optional chat history
            output_format: Format of the output (text, search_result, analysis, comparison)
            num_documents: Number of documents to retrieve
            
        Returns:
            Processing results
        """
        if chat_history is None:
            chat_history = []
        
        if num_documents is None:
            num_documents = self.config["num_documents"]
        
        # Check if query requires tools
        tool_detection = None
        tool_result = None
        
        if self.config["use_tools"]:
            tool_detection = self.tool_manager.detect_tool_needs(query)
            
            if tool_detection.get("needs_tool", False):
                tool_name = tool_detection.get("tool_name")
                tool_input = tool_detection.get("tool_input")
                
                if tool_name and tool_input:
                    tool_result = self.tool_manager.execute_tool(tool_name, tool_input)
        
        # Retrieve documents
        retrieved_documents = self.retriever.get_relevant_documents(query)
        retrieved_documents = retrieved_documents[:num_documents]
        
        # Extract content and metadata
        context_content = [doc.page_content for doc in retrieved_documents]
        context_metadata = [doc.metadata for doc in retrieved_documents]
        
        # Generate response based on output format
        response = None
        structured_output = None
        
        if tool_result and self.config["use_tools"]:
            # Integrate tool result with RAG context
            response = self.tool_manager.integrate_tool_result(
                query=query,
                tool_name=tool_detection.get("tool_name", "unknown_tool"),
                tool_input=tool_detection.get("tool_input", ""),
                tool_result=tool_result,
                context=context_content
            )
        
        elif output_format == "text" or not self.config["structured_output"]:
            # Generate text response
            from langchain_core.prompts import ChatPromptTemplate
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that answers questions based on the provided context.
                Provide a comprehensive answer that addresses the user's query.
                If the context doesn't contain relevant information, acknowledge that.
                Cite sources when appropriate."""),
                ("human", """Query: {query}
                
                Context:
                {context}
                
                Generate a helpful answer:""")
            ])
            
            # Format the context
            formatted_context = "\n\n".join([
                f"Source: {context_metadata[i].get('source', 'Unknown')}\n{context_content[i]}"
                for i in range(len(context_content))
            ])
            
            # Generate the response
            llm_response = self.llm.invoke(
                prompt.format_messages(
                    query=query,
                    context=formatted_context
                )
            )
            
            response = llm_response.content
        
        elif output_format == "search_result":
            # Generate structured search result
            result = self.structured_output_generator.generate_search_result(
                query=query,
                context=context_content,
                metadata=context_metadata
            )
            
            response = result.answer
            structured_output = result.model_dump()
        
        elif output_format == "analysis":
            # Generate structured analysis result
            result = self.structured_output_generator.generate_analysis_result(
                query=query,
                context=context_content,
                metadata=context_metadata
            )
            
            response = result.summary
            structured_output = result.model_dump()
        
        elif output_format == "comparison":
            # Extract items to compare from query
            # This is a simple implementation - in production, use more sophisticated extraction
            import re
            items_pattern = r"compare\s+(.+?)\s+and\s+(.+?)(?:\s+and\s+(.+?))?(?:\s+|$)"
            match = re.search(items_pattern, query.lower())
            
            items_to_compare = []
            if match:
                items_to_compare = [group for group in match.groups() if group]
            
            # If no items found, use a default approach
            if not items_to_compare:
                items_to_compare = ["Item 1", "Item 2"]
            
            # Generate structured comparison result
            result = self.structured_output_generator.generate_comparison_result(
                query=query,
                items_to_compare=items_to_compare,
                context=context_content,
                metadata=context_metadata
            )
            
            response = f"Comparison: {result.recommendation}"
            structured_output = result.model_dump()
        
        # Perform evaluation if enabled
        evaluation = None
        if self.config["auto_evaluation"]:
            evaluation = self.evaluator.evaluate_response(
                query=query,
                response=response,
                retrieved_documents=retrieved_documents
            )
        
        # Prepare result
        result = {
            "query": query,
            "response": response,
            "sources": [doc.metadata.get("source", "Unknown") for doc in retrieved_documents],
            "num_documents_retrieved": len(retrieved_documents),
            "retriever_type": self.retriever_type
        }
        
        if structured_output:
            result["structured_output"] = structured_output
        
        if tool_detection and tool_detection.get("needs_tool", False):
            result["tool_used"] = {
                "name": tool_detection.get("tool_name"),
                "input": tool_detection.get("tool_input"),
                "result": tool_result
            }
        
        if evaluation:
            result["evaluation"] = evaluation
        
        return result
    
    def collect_feedback(
        self,
        query: str,
        response: str,
        retrieved_documents: List[Document],
        user_rating: int,
        user_feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Collect user feedback on a response
        
        Args:
            query: The user's query
            response: The generated response
            retrieved_documents: The documents retrieved for the query
            user_rating: User rating (1-5)
            user_feedback: Optional user feedback text
            
        Returns:
            Feedback processing results
        """
        if not self.config["feedback_collection"]:
            return {"status": "feedback_collection_disabled"}
        
        return self.feedback_loop.process_user_feedback(
            query=query,
            response=response,
            retrieved_documents=retrieved_documents,
            user_rating=user_rating,
            user_feedback=user_feedback
        )
    
    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the system configuration
        
        Args:
            new_config: New configuration values
            
        Returns:
            Updated configuration
        """
        # Update configuration
        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value
        
        # Update retriever if type changed
        if "retriever_type" in new_config and new_config["retriever_type"] != self.retriever_type:
            self.retriever_type = new_config["retriever_type"]
            self.retriever = get_advanced_retriever(self.vector_store, self.retriever_type)
        
        return self.config
    
    def analyze_system_performance(self) -> Dict[str, Any]:
        """
        Analyze the system performance based on evaluation logs
        
        Returns:
            Performance analysis
        """
        return self.evaluator.analyze_evaluation_logs()