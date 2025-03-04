"""
Streaming response capability for the langgraph RAG system
"""

import asyncio
import json
from typing import Dict, List, Any, AsyncGenerator, Callable, Optional, Union
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.utils import AddableDict

async def stream_process_query(
    query: str,
    chat_history: List[Dict[str, str]] = None,
    retriever = None,
    callback: Optional[Callable[[str], None]] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Process a user query with streaming response capability.
    
    Args:
        query: The user's query
        chat_history: Optional chat history
        retriever: The retriever to use (if None, will use the default retriever)
        callback: Optional callback function to receive chunks
        
    Yields:
        Dictionary containing response chunks and metadata
    """
    # Initialize streaming LLM
    streaming_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        streaming=True
    )
    
    # Use provided retriever or import the default one
    if retriever is None:
        from langgraph_rag_system import retrieve_context
        documents = retrieve_context(query)
    else:
        documents = retriever.get_relevant_documents(query)
    
    # Format context for the prompt
    formatted_context = "\n\n".join([
        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in documents
    ])
    
    # Create the prompt
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
    
    # Create the chain
    chain = (
        {"query": RunnablePassthrough(), "context": lambda _: formatted_context}
        | prompt
        | streaming_llm
    )
    
    # Stream the response
    response_chunks = []
    metadata = {
        "query": query,
        "num_documents": len(documents),
        "sources": [doc.metadata.get("source", "Unknown") for doc in documents]
    }
    
    # Start streaming
    async for chunk in chain.astream(query):
        if hasattr(chunk, "content"):
            content = chunk.content
        else:
            content = str(chunk)
            
        # Append to collected chunks
        response_chunks.append(content)
        
        # Call the callback if provided
        if callback:
            callback(content)
        
        # Yield the chunk and metadata
        yield {
            "chunk": content,
            "metadata": metadata,
            "complete": False
        }
    
    # Yield the complete response
    complete_response = "".join(response_chunks)
    yield {
        "chunk": complete_response,
        "metadata": metadata,
        "complete": True
    }

class StreamingRAGSystem:
    """
    Streaming RAG system that provides real-time responses
    """
    
    def __init__(self, retriever=None, llm_model="gpt-4o"):
        """
        Initialize the streaming RAG system
        
        Args:
            retriever: The retriever to use
            llm_model: The LLM model to use
        """
        self.retriever = retriever
        
        # Initialize streaming LLM
        self.streaming_llm = ChatOpenAI(
            model=llm_model,
            temperature=0,
            streaming=True
        )
    
    async def stream_response(
        self,
        query: str,
        chat_history: List[Dict[str, str]] = None,
        callback: Optional[Callable[[str], None]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a response to a query
        
        Args:
            query: The user's query
            chat_history: Optional chat history
            callback: Optional callback function to receive chunks
            
        Yields:
            Dictionary containing response chunks and metadata
        """
        # Use provided retriever or import the default one
        if self.retriever is None:
            from langgraph_rag_system import retrieve_context
            documents = retrieve_context(query)
        else:
            documents = self.retriever.get_relevant_documents(query)
        
        # Format context for the prompt
        formatted_context = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in documents
        ])
        
        # Create the prompt
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
        
        # Create the chain
        chain = (
            {"query": RunnablePassthrough(), "context": lambda _: formatted_context}
            | prompt
            | self.streaming_llm
        )
        
        # Stream the response
        response_chunks = []
        metadata = {
            "query": query,
            "num_documents": len(documents),
            "sources": [doc.metadata.get("source", "Unknown") for doc in documents]
        }
        
        # Start streaming
        async for chunk in chain.astream(query):
            if hasattr(chunk, "content"):
                content = chunk.content
            else:
                content = str(chunk)
                
            # Append to collected chunks
            response_chunks.append(content)
            
            # Call the callback if provided
            if callback:
                callback(content)
            
            # Yield the chunk and metadata
            yield {
                "chunk": content,
                "metadata": metadata,
                "complete": False
            }
        
        # Yield the complete response
        complete_response = "".join(response_chunks)
        yield {
            "chunk": complete_response,
            "metadata": metadata,
            "complete": True
        }
    
    async def stream_agentic_response(
        self,
        query: str,
        chat_history: List[Dict[str, str]] = None,
        callback: Optional[Callable[[str], None]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a response using the full agentic workflow
        
        Args:
            query: The user's query
            chat_history: Optional chat history
            callback: Optional callback function to receive chunks
            
        Yields:
            Dictionary containing response chunks, state updates, and metadata
        """
        # First, analyze the query to determine the workflow
        from enhanced_langgraph_rag_system import analyze_query
        
        # Use provided retriever or import the default one
        if self.retriever is None:
            from langgraph_rag_system import retrieve_context
            documents = retrieve_context(query)
        else:
            documents = self.retriever.get_relevant_documents(query)
        
        # Initialize state
        state = {
            "messages": [],
            "query": query,
            "context": documents,
            "urls_to_ingest": [],
            "needs_more_info": False,
            "needs_web_search": False,
            "needs_tool": False,
            "tool_name": None,
            "tool_input": None,
            "tool_result": None,
            "output_format": "text",
            "structured_output": None,
            "final_answer": "",
            "evaluation": None,
            "user_feedback": None
        }
        
        # Analyze the query
        analysis_state = analyze_query(state)
        state.update(analysis_state)
        
        # Yield the analysis state
        yield {
            "chunk": "Analyzing query...",
            "state_update": "analysis",
            "metadata": {
                "needs_more_info": state["needs_more_info"],
                "needs_tool": state["needs_tool"],
                "output_format": state["output_format"]
            },
            "complete": False
        }
        
        # Handle URL ingestion if needed
        if state["needs_more_info"]:
            yield {
                "chunk": f"Ingesting URLs: {', '.join(state['urls_to_ingest'])}...",
                "state_update": "ingestion",
                "metadata": {"urls": state["urls_to_ingest"]},
                "complete": False
            }
            
            from enhanced_langgraph_rag_system import ingest_new_urls
            ingestion_state = ingest_new_urls(state)
            state.update(ingestion_state)
            
            yield {
                "chunk": "Finished ingesting URLs.",
                "state_update": "ingestion_complete",
                "metadata": {},
                "complete": False
            }
        
        # Handle tool use if needed
        if state["needs_tool"]:
            yield {
                "chunk": f"Using tool: {state['tool_name']} with input: {state['tool_input']}...",
                "state_update": "tool_use",
                "metadata": {"tool": state["tool_name"], "input": state["tool_input"]},
                "complete": False
            }
            
            from enhanced_langgraph_rag_system import use_tool
            tool_state = use_tool(state)
            state.update(tool_state)
            
            yield {
                "chunk": f"Tool result: {state['tool_result']}",
                "state_update": "tool_complete",
                "metadata": {"tool_result": state["tool_result"]},
                "complete": False
            }
        
        # Format context for the prompt
        formatted_context = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in state["context"]
        ])
        
        # Create the prompt based on the state
        if state["tool_result"]:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that integrates tool results with context.
                Generate a comprehensive answer that combines information from both the tool result
                and the provided context. Cite sources when appropriate."""),
                ("human", """Query: {query}
                
                Tool Result: {tool_result}
                
                Context:
                {context}
                
                Generate a helpful answer:""")
            ])
            
            # Create the chain
            chain = (
                {
                    "query": lambda _: query,
                    "tool_result": lambda _: state["tool_result"],
                    "context": lambda _: formatted_context
                }
                | prompt
                | self.streaming_llm
            )
        else:
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
            
            # Create the chain
            chain = (
                {"query": lambda _: query, "context": lambda _: formatted_context}
                | prompt
                | self.streaming_llm
            )
        
        # Stream the response
        response_chunks = []
        metadata = {
            "query": query,
            "num_documents": len(state["context"]),
            "sources": [doc.metadata.get("source", "Unknown") for doc in state["context"]]
        }
        
        # Start streaming
        yield {
            "chunk": "Generating response...",
            "state_update": "generation",
            "metadata": metadata,
            "complete": False
        }
        
        async for chunk in chain.astream({}):
            if hasattr(chunk, "content"):
                content = chunk.content
            else:
                content = str(chunk)
                
            # Append to collected chunks
            response_chunks.append(content)
            
            # Call the callback if provided
            if callback:
                callback(content)
            
            # Yield the chunk and metadata
            yield {
                "chunk": content,
                "state_update": "generation_chunk",
                "metadata": metadata,
                "complete": False
            }
        
        # Yield the complete response
        complete_response = "".join(response_chunks)
        state["final_answer"] = complete_response
        
        yield {
            "chunk": complete_response,
            "state_update": "complete",
            "metadata": metadata,
            "state": state,
            "complete": True
        }

# Example usage
async def example_usage():
    # Simple streaming example
    print("Simple streaming example:")
    async for chunk in stream_process_query("What is LangGraph?"):
        if chunk["complete"]:
            print("\nComplete response:", chunk["chunk"])
        else:
            print(chunk["chunk"], end="", flush=True)
    
    # Streaming RAG system example
    print("\n\nStreaming RAG system example:")
    streaming_rag = StreamingRAGSystem()
    
    async for chunk in streaming_rag.stream_response("Explain the concept of retrieval augmented generation"):
        if chunk["complete"]:
            print("\nComplete response:", chunk["chunk"])
        else:
            print(chunk["chunk"], end="", flush=True)
    
    # Agentic streaming example
    print("\n\nAgentic streaming example:")
    
    async for chunk in streaming_rag.stream_agentic_response("What is the weather in New York?"):
        if chunk["state_update"] == "analysis":
            print("Analyzing query...")
        elif chunk["state_update"] == "tool_use":
            print(f"\nUsing tool: {chunk['metadata'].get('tool')}...")
        elif chunk["state_update"] == "tool_complete":
            print(f"\nTool result: {chunk['metadata'].get('tool_result')}")
        elif chunk["state_update"] == "generation":
            print("\nGenerating response:")
        elif chunk["state_update"] == "generation_chunk":
            print(chunk["chunk"], end="", flush=True)
        elif chunk["state_update"] == "complete":
            print("\n\nComplete response generated.")

if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())