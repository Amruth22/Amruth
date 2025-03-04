"""
Example script demonstrating streaming responses with the langgraph RAG system
"""

import asyncio
import time
from typing import Dict, Any

from streaming_responses import StreamingRAGSystem, stream_process_query

# Simple callback function to print chunks as they arrive
def print_callback(chunk: str):
    print(chunk, end="", flush=True)

async def demo_simple_streaming():
    """
    Demonstrate simple streaming responses
    """
    print("\n" + "="*50)
    print("SIMPLE STREAMING DEMO")
    print("="*50)
    
    print("\nQuerying: 'What is retrieval augmented generation?'")
    print("\nResponse: ", end="")
    
    start_time = time.time()
    
    # Stream the response
    complete_response = ""
    async for chunk in stream_process_query("What is retrieval augmented generation?"):
        if not chunk["complete"]:
            print(chunk["chunk"], end="", flush=True)
            complete_response += chunk["chunk"]
    
    end_time = time.time()
    
    print(f"\n\nTime taken: {end_time - start_time:.2f} seconds")
    print(f"Response length: {len(complete_response)} characters")

async def demo_streaming_rag():
    """
    Demonstrate streaming RAG system
    """
    print("\n" + "="*50)
    print("STREAMING RAG SYSTEM DEMO")
    print("="*50)
    
    # Initialize the streaming RAG system
    streaming_rag = StreamingRAGSystem()
    
    # Example queries
    queries = [
        "What are the key components of a RAG system?",
        "How does LangGraph work with agents?"
    ]
    
    for query in queries:
        print(f"\nQuerying: '{query}'")
        print("\nResponse: ", end="")
        
        start_time = time.time()
        
        # Stream the response
        complete_response = ""
        async for chunk in streaming_rag.stream_response(query):
            if not chunk["complete"]:
                print(chunk["chunk"], end="", flush=True)
                complete_response += chunk["chunk"]
        
        end_time = time.time()
        
        print(f"\n\nTime taken: {end_time - start_time:.2f} seconds")
        print(f"Response length: {len(complete_response)} characters")
        print("-"*50)

async def demo_agentic_streaming():
    """
    Demonstrate agentic streaming responses
    """
    print("\n" + "="*50)
    print("AGENTIC STREAMING DEMO")
    print("="*50)
    
    # Initialize the streaming RAG system
    streaming_rag = StreamingRAGSystem()
    
    # Example queries that might trigger different agent behaviors
    queries = [
        "What is the weather in San Francisco?",  # Might trigger tool use
        "Please ingest information about LangGraph from https://python.langchain.com/docs/langgraph/",  # URL ingestion
        "Compare Python and JavaScript programming languages"  # Might trigger comparison format
    ]
    
    for query in queries:
        print(f"\nQuerying: '{query}'")
        
        start_time = time.time()
        
        # Stream the response with state updates
        complete_response = ""
        async for chunk in streaming_rag.stream_agentic_response(query):
            if chunk["state_update"] == "analysis":
                print("\nAnalyzing query...")
            elif chunk["state_update"] == "ingestion":
                print(f"\nIngesting URLs: {', '.join(chunk['metadata'].get('urls', []))}...")
            elif chunk["state_update"] == "ingestion_complete":
                print("\nFinished ingesting URLs.")
            elif chunk["state_update"] == "tool_use":
                print(f"\nUsing tool: {chunk['metadata'].get('tool', 'unknown')}...")
            elif chunk["state_update"] == "tool_complete":
                print(f"\nTool result received.")
            elif chunk["state_update"] == "generation":
                print("\nGenerating response: ", end="")
            elif chunk["state_update"] == "generation_chunk":
                print(chunk["chunk"], end="", flush=True)
                complete_response += chunk["chunk"]
            elif chunk["state_update"] == "complete":
                print("\n\nResponse complete.")
        
        end_time = time.time()
        
        print(f"\nTime taken: {end_time - start_time:.2f} seconds")
        print(f"Response length: {len(complete_response)} characters")
        print("-"*50)

async def interactive_demo():
    """
    Interactive demo allowing user to try streaming responses
    """
    print("\n" + "="*50)
    print("INTERACTIVE STREAMING DEMO")
    print("="*50)
    print("\nType your questions and see streaming responses.")
    print("Type 'exit' to quit.")
    
    # Initialize the streaming RAG system
    streaming_rag = StreamingRAGSystem()
    
    while True:
        query = input("\nYour question: ")
        
        if query.lower() == 'exit':
            break
        
        print("\nResponse: ", end="")
        
        start_time = time.time()
        
        # Stream the response
        complete_response = ""
        async for chunk in streaming_rag.stream_response(query):
            if not chunk["complete"]:
                print(chunk["chunk"], end="", flush=True)
                complete_response += chunk["chunk"]
        
        end_time = time.time()
        
        print(f"\n\nTime taken: {end_time - start_time:.2f} seconds")

async def main():
    """
    Main function to run all demos
    """
    print("\nSTREAMING RESPONSES DEMO")
    print("Choose a demo to run:")
    print("1. Simple Streaming")
    print("2. Streaming RAG System")
    print("3. Agentic Streaming")
    print("4. Interactive Demo")
    print("5. Run All Demos")
    
    choice = input("\nEnter your choice (1-5): ")
    
    if choice == '1':
        await demo_simple_streaming()
    elif choice == '2':
        await demo_streaming_rag()
    elif choice == '3':
        await demo_agentic_streaming()
    elif choice == '4':
        await interactive_demo()
    elif choice == '5':
        await demo_simple_streaming()
        await demo_streaming_rag()
        await demo_agentic_streaming()
        await interactive_demo()
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())