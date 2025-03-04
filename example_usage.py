"""
Example usage of the Langgraph Agentic RAG system
"""

from langgraph_rag_system import process_query

def main():
    print("Langgraph Agentic RAG System Example")
    print("====================================")
    
    # Example 1: Basic query
    print("\nExample 1: Basic query")
    print("----------------------")
    query1 = "What are the key features of LangGraph?"
    print(f"Query: {query1}")
    result1 = process_query(query1)
    print(f"Answer: {result1['final_answer']}")
    
    # Example 2: Ingesting a URL
    print("\nExample 2: Ingesting a URL")
    print("-------------------------")
    query2 = "Please ingest information about LangGraph from https://python.langchain.com/docs/langgraph/"
    print(f"Query: {query2}")
    result2 = process_query(query2)
    print(f"Answer: {result2['final_answer']}")
    
    # Example 3: Follow-up query after ingestion
    print("\nExample 3: Follow-up query after ingestion")
    print("----------------------------------------")
    query3 = "Now explain the key components of LangGraph based on what you've ingested"
    print(f"Query: {query3}")
    
    # Create chat history from previous interaction
    chat_history = [
        {"role": "user", "content": query2},
        {"role": "assistant", "content": result2["final_answer"]}
    ]
    
    result3 = process_query(query3, chat_history)
    print(f"Answer: {result3['final_answer']}")
    
    # Example 4: Interactive mode
    print("\nExample 4: Interactive mode")
    print("-------------------------")
    print("Type 'exit' to quit")
    
    interactive_chat_history = []
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'exit':
            break
            
        result = process_query(user_input, interactive_chat_history)
        print(f"\nAnswer: {result['final_answer']}")
        
        # Update chat history
        interactive_chat_history.append({"role": "user", "content": user_input})
        interactive_chat_history.append({"role": "assistant", "content": result["final_answer"]})

if __name__ == "__main__":
    main()