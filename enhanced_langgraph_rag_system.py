from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Import multilingual support
from multilingual_support import detect_language, translate_with_gpt

# Initialize OpenAI models
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Initialize vector store
vector_store = None
vector_store_path = "faiss_index"

def initialize_vector_store():
    """Initialize the vector store if it doesn't exist."""
    global vector_store
    try:
        # Try to load from disk if it exists
        vector_store = FAISS.load_local(vector_store_path, embeddings)
        print("Loaded existing vector store from disk.")
    except:
        # Create a new one if it doesn't exist
        vector_store = FAISS.from_documents(
            [Document(page_content="Initial document to create the vector store.")], 
            embeddings
        )
        vector_store.save_local(vector_store_path)
        print("Created new vector store.")

# Initialize the vector store
initialize_vector_store()

# Define the state graph
workflow = StateGraph(EnhancedAgentState)

# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("analyze_query", analyze_query)
workflow.add_node("ingest_new_urls", ingest_new_urls)
workflow.add_node("use_tool", use_tool)
workflow.add_node("generate_answer", generate_answer)

# Add edges
workflow.add_edge("retrieve", "analyze_query")

# Add conditional edges from analyze_query
workflow.add_conditional_edges(
    "analyze_query",
    lambda state: (
        "ingest_new_urls" if state["needs_more_info"] else
        "use_tool" if state["needs_tool"] else
        "generate_answer"
    )
)

workflow.add_edge("ingest_new_urls", "generate_answer")
workflow.add_edge("use_tool", "generate_answer")
workflow.add_edge("generate_answer", END)

# Set the entry point
workflow.set_entry_point("retrieve")

# Compile the graph
app = workflow.compile()

def process_query(
    query: str, 
    chat_history: List[Dict[str, str]] = None,
    collect_user_feedback: bool = False,
    user_rating: Optional[int] = None,
    user_feedback_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a user query through the enhanced agentic RAG system.
    
    Args:
        query: The user's query
        chat_history: Optional chat history
        collect_user_feedback: Whether to collect user feedback
        user_rating: User rating (1-5) if collecting feedback
        user_feedback_text: Optional user feedback text
        
    Returns:
        The final state of the workflow
    """
    # Detect the language of the query
    detected_language = detect_language(query)
    
    # Translate the query to English if necessary
    translated_query = translate_with_gpt(query, detected_language, 'en')
    
    # Process the translated query
    if chat_history is None:
        chat_history = []
    
    # Convert chat history to LangChain message format
    messages = []
    for message in chat_history:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        else:
            messages.append(AIMessage(content=message["content"]))
    
    # Add the current query
    messages.append(HumanMessage(content=translated_query))
    
    # Initialize the state
    initial_state = {
        "messages": messages,
        "query": translated_query,
        "context": [],
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
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Translate the final answer back to the original language
    result["final_answer"] = translate_with_gpt(result["final_answer"], 'en', detected_language)
    
    # Collect feedback if requested
    if collect_user_feedback and user_rating is not None:
        feedback_state = collect_feedback(result, user_rating, user_feedback_text)
        result["user_feedback"] = feedback_state["user_feedback"]
    
    return result

# Example usage
if __name__ == "__main__":
    # Example query
    query = "What are the key features of LangGraph?"
    
    # Process the query
    result = process_query(query)
    
    # Print the final answer
    print("Final Answer:", result["final_answer"])