"""
Langgraph Agentic RAG System with GPT-4o, text-embedding-3-large, FAISS DB and URL ingestion
This system implements:
1. URL content ingestion using LangChain's WebBaseLoader
2. Text embedding using OpenAI's text-embedding-3-large
3. Vector storage using FAISS
4. RAG pipeline with GPT-4o
5. Agentic workflow using LangGraph
"""

import os
import json
from typing import Dict, List, Any, TypedDict, Annotated, Sequence, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# Define state types for the graph
class AgentState(TypedDict):
    """State for the agent."""
    messages: Annotated[Sequence[Any], "Messages in the conversation so far"]
    query: Annotated[str, "User's original query"]
    context: Annotated[List[Document], "Retrieved documents from vector store"]
    urls_to_ingest: Annotated[List[str], "URLs to ingest into the knowledge base"]
    needs_more_info: Annotated[bool, "Whether the agent needs more information"]
    needs_web_search: Annotated[bool, "Whether the agent needs to search the web"]
    final_answer: Annotated[str, "Final answer to return to the user"]

# Initialize OpenAI models
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Initialize vector store
# This will be populated with documents as they are ingested
vector_store = None

def initialize_vector_store():
    """Initialize the vector store if it doesn't exist."""
    global vector_store
    try:
        # Try to load from disk if it exists
        vector_store = FAISS.load_local("faiss_index", embeddings)
        print("Loaded existing vector store from disk.")
    except:
        # Create a new one if it doesn't exist
        vector_store = FAISS.from_documents(
            [Document(page_content="Initial document to create the vector store.")], 
            embeddings
        )
        vector_store.save_local("faiss_index")
        print("Created new vector store.")

# Initialize the vector store
initialize_vector_store()

def ingest_urls(urls: List[str]) -> str:
    """
    Ingest content from URLs into the vector store.
    
    Args:
        urls: List of URLs to ingest
        
    Returns:
        A message indicating the ingestion results
    """
    global vector_store
    
    if not urls:
        return "No URLs provided for ingestion."
    
    results = []
    
    for url in urls:
        try:
            # Load content from URL
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Add document metadata
            for doc in splits:
                doc.metadata["source"] = url
            
            # Add to vector store
            vector_store.add_documents(splits)
            
            # Save updated vector store
            vector_store.save_local("faiss_index")
            
            results.append(f"Successfully ingested {len(splits)} chunks from {url}")
        except Exception as e:
            results.append(f"Failed to ingest {url}: {str(e)}")
    
    return "\n".join(results)

def retrieve_context(query: str, k: int = 5) -> List[Document]:
    """
    Retrieve relevant documents from the vector store.
    
    Args:
        query: The query to search for
        k: Number of documents to retrieve
        
    Returns:
        List of retrieved documents
    """
    global vector_store
    
    if not vector_store:
        initialize_vector_store()
        
    retrieved_docs = vector_store.similarity_search(query, k=k)
    return retrieved_docs

# Define the RAG prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with access to a knowledge base.
    Answer the user's question based on the following context. If the context doesn't contain
    relevant information, say so and suggest that more information might need to be ingested.
    
    If you need to search for more information or ingest new URLs, indicate this in your response.
    """),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{query}"),
    ("system", "Context from knowledge base:\n{context}")
])

# Define the agent prompt
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent agent that can analyze user queries and determine the best course of action.
    
    You have the following capabilities:
    1. Answer questions using your existing knowledge
    2. Retrieve information from a knowledge base
    3. Ingest new URLs to expand the knowledge base
    
    Based on the user's query, determine if:
    - You need to ingest new URLs (set needs_more_info=True and provide urls_to_ingest)
    - You can answer with existing knowledge and context
    
    Format your response as a JSON object with the following fields:
    {
        "needs_more_info": boolean,
        "urls_to_ingest": [list of URLs] or [],
        "needs_web_search": boolean,
        "final_answer": "Your answer to the user's query if you have enough information"
    }
    """),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{query}"),
])

# Define the nodes for the graph

def retrieve(state: AgentState) -> AgentState:
    """Retrieve relevant documents from the vector store."""
    query = state["query"]
    context = retrieve_context(query)
    return {"context": context}

def analyze_query(state: AgentState) -> AgentState:
    """Analyze the query and determine next steps."""
    # Format context for the prompt
    formatted_context = "\n\n".join([doc.page_content for doc in state["context"]])
    
    # Run the agent prompt
    response = agent_prompt.invoke({
        "messages": state["messages"],
        "query": state["query"],
        "context": formatted_context
    })
    
    agent_response = llm.invoke(response)
    
    # Parse the JSON response
    try:
        parsed_response = json.loads(agent_response.content)
        return {
            "needs_more_info": parsed_response.get("needs_more_info", False),
            "urls_to_ingest": parsed_response.get("urls_to_ingest", []),
            "needs_web_search": parsed_response.get("needs_web_search", False),
            "final_answer": parsed_response.get("final_answer", "")
        }
    except:
        # If JSON parsing fails, assume we can answer with what we have
        return {
            "needs_more_info": False,
            "urls_to_ingest": [],
            "needs_web_search": False,
            "final_answer": agent_response.content
        }

def ingest_new_urls(state: AgentState) -> AgentState:
    """Ingest new URLs into the knowledge base."""
    urls = state["urls_to_ingest"]
    ingestion_result = ingest_urls(urls)
    
    # After ingestion, retrieve updated context
    updated_context = retrieve_context(state["query"])
    
    # Add the ingestion result to messages
    new_messages = list(state["messages"])
    new_messages.append(AIMessage(content=f"I've ingested the following URLs:\n{ingestion_result}"))
    
    return {
        "context": updated_context,
        "messages": new_messages,
        "needs_more_info": False  # Reset this flag
    }

def generate_answer(state: AgentState) -> AgentState:
    """Generate the final answer using RAG."""
    # Format context for the prompt
    formatted_context = "\n\n".join([
        f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
        for doc in state["context"]
    ])
    
    # Run the RAG prompt
    response = rag_prompt.invoke({
        "messages": state["messages"],
        "query": state["query"],
        "context": formatted_context
    })
    
    answer = llm.invoke(response)
    
    # Add the answer to messages
    new_messages = list(state["messages"])
    new_messages.append(AIMessage(content=answer.content))
    
    return {
        "messages": new_messages,
        "final_answer": answer.content
    }

# Define the state graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("analyze_query", analyze_query)
workflow.add_node("ingest_new_urls", ingest_new_urls)
workflow.add_node("generate_answer", generate_answer)

# Add edges
workflow.add_edge("retrieve", "analyze_query")
workflow.add_conditional_edges(
    "analyze_query",
    lambda state: "ingest_new_urls" if state["needs_more_info"] else "generate_answer"
)
workflow.add_edge("ingest_new_urls", "generate_answer")
workflow.add_edge("generate_answer", END)

# Set the entry point
workflow.set_entry_point("retrieve")

# Compile the graph
app = workflow.compile()

def process_query(query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Process a user query through the agentic RAG system.
    
    Args:
        query: The user's query
        chat_history: Optional chat history
        
    Returns:
        The final state of the workflow
    """
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
    messages.append(HumanMessage(content=query))
    
    # Initialize the state
    initial_state = {
        "messages": messages,
        "query": query,
        "context": [],
        "urls_to_ingest": [],
        "needs_more_info": False,
        "needs_web_search": False,
        "final_answer": ""
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    return result

# Example usage
if __name__ == "__main__":
    # Example query
    query = "What are the key features of LangGraph?"
    
    # Process the query
    result = process_query(query)
    
    # Print the final answer
    print("Final Answer:", result["final_answer"])
    
    # Example of ingesting a URL and then querying
    ingest_query = "Please ingest information about LangGraph from https://python.langchain.com/docs/langgraph/"
    ingest_result = process_query(ingest_query)
    
    # Follow-up query after ingestion
    followup_query = "Now explain the key components of LangGraph based on what you've ingested"
    chat_history = [
        {"role": "user", "content": ingest_query},
        {"role": "assistant", "content": ingest_result["final_answer"]}
    ]
    followup_result = process_query(followup_query, chat_history)
    
    print("\nFollow-up Answer:", followup_result["final_answer"])