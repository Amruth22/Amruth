"""
Enhanced Langgraph Agentic RAG System with GPT-4o, text-embedding-3-large, FAISS DB and URL ingestion
This system incorporates all improvements:
1. Advanced retrieval techniques
2. Multi-modal content processing
3. Evaluation and feedback loop
4. Structured output and tool use
5. Integrated workflow with LangGraph
"""

import os
import json
from typing import Dict, List, Any, TypedDict, Annotated, Sequence, Tuple, Optional, Union
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Import improvement modules
from improvements.advanced_retrieval import get_advanced_retriever
from improvements.multimodal_rag import ingest_multimodal_content
from improvements.evaluation import RAGEvaluator, FeedbackLoop
from improvements.structured_output import StructuredOutputGenerator, ToolUseManager

# Load environment variables
load_dotenv()

# Define state types for the graph
class EnhancedAgentState(TypedDict):
    """State for the enhanced agent."""
    messages: Annotated[Sequence[Any], "Messages in the conversation so far"]
    query: Annotated[str, "User's original query"]
    context: Annotated[List[Document], "Retrieved documents from vector store"]
    urls_to_ingest: Annotated[List[str], "URLs to ingest into the knowledge base"]
    needs_more_info: Annotated[bool, "Whether the agent needs more information"]
    needs_web_search: Annotated[bool, "Whether the agent needs to search the web"]
    needs_tool: Annotated[bool, "Whether the agent needs to use a tool"]
    tool_name: Annotated[Optional[str], "Name of the tool to use"]
    tool_input: Annotated[Optional[str], "Input for the tool"]
    tool_result: Annotated[Optional[str], "Result from the tool"]
    output_format: Annotated[str, "Format of the output (text, search_result, analysis, comparison)"]
    structured_output: Annotated[Optional[Dict[str, Any]], "Structured output data"]
    final_answer: Annotated[str, "Final answer to return to the user"]
    evaluation: Annotated[Optional[Dict[str, Any]], "Evaluation of the response"]
    user_feedback: Annotated[Optional[Dict[str, Any]], "User feedback on the response"]

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

# Initialize components
evaluator = RAGEvaluator()
feedback_loop = FeedbackLoop(vector_store)
structured_output_generator = StructuredOutputGenerator()
tool_manager = ToolUseManager()

# Initialize advanced retriever
retriever = get_advanced_retriever(vector_store, "hybrid")

# Define the nodes for the graph

def retrieve(state: EnhancedAgentState) -> EnhancedAgentState:
    """Retrieve relevant documents from the vector store."""
    query = state["query"]
    context = retriever.get_relevant_documents(query)
    return {"context": context}

def analyze_query(state: EnhancedAgentState) -> EnhancedAgentState:
    """Analyze the query and determine next steps."""
    query = state["query"]
    context = state["context"]
    
    # Format context for the prompt
    formatted_context = "\n\n".join([
        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in context
    ])
    
    # Create analysis prompt
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent agent that analyzes user queries and determines the best course of action.
        
        You have the following capabilities:
        1. Answer questions using your existing knowledge and retrieved context
        2. Ingest new URLs to expand the knowledge base
        3. Use tools to get additional information
        4. Generate structured outputs in different formats
        
        Based on the user's query and the retrieved context, determine:
        - If you need to ingest new URLs (set needs_more_info=True and provide urls_to_ingest)
        - If you need to use a tool (set needs_tool=True and provide tool_name and tool_input)
        - What output format would be most appropriate for this query
        
        Available tools:
        - search_web: Search the web for information
        - get_weather: Get current weather information for a location
        - calculate: Calculate the result of a mathematical expression
        
        Available output formats:
        - text: Standard text response
        - search_result: Structured search result with sources and confidence
        - analysis: Structured analysis with key points and entities
        - comparison: Structured comparison between items
        
        Format your response as a JSON object with the following fields:
        {
            "needs_more_info": boolean,
            "urls_to_ingest": [list of URLs] or [],
            "needs_tool": boolean,
            "tool_name": "name of the tool or null",
            "tool_input": "input for the tool or null",
            "output_format": "text/search_result/analysis/comparison"
        }
        """),
        ("human", """Query: {query}
        
        Retrieved Context:
        {context}
        
        Analyze this query and determine the next steps:""")
    ])
    
    # Run the analysis
    response = llm.invoke(
        analysis_prompt.format_messages(
            query=query,
            context=formatted_context
        )
    )
    
    # Parse the JSON response
    try:
        parsed_response = json.loads(response.content)
        return {
            "needs_more_info": parsed_response.get("needs_more_info", False),
            "urls_to_ingest": parsed_response.get("urls_to_ingest", []),
            "needs_tool": parsed_response.get("needs_tool", False),
            "tool_name": parsed_response.get("tool_name"),
            "tool_input": parsed_response.get("tool_input"),
            "output_format": parsed_response.get("output_format", "text")
        }
    except:
        # If JSON parsing fails, use default values
        return {
            "needs_more_info": False,
            "urls_to_ingest": [],
            "needs_tool": False,
            "tool_name": None,
            "tool_input": None,
            "output_format": "text"
        }

def ingest_new_urls(state: EnhancedAgentState) -> EnhancedAgentState:
    """Ingest new URLs into the knowledge base."""
    urls = state["urls_to_ingest"]
    results = []
    
    for url in urls:
        try:
            # Use multimodal ingestion
            documents = ingest_multimodal_content(url)
            
            # Add to vector store
            vector_store.add_documents(documents)
            
            # Save updated vector store
            vector_store.save_local(vector_store_path)
            
            results.append(f"Successfully ingested {len(documents)} documents from {url}")
        except Exception as e:
            results.append(f"Failed to ingest {url}: {str(e)}")
    
    # After ingestion, retrieve updated context
    updated_context = retriever.get_relevant_documents(state["query"])
    
    # Add the ingestion result to messages
    new_messages = list(state["messages"])
    new_messages.append(AIMessage(content=f"I've ingested the following URLs:\n" + "\n".join(results)))
    
    return {
        "context": updated_context,
        "messages": new_messages,
        "needs_more_info": False  # Reset this flag
    }

def use_tool(state: EnhancedAgentState) -> EnhancedAgentState:
    """Use a tool to get additional information."""
    tool_name = state["tool_name"]
    tool_input = state["tool_input"]
    
    if not tool_name or not tool_input:
        return {
            "tool_result": "Error: Tool name or input not provided",
            "needs_tool": False  # Reset this flag
        }
    
    # Execute the tool
    tool_result = tool_manager.execute_tool(tool_name, tool_input)
    
    # Add the tool result to messages
    new_messages = list(state["messages"])
    new_messages.append(AIMessage(content=f"I used the {tool_name} tool with input '{tool_input}' and got the following result:\n{tool_result}"))
    
    return {
        "tool_result": tool_result,
        "messages": new_messages,
        "needs_tool": False  # Reset this flag
    }

def generate_answer(state: EnhancedAgentState) -> EnhancedAgentState:
    """Generate the final answer using the appropriate format."""
    query = state["query"]
    context = state["context"]
    output_format = state["output_format"]
    tool_result = state.get("tool_result")
    
    # Extract content and metadata
    context_content = [doc.page_content for doc in context]
    context_metadata = [doc.metadata for doc in context]
    
    # Generate response based on output format
    response = None
    structured_output = None
    
    if tool_result:
        # Integrate tool result with RAG context
        response = tool_manager.integrate_tool_result(
            query=query,
            tool_name=state.get("tool_name", "unknown_tool"),
            tool_input=state.get("tool_input", ""),
            tool_result=tool_result,
            context=context_content
        )
    
    elif output_format == "text":
        # Generate text response
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
        llm_response = llm.invoke(
            prompt.format_messages(
                query=query,
                context=formatted_context
            )
        )
        
        response = llm_response.content
    
    elif output_format == "search_result":
        # Generate structured search result
        result = structured_output_generator.generate_search_result(
            query=query,
            context=context_content,
            metadata=context_metadata
        )
        
        response = result.answer
        structured_output = result.model_dump()
    
    elif output_format == "analysis":
        # Generate structured analysis result
        result = structured_output_generator.generate_analysis_result(
            query=query,
            context=context_content,
            metadata=context_metadata
        )
        
        response = result.summary
        structured_output = result.model_dump()
    
    elif output_format == "comparison":
        # Extract items to compare from query
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
        result = structured_output_generator.generate_comparison_result(
            query=query,
            items_to_compare=items_to_compare,
            context=context_content,
            metadata=context_metadata
        )
        
        response = f"Comparison: {result.recommendation}"
        structured_output = result.model_dump()
    
    # Perform evaluation
    evaluation = evaluator.evaluate_response(
        query=query,
        response=response,
        retrieved_documents=context
    )
    
    # Add the answer to messages
    new_messages = list(state["messages"])
    new_messages.append(AIMessage(content=response))
    
    return {
        "messages": new_messages,
        "final_answer": response,
        "structured_output": structured_output,
        "evaluation": evaluation
    }

def collect_feedback(state: EnhancedAgentState, user_rating: int, user_feedback_text: Optional[str] = None) -> EnhancedAgentState:
    """Collect user feedback on the response."""
    query = state["query"]
    final_answer = state["final_answer"]
    context = state["context"]
    
    feedback_result = feedback_loop.process_user_feedback(
        query=query,
        response=final_answer,
        retrieved_documents=context,
        user_rating=user_rating,
        user_feedback=user_feedback_text
    )
    
    return {
        "user_feedback": feedback_result
    }

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
    
    # Example with tool use
    tool_query = "What is the weather in New York?"
    tool_result = process_query(tool_query)
    
    print("\nTool-based Answer:", tool_result["final_answer"])
    
    # Example with structured output
    analysis_query = "Analyze the key trends in artificial intelligence for 2023"
    analysis_result = process_query(analysis_query)
    
    print("\nAnalysis Answer:", analysis_result["final_answer"])
    
    # Example with user feedback
    feedback_result = process_query(
        "Explain quantum computing",
        collect_user_feedback=True,
        user_rating=4,
        user_feedback_text="Good explanation but could use more examples"
    )
    
    print("\nFeedback processing:", feedback_result["user_feedback"])