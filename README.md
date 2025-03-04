# Enhanced Langgraph Agentic RAG System

This repository contains an implementation of an advanced Retrieval-Augmented Generation (RAG) system using LangGraph, OpenAI's GPT-4o, text-embedding-3-large, and FAISS vector database.

## Features

### Core Functionality
- **URL Content Ingestion**: Automatically extract and process content from web URLs using LangChain's WebBaseLoader
- **Advanced Embeddings**: Uses OpenAI's text-embedding-3-large for high-quality vector representations
- **Vector Storage**: FAISS vector database for efficient similarity search
- **LLM Integration**: Powered by OpenAI's GPT-4o for high-quality responses
- **Agentic Workflow**: Uses LangGraph to create a flexible, state-based workflow
- **Conversation Memory**: Maintains chat history for contextual responses

### Advanced Features

#### 1. Advanced Retrieval Techniques
- **Hybrid Search**: Combines keyword-based (BM25) and semantic search for better results
- **Contextual Compression**: Extracts only the most relevant parts of documents
- **Query Transformation**: Rewrites queries to improve retrieval effectiveness
- **Self-Query Retrieval**: Filters documents based on metadata
- **Parent Document Retrieval**: Returns parent documents after searching through child chunks
- **Multi-Query Retrieval**: Generates multiple queries from a single user question

#### 2. Multi-Modal Capabilities
- **PDF Processing**: Extract and process text from PDF documents
- **Image Analysis**: Uses GPT-4o Vision to analyze and describe images
- **YouTube Transcripts**: Extract and process transcripts from YouTube videos
- **Automatic Content Type Detection**: Identifies and processes different content types

#### 3. Evaluation and Feedback Loop
- **Response Evaluation**: Automatically evaluates responses on relevance, faithfulness, and more
- **User Feedback Collection**: Collects and processes user feedback
- **Performance Analysis**: Analyzes evaluation logs to identify improvement areas
- **Feedback-Driven Improvements**: Updates retrieval strategy based on feedback

#### 4. Structured Output and Tool Use
- **Structured Search Results**: Generates structured search results with sources and confidence
- **Analysis Output**: Provides structured analysis with key points and entities
- **Comparison Output**: Creates structured comparisons between items
- **Tool Integration**: Uses external tools for weather, calculations, and web search
- **Tool Detection**: Automatically detects when tools are needed

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Basic Query

```python
from enhanced_langgraph_rag_system import process_query

# Ask a question
result = process_query("What are the key features of LangGraph?")
print(result["final_answer"])
```

### Ingesting New URLs

The system can ingest content from URLs to expand its knowledge base:

```python
# Ingest a URL
ingest_result = process_query("Please ingest information about LangGraph from https://python.langchain.com/docs/langgraph/")

# Ask a follow-up question about the ingested content
chat_history = [
    {"role": "user", "content": "Please ingest information about LangGraph from https://python.langchain.com/docs/langgraph/"},
    {"role": "assistant", "content": ingest_result["final_answer"]}
]
followup_result = process_query("Now explain the key components of LangGraph", chat_history)
print(followup_result["final_answer"])
```

### Using Tools

The system can automatically detect when tools are needed:

```python
# Query that requires a tool
tool_result = process_query("What is the weather in New York?")
print(tool_result["final_answer"])
```

### Structured Output

You can get structured output in different formats:

```python
# Get a structured analysis
analysis_result = process_query("Analyze the key trends in artificial intelligence for 2023")
print(analysis_result["structured_output"])
```

### Collecting User Feedback

```python
# Process a query and collect feedback
feedback_result = process_query(
    "Explain quantum computing",
    collect_user_feedback=True,
    user_rating=4,
    user_feedback_text="Good explanation but could use more examples"
)
```

## System Architecture

The system uses a LangGraph workflow with the following nodes:
- `retrieve`: Fetches relevant documents from the vector store
- `analyze_query`: Determines if more information is needed or if tools should be used
- `ingest_new_urls`: Processes and stores content from URLs
- `use_tool`: Executes external tools when needed
- `generate_answer`: Creates the final response using RAG

## Improvement Modules

The system is organized into several improvement modules:

1. **advanced_retrieval.py**: Advanced retrieval techniques
2. **multimodal_rag.py**: Multi-modal content processing
3. **evaluation.py**: Evaluation and feedback loop
4. **structured_output.py**: Structured output and tool use
5. **integration.py**: Integration of all improvements

## Customization

You can customize various aspects of the system:
- Change the embedding model or LLM in the initialization section
- Adjust the chunk size and overlap in the ingestion functions
- Modify the prompts for different components
- Add additional nodes to the workflow for more complex behaviors
- Implement new tools in the ToolUseManager

## Example Scripts

- **example_usage.py**: Basic usage examples
- **enhanced_langgraph_rag_system.py**: The main system with all improvements

## Future Improvements

Potential areas for further enhancement:
- Streaming responses for better user experience
- Fine-tuning models on domain-specific data
- Adding more advanced tools and integrations
- Implementing a web interface for easier interaction
- Adding support for more content types and sources