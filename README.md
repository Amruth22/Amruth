# Langgraph Agentic RAG System

This repository contains an implementation of an advanced Retrieval-Augmented Generation (RAG) system using LangGraph, OpenAI's GPT-4o, text-embedding-3-large, and FAISS vector database.

## Features

- **URL Content Ingestion**: Automatically extract and process content from web URLs using LangChain's WebBaseLoader
- **Advanced Embeddings**: Uses OpenAI's text-embedding-3-large for high-quality vector representations
- **Vector Storage**: FAISS vector database for efficient similarity search
- **LLM Integration**: Powered by OpenAI's GPT-4o for high-quality responses
- **Agentic Workflow**: Uses LangGraph to create a flexible, state-based workflow
- **Conversation Memory**: Maintains chat history for contextual responses

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
from langgraph_rag_system import process_query

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

## How It Works

1. **Query Analysis**: The system analyzes the user's query to determine if it needs to ingest new information
2. **Context Retrieval**: Relevant documents are retrieved from the FAISS vector store
3. **URL Ingestion**: If needed, content from URLs is extracted, chunked, and stored in the vector database
4. **Response Generation**: GPT-4o generates a response based on the retrieved context and conversation history

## System Architecture

The system uses a LangGraph workflow with the following nodes:
- `retrieve`: Fetches relevant documents from the vector store
- `analyze_query`: Determines if more information is needed
- `ingest_new_urls`: Processes and stores content from URLs
- `generate_answer`: Creates the final response using RAG

## Customization

You can customize various aspects of the system:
- Change the embedding model or LLM in the initialization section
- Adjust the chunk size and overlap in the `ingest_urls` function
- Modify the prompts in `rag_prompt` and `agent_prompt`
- Add additional nodes to the workflow for more complex behaviors