import streamlit as st
import asyncio
from enhanced_langgraph_rag_system import process_query
from streaming_responses import StreamingRAGSystem

# Initialize the streaming RAG system
streaming_rag = StreamingRAGSystem()

# Streamlit app title
st.title("Enhanced Langgraph Agentic RAG System")

# Sidebar for user input
st.sidebar.header("User Input")
query = st.sidebar.text_area("Enter your query:", "What are the key features of LangGraph?")

# Option to choose response type
response_type = st.sidebar.selectbox(
    "Choose response type:",
    ("Standard Response", "Streaming Response")
)

# Option to choose language
language = st.sidebar.selectbox(
    "Choose language:",
    ("English", "Spanish", "French", "German", "Chinese")
)

# Button to submit query
if st.sidebar.button("Submit"):
    # Detect language and translate query if necessary
    if language != "English":
        query = translate_with_gpt(query, detect_language(query), 'en')

    if response_type == "Standard Response":
        # Process the query using the standard RAG system
        result = process_query(query)
        
        # Translate the response back to the selected language
        if language != "English":
            result["final_answer"] = translate_with_gpt(result["final_answer"], 'en', detect_language(query))
        
        # Display the result
        st.subheader("Response:")
        st.write(result["final_answer"])
    else:
        # Streaming response
        st.subheader("Streaming Response:")
        response_container = st.empty()
        complete_response = ""

        async def stream_response():
            async for chunk in streaming_rag.stream_response(query):
                if not chunk["complete"]:
                    complete_response += chunk["chunk"]
                    response_container.text(complete_response)

        asyncio.run(stream_response())

        # Translate the complete response back to the selected language
        if language != "English":
            complete_response = translate_with_gpt(complete_response, 'en', detect_language(query))
            response_container.text(complete_response)

# Instructions
st.sidebar.subheader("Instructions:")
st.sidebar.write("1. Enter your query in the text area.")
st.sidebar.write("2. Choose the response type: Standard or Streaming.")
st.sidebar.write("3. Select the language for the query and response.")
st.sidebar.write("4. Click 'Submit' to see the response.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed using LangGraph, GPT-4o, and Streamlit.")