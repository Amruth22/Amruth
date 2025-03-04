"""
Multi-modal RAG capabilities to handle images, PDFs, and other document types
"""

import os
import tempfile
import requests
from typing import List, Dict, Any, Union, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, ImageLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from io import BytesIO

# Initialize models
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def download_file(url: str) -> str:
    """
    Download a file from a URL and save it to a temporary file
    
    Args:
        url: URL of the file to download
        
    Returns:
        Path to the downloaded file
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Determine file extension from content type or URL
    content_type = response.headers.get('content-type', '')
    if 'pdf' in content_type or url.lower().endswith('.pdf'):
        ext = '.pdf'
    elif any(img_type in content_type for img_type in ['image/jpeg', 'image/png', 'image/jpg']):
        if 'jpeg' in content_type or 'jpg' in url.lower():
            ext = '.jpg'
        else:
            ext = '.png'
    else:
        # Default to binary file
        ext = '.bin'
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    
    # Write the content to the file
    for chunk in response.iter_content(chunk_size=8192):
        temp_file.write(chunk)
    
    temp_file.close()
    return temp_file.name

def process_pdf(file_path: str) -> List[Document]:
    """
    Process a PDF file and extract text content
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of documents extracted from the PDF
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Add metadata
    for doc in documents:
        doc.metadata["file_path"] = file_path
        doc.metadata["file_type"] = "pdf"
        doc.metadata["page_number"] = doc.metadata.get("page", 0)
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    return text_splitter.split_documents(documents)

def process_image(file_path: str) -> List[Document]:
    """
    Process an image file and extract text and visual content
    
    Args:
        file_path: Path to the image file
        
    Returns:
        List of documents extracted from the image
    """
    # Use GPT-4o Vision to describe the image
    image = Image.open(file_path)
    
    # Convert image to base64 for API
    import base64
    from io import BytesIO
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Create vision model
    vision_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Get image description
    from langchain_core.messages import HumanMessage
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image in detail, including any text visible in the image."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
        ]
    )
    
    response = vision_llm.invoke([message])
    
    # Create document
    document = Document(
        page_content=response.content,
        metadata={
            "file_path": file_path,
            "file_type": "image",
            "image_description": "GPT-4o Vision analysis"
        }
    )
    
    return [document]

def ingest_multimodal_url(url: str) -> List[Document]:
    """
    Ingest content from a URL that might be a PDF, image, or other file type
    
    Args:
        url: URL of the content to ingest
        
    Returns:
        List of documents extracted from the URL
    """
    try:
        # Download the file
        file_path = download_file(url)
        
        # Process based on file type
        if file_path.lower().endswith('.pdf'):
            documents = process_pdf(file_path)
        elif any(file_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            documents = process_image(file_path)
        else:
            # Unsupported file type
            documents = [Document(
                page_content=f"Unsupported file type from URL: {url}",
                metadata={"source": url, "file_type": "unknown"}
            )]
        
        # Add source URL to metadata
        for doc in documents:
            doc.metadata["source"] = url
        
        # Clean up the temporary file
        os.unlink(file_path)
        
        return documents
    
    except Exception as e:
        # Return error document
        return [Document(
            page_content=f"Error processing URL {url}: {str(e)}",
            metadata={"source": url, "error": str(e)}
        )]

def process_youtube_video(url: str) -> List[Document]:
    """
    Process a YouTube video and extract transcript
    
    Args:
        url: URL of the YouTube video
        
    Returns:
        List of documents containing the video transcript
    """
    from langchain_community.document_loaders import YoutubeLoader
    
    try:
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,
            language=["en"]
        )
        
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        split_docs = text_splitter.split_documents(documents)
        
        # Add metadata
        for doc in split_docs:
            doc.metadata["source"] = url
            doc.metadata["file_type"] = "youtube"
        
        return split_docs
    
    except Exception as e:
        # Return error document
        return [Document(
            page_content=f"Error processing YouTube video {url}: {str(e)}",
            metadata={"source": url, "error": str(e), "file_type": "youtube"}
        )]

def ingest_multimodal_content(url: str) -> List[Document]:
    """
    Ingest content from a URL, detecting the type automatically
    
    Args:
        url: URL of the content to ingest
        
    Returns:
        List of documents extracted from the URL
    """
    # Check if it's a YouTube URL
    if "youtube.com" in url or "youtu.be" in url:
        return process_youtube_video(url)
    
    # Check if it's a web page or a file
    if any(url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.jpeg', '.png']):
        return ingest_multimodal_url(url)
    
    # Default to web page
    from langchain_community.document_loaders import WebBaseLoader
    
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        split_docs = text_splitter.split_documents(documents)
        
        # Add metadata
        for doc in split_docs:
            doc.metadata["source"] = url
            doc.metadata["file_type"] = "web"
        
        return split_docs
    
    except Exception as e:
        # Return error document
        return [Document(
            page_content=f"Error processing web page {url}: {str(e)}",
            metadata={"source": url, "error": str(e), "file_type": "web"}
        )]