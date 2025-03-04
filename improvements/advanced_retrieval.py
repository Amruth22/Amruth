"""
Advanced retrieval techniques to improve RAG performance
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any

# Initialize models
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def create_hybrid_retriever(vector_store):
    """
    Create a hybrid retriever that combines keyword search with vector search
    """
    from langchain_community.retrievers import BM25Retriever
    
    # Create BM25 retriever for keyword-based search
    document_list = []
    for doc_id in vector_store.docstore._dict:
        document = vector_store.docstore.search(doc_id)
        document_list.append(Document(page_content=document.page_content, metadata=document.metadata))
    
    bm25_retriever = BM25Retriever.from_documents(document_list)
    bm25_retriever.k = 5
    
    # Create vector retriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Create hybrid retriever
    from langchain.retrievers import EnsembleRetriever
    
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    return hybrid_retriever

def create_contextual_compression_retriever(vector_store):
    """
    Create a retriever that compresses documents to extract only relevant parts
    """
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 8})
    
    # Create compressor using LLM
    compressor = LLMChainExtractor.from_llm(llm)
    
    # Create compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever

def create_query_transformation_retriever(vector_store):
    """
    Create a retriever that transforms the query before retrieval
    """
    # Query transformation prompt
    query_transform_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at transforming user questions into effective search queries.
        Your goal is to create a search query that will help find documents to answer the user's question.
        Identify the key concepts and terms that should be included in the search query.
        Do not try to answer the question, just reformulate it into an effective search query."""),
        ("human", "{question}")
    ])
    
    # Query transformation chain
    query_transform_chain = (
        query_transform_prompt 
        | llm 
        | StrOutputParser()
    )
    
    # Base retriever
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Create retriever with query transformation
    from langchain.retrievers import TransformationRetriever
    
    transformation_retriever = TransformationRetriever(
        retriever=base_retriever,
        transform_query_fn=lambda x: query_transform_chain.invoke({"question": x})
    )
    
    return transformation_retriever

def create_self_query_retriever(vector_store):
    """
    Create a retriever that can filter documents based on metadata
    """
    from langchain.retrievers.self_query.base import SelfQueryRetriever
    from langchain.chains.query_constructor.base import AttributeInfo
    
    # Define metadata fields
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The URL source of the document",
            type="string",
        ),
        AttributeInfo(
            name="title",
            description="The title of the document or webpage",
            type="string",
        ),
        AttributeInfo(
            name="date",
            description="The date when the document was created or ingested",
            type="string",
        )
    ]
    
    # Create self-query retriever
    self_query_retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vector_store,
        document_contents="Documents containing information about various topics",
        metadata_field_info=metadata_field_info,
        search_kwargs={"k": 5}
    )
    
    return self_query_retriever

def create_parent_document_retriever(vector_store, collection_name="parent_documents"):
    """
    Create a retriever that returns parent documents after searching through child chunks
    """
    from langchain.retrievers import ParentDocumentRetriever
    from langchain_community.storage import InMemoryStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Create storage for parent documents
    store = InMemoryStore()
    
    # Create text splitter for child documents
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    
    # Create parent document retriever
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        search_kwargs={"k": 5}
    )
    
    return parent_retriever

def create_multi_query_retriever(vector_store):
    """
    Create a retriever that generates multiple queries from a single user question
    """
    from langchain.retrievers import MultiQueryRetriever
    
    # Create multi-query retriever
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        llm=llm
    )
    
    return multi_query_retriever

def get_advanced_retriever(vector_store, retriever_type="hybrid"):
    """
    Get an advanced retriever based on the specified type
    
    Args:
        vector_store: The FAISS vector store
        retriever_type: Type of advanced retriever to create
        
    Returns:
        An advanced retriever
    """
    retriever_map = {
        "hybrid": create_hybrid_retriever,
        "contextual": create_contextual_compression_retriever,
        "query_transform": create_query_transformation_retriever,
        "self_query": create_self_query_retriever,
        "parent": create_parent_document_retriever,
        "multi_query": create_multi_query_retriever
    }
    
    if retriever_type not in retriever_map:
        raise ValueError(f"Retriever type '{retriever_type}' not supported. Choose from: {list(retriever_map.keys())}")
    
    return retriever_map[retriever_type](vector_store)