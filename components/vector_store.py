from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
import datetime
import os

def create_vector_store(docs: List[Document], persist_directory: str = "./chroma_db"):
    """Create and persist Chroma vector store with metadata."""
    # Add metadata
    for doc in docs:
        doc.metadata.update({
            "file_type": doc.metadata.get("type", "unknown"),
            "upload_timestamp": datetime.datetime.now().isoformat(),
            "source": doc.metadata.get("source", "unknown")
        })
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    # Note: Chroma 0.4.x auto-persists, no manual persist() needed
    return vectorstore
