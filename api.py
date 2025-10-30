"""
FastAPI backend for DataMiner - Document processing and querying API
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from datetime import datetime
from tempfile import NamedTemporaryFile
import logging

from components.document_loader import load_documents
from components.vision_processor import process_image_with_vision
from components.text_processor import split_documents
from components.vector_store import create_vector_store
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

app = FastAPI(title="DataMiner API", version="1.0.0")

# Global state for vector store
vectorstore = None
processed_docs = []


class QueryRequest(BaseModel):
    """Query request model"""
    query: str
    top_k: Optional[int] = 2


class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    sources: List[dict]
    relevance_scores: List[float]
    processing_time: float


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "DataMiner API",
        "version": "1.0.0",
        "endpoints": {
            "/upload": "POST - Upload documents",
            "/query": "POST - Query documents",
            "/documents": "GET - List processed documents",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "documents_processed": len(processed_docs),
        "vectorstore_initialized": vectorstore is not None
    }


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and process documents (text, PDF, images)
    
    Parameters:
    - files: List of files to upload
    
    Returns:
    - Success message with processed document count
    """
    global vectorstore, processed_docs
    
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        logger.info(f"Processing {len(files)} files")
        
        # Save uploaded files temporarily
        file_paths = []
        for file in files:
            suffix = f".{file.filename.split('.')[-1]}"
            with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                file_paths.append((tmp.name, file.filename))
        
        # Process documents
        docs = []
        errors = []
        for path, name in file_paths:
            ext = os.path.splitext(path)[1].lower()
            
            try:
                if ext in ['.png', '.jpg', '.jpeg']:
                    # Process image with vision
                    description = process_image_with_vision(path, GEMINI_API_KEY)
                    doc = Document(
                        page_content=description,
                        metadata={
                            "source": name,
                            "type": "image",
                            "upload_timestamp": datetime.now().isoformat()
                        }
                    )
                    docs.append(doc)
                elif ext in ['.txt', '.pdf']:
                    # Load text/PDF
                    loaded_docs = load_documents([path])
                    if loaded_docs:
                        for d in loaded_docs:
                            d.metadata["source"] = name
                            d.metadata["upload_timestamp"] = datetime.now().isoformat()
                        docs.extend(loaded_docs)
                    else:
                        errors.append(f"{name}: No content could be extracted")
                else:
                    errors.append(f"{name}: File type {ext} not supported (use .txt, .pdf, .png, .jpg, .jpeg)")
                
                # Clean up temp file
                if os.path.exists(path):
                    os.unlink(path)
                
            except Exception as e:
                logger.error(f"Error processing {name}: {str(e)}")
                errors.append(f"{name}: {str(e)}")
                # Clean up temp file on error
                if os.path.exists(path):
                    os.unlink(path)
        
        if not docs:
            error_details = "No documents could be processed. Errors: " + "; ".join(errors) if errors else "No documents could be processed"
            raise HTTPException(status_code=400, detail=error_details)
        
        # Split and store in vector database
        splits = split_documents(docs)
        vectorstore = create_vector_store(splits)
        processed_docs.extend(docs)
        
        logger.info(f"Successfully processed {len(docs)} documents")
        
        response = {
            "message": "Documents processed successfully",
            "documents_processed": len(docs),
            "chunks_created": len(splits),
            "total_documents": len(processed_docs)
        }
        
        if errors:
            response["errors"] = errors
            response["message"] = f"Processed {len(docs)} documents with {len(errors)} error(s)"
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query processed documents
    
    Parameters:
    - query: The question to ask
    - top_k: Number of relevant chunks to retrieve (default: 2)
    
    Returns:
    - Answer, sources, and relevance scores
    """
    global vectorstore
    
    try:
        if vectorstore is None:
            raise HTTPException(
                status_code=400,
                detail="No documents have been processed. Please upload documents first."
            )
        
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        start_time = datetime.now()
        
        # Retrieve relevant documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": request.top_k})
        relevant_docs = retriever.invoke(request.query)
        
        # Get relevance scores
        relevance_scores = []
        sources = []
        for doc in relevant_docs:
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "type": doc.metadata.get("type", "unknown"),
                "content_preview": doc.page_content[:200] + "..."
            })
            # Note: Actual relevance scores would require similarity search
            relevance_scores.append(0.85)  # Placeholder
        
        # Generate answer using LLM
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            api_key=GROQ_API_KEY,
            temperature=0.2,
            max_tokens=1024
        )
        
        template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always cite the sources you used.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt_template = ChatPromptTemplate.from_template(template)
        qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | llm
        )
        
        response = qa_chain.invoke(request.query)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Query processed in {processing_time:.2f}s")
        
        return QueryResponse(
            answer=response.content,
            sources=sources,
            relevance_scores=relevance_scores,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/documents")
def list_documents():
    """
    List all processed documents with metadata
    
    Returns:
    - List of documents with metadata
    """
    if not processed_docs:
        return {"documents": [], "count": 0}
    
    docs_info = []
    for doc in processed_docs:
        docs_info.append({
            "source": doc.metadata.get("source", "Unknown"),
            "type": doc.metadata.get("type", "unknown"),
            "upload_timestamp": doc.metadata.get("upload_timestamp", "N/A"),
            "content_length": len(doc.page_content)
        })
    
    return {"documents": docs_info, "count": len(docs_info)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
