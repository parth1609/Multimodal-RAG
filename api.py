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
from langchain_community.retrievers import BM25Retriever
import re
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
indexed_docs = []
bm25_retriever = None


class QueryRequest(BaseModel):
    """Query request model"""
    query: str
    top_k: Optional[int] = 5
    hybrid: Optional[bool] = True
    rerank: Optional[bool] = False


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
        
        splits = split_documents(docs)
        vectorstore = create_vector_store(splits)
        processed_docs.extend(docs)
        indexed_docs.clear()
        indexed_docs.extend(splits)
        try:
            bm25 = BM25Retriever.from_documents(indexed_docs)
            bm25.k = 20
            globals()["bm25_retriever"] = bm25
        except Exception as e:
            logger.error(f"BM25 init error: {str(e)}")
        
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
        
        def is_image_like(doc: Document) -> bool:
            t = (doc.metadata or {}).get("type", "").lower()
            ct = (doc.metadata or {}).get("content_type", "").lower()
            return t in {"image", "pdf_image", "pdf_scanned"} or ct in {"image_ocr", "scanned_page"}

        def wants_images(q: str) -> bool:
            kw = ["image", "chart", "figure", "diagram", "photo", "screenshot", "scan", "table", "graph"]
            return any(re.search(r"\b" + k + r"\b", q, flags=re.I) for k in kw)

        k = max(1, request.top_k or 5)
        dense_k = min(50, max(k * 2, 10))
        sparse_k = min(50, max(k * 2, 10))

        dense_results = []
        try:
            dense_results = vectorstore.similarity_search_with_score(request.query, k=dense_k)
        except Exception:
            try:
                docs_only = vectorstore.similarity_search(request.query, k=dense_k)
                dense_results = [(d, 0.0) for d in docs_only]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Dense retrieval failed: {str(e)}")

        sparse_results = []
        if request.hybrid and bm25_retriever is not None:
            try:
                sparse_docs = bm25_retriever.get_relevant_documents(request.query)[:sparse_k]
                sparse_results = [(d, 0.0) for d in sparse_docs]
            except Exception as e:
                logger.error(f"BM25 retrieval error: {str(e)}")

        def key_for(doc: Document) -> str:
            src = (doc.metadata or {}).get("source", "")
            typ = (doc.metadata or {}).get("type", "")
            return f"{src}|{typ}|{hash(doc.page_content)}"

        combined = {}
        for rank, (doc, score) in enumerate(dense_results):
            combined[key_for(doc)] = {"doc": doc, "dense_rank": rank, "sparse_rank": None}
        for rank, (doc, score) in enumerate(sparse_results):
            kf = key_for(doc)
            if kf in combined:
                combined[kf]["sparse_rank"] = rank
            else:
                combined[kf] = {"doc": doc, "dense_rank": None, "sparse_rank": rank}

        alpha = 0.5 if request.hybrid else 1.0
        want_img = wants_images(request.query)
        scored = []
        for item in combined.values():
            d_rank = item["dense_rank"]
            s_rank = item["sparse_rank"]
            d_score = 1.0 / (1 + d_rank) if d_rank is not None else 0.0
            s_score = 1.0 / (1 + s_rank) if s_rank is not None else 0.0
            score = alpha * d_score + (1 - alpha) * s_score
            if want_img and is_image_like(item["doc"]):
                score += 0.15
            scored.append((item["doc"], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [d for d, s in scored[:k]]

        if request.rerank:
            try:
                from langchain_cohere import CohereRerank
                cohere_key = os.getenv("COHERE_API_KEY")
                if cohere_key:
                    reranker = CohereRerank(model="rerank-english-v3.0", top_n=k, api_key=cohere_key)
                    selected = reranker.compress_documents(selected, query=request.query)
            except Exception as e:
                logger.error(f"Rerank error: {str(e)}")

        relevance_scores = []
        sources = []
        for doc in selected:
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "type": doc.metadata.get("type", "unknown"),
                "content_preview": doc.page_content[:200] + "...",
                "page_number": doc.metadata.get("page_number"),
                "image_index": doc.metadata.get("image_index")
            })
            relevance_scores.append(0.0)
        
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
        context_str = "\n\n---\n\n".join([d.page_content for d in selected])
        qa_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt_template
            | llm
        )
        response = qa_chain.invoke({"context": context_str, "question": request.query})
        
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
