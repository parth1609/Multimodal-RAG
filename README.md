# DataMiner - Multimodal RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system that processes and queries documents, images, and PDFs with mixed content using Langchain, ChromaDB, and Groq LLMs.

## Features

✅ **Multi-format Support**
- Plain text documents (.txt)
- Images (PNG, JPG, JPEG) with OCR
- PDFs with text, images, or mixed content
- Automatic image extraction from PDFs

✅ **Intelligent Processing**
- Vision model integration (Google Gemini Vision)
- OCR for scanned documents (Tesseract)
- Automatic content type detection
- Metadata tracking (file type, timestamp, source)

✅ **Powerful Querying**
- Vector-based semantic search (ChromaDB + HuggingFace embeddings)
- LLM-powered answers (Groq Llama models)
- Source attribution and relevance scoring
- Context-aware responses

✅ **Dual Interface**
- **FastAPI**: RESTful API for programmatic access
- **Streamlit**: Interactive web UI for testing

## Architecture

```
┌─────────────────────────────────────────┐
│         User Interface Layer            │
│  ┌──────────────┐  ┌─────────────────┐ │
│  │   Streamlit  │  │    FastAPI      │ │
│  │     (UI)     │  │    (REST API)   │ │
│  └──────────────┘  └─────────────────┘ │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│       Document Processing Layer         │
│  ┌────────────────────────────────────┐ │
│  │  • Text Loader                     │ │
│  │  • PDF Processor (text/image)      │ │
│  │  • Image Processor (OCR + Vision)  │ │
│  └────────────────────────────────────┘ │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│         Storage & Retrieval Layer       │
│  ┌────────────────────────────────────┐ │
│  │  ChromaDB (Vector Store)           │ │
│  │  • HuggingFace Embeddings          │ │
│  │  • Metadata Management             │ │
│  │  • Semantic Search                 │ │
│  └────────────────────────────────────┘ │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│         Generation Layer                │
│  ┌────────────────────────────────────┐ │
│  │  Groq LLM (Llama 3)                │ │
│  │  • Context-aware answers           │ │
│  │  • Source citation                 │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

## Tech Stack

- **Framework**: Langchain, FastAPI, Streamlit
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace sentence-transformers
- **LLM**: Groq (Llama 3)
- **Vision**: Google Gemini Vision API
- **OCR**: Tesseract
- **PDF Processing**: pypdf, pdf2image

## Installation

### Prerequisites

- Python 3.9+
- Tesseract OCR installed:
  - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
  - Linux: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd DataMiner
```

2. **Create virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

**Get API Keys:**
- Gemini: [Google AI Studio](https://makersuite.google.com/app/apikey)
- Groq: [Groq Console](https://console.groq.com/)

## Usage

### Option 1: FastAPI (REST API)

1. **Start the API server**
```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```
The API will be available at `http://localhost:8000`

2. **API Documentation**
Visit `http://localhost:8000/docs` for interactive API documentation

#### API Endpoints

##### Upload Documents
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document.pdf" \
  -F "files=@image.png"
```

Response:
```json
{
  "message": "Documents processed successfully",
  "documents_processed": 2,
  "chunks_created": 15,
  "total_documents": 2
}
```

##### Query Documents
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings in the report?",
    "top_k": 3
  }'
```

Response:
```json
{
  "answer": "Based on the documents...",
  "sources": [
    {
      "source": "report.pdf",
      "type": "pdf_text",
      "content_preview": "The analysis shows..."
    }
  ],
  "relevance_scores": [0.92, 0.85],
  "processing_time": 1.23
}
```

##### List Documents
```bash
curl -X GET "http://localhost:8000/documents"
```

##### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### Option 2: Streamlit UI

1. **Start the Streamlit app**
```bash
streamlit run app.py
```

2. **Access the UI**
Open your browser to `http://localhost:8501`

3. **Using the Interface**
   - Upload files using the file uploader
   - View processed documents in the sidebar
   - Chat with your documents using the input box
   - See conversation history and responses

## Sample Queries

### Factual Questions
```
Q: What is the total revenue mentioned in the financial report?
Q: Who are the authors of this research paper?
Q: What date was this document created?
```

### Exploratory Queries
```
Q: Summarize the main points of the document
Q: What are the key findings?
Q: Explain the methodology used
```

### Cross-modal Queries
```
Q: What does the chart in the document show?
Q: Describe the image on page 3
Q: What information is in the diagram?
```

## Project Structure

```
DataMiner/
├── app.py                    # Streamlit UI
├── api.py                    # FastAPI backend
├── requirements.txt          # Dependencies
├── .env                      # Environment variables (create this)
├── README.md                 # This file
├── components/
│   ├── __init__.py
│   ├── document_loader.py    # Document loading logic
│   ├── pdf_processor.py      # Enhanced PDF processing
│   ├── vision_processor.py   # Image vision processing
│   ├── text_processor.py     # Text splitting
│   └── vector_store.py       # ChromaDB integration
└── chroma_db/                # Vector database storage (auto-created)
```

## Design Decisions

### 1. Vector Database: ChromaDB
- **Why**: Open-source, lightweight, local-first
- **Trade-off**: Not as scalable as Pinecone for production, but perfect for development

### 2. Embeddings: HuggingFace sentence-transformers
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Why**: Fast, efficient, runs locally, no API costs
- **Trade-off**: Slightly less accurate than OpenAI embeddings

### 3. LLM: Groq
- **Model**: Llama 3 (70B/8B)
- **Why**: Extremely fast inference, free tier available
- **Trade-off**: Requires API key, rate limits

### 4. Vision: Google Gemini Vision
- **Why**: Excellent image understanding, handles complex visuals
- **Alternative**: OCR for text-heavy images

### 5. PDF Processing Strategy
- Text extraction first (pypdf)
- Image extraction for embedded images
- OCR fallback for scanned PDFs
- Maintains text-image relationships

## Performance

- **Upload Processing**: ~2-5 seconds per document
- **Query Response Time**: < 2 seconds (target met)
- **Embedding Generation**: ~100ms per chunk
- **LLM Generation**: ~1-2 seconds (Groq)

## Error Handling

- Comprehensive try-catch blocks
- Logging for debugging
- Graceful fallbacks (e.g., OCR if text extraction fails)
- User-friendly error messages

## Limitations

1. **API Rate Limits**: Groq and Gemini have free tier limits
2. **Local Storage**: ChromaDB persists locally (not distributed)
3. **OCR Accuracy**: Depends on image quality
4. **Large PDFs**: May take longer to process

## Future Enhancements

- [ ] Hybrid search (dense + sparse)
- [ ] Reranking mechanism
- [ ] Query expansion
- [ ] Caching layer
- [ ] Batch processing
- [ ] DOCX, XLSX support
- [ ] Conversation memory
- [ ] Document summarization
- [ ] Unit tests
- [ ] Docker containerization

## Troubleshooting

### Tesseract not found
```
Error: TesseractNotFoundError
```
**Solution**: Install Tesseract OCR and add to PATH

### ChromaDB errors
```
Error: Failed to persist vectorstore
```
**Solution**: Delete `chroma_db` folder and restart

### API Key errors
```
Error: The api_key client option must be set
```
**Solution**: Check `.env` file has correct keys

### Import errors
```
ModuleNotFoundError: No module named 'X'
```
**Solution**: `pip install -r requirements.txt`

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License

## Contact

For issues or questions, please open an issue on GitHub.

---

**Built with ❤️ using Langchain, ChromaDB, and Groq**