from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from typing import List
import os
from PIL import Image
import pytesseract
from components.pdf_processor import process_pdf_with_mixed_content


def load_documents(file_paths: List[str]) -> List[Document]:
    """Load documents from file paths, supporting text, PDF, and images."""
    docs = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            continue
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.txt':
                loader = TextLoader(file_path)
                docs.extend(loader.load())
            elif ext == '.pdf':
                # Use enhanced PDF processor for mixed content
                pdf_docs = process_pdf_with_mixed_content(file_path, use_ocr=True)
                docs.extend(pdf_docs)
            elif ext in ['.png', '.jpg', '.jpeg']:
                # For images, create a document with OCR text
                image = Image.open(file_path)
                text = pytesseract.image_to_string(image)
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(file_path),
                        "type": "image"
                    }
                )
                docs.append(doc)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return docs
