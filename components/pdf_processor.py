"""
Enhanced PDF processing module for handling text, images, and mixed content
"""
from pypdf import PdfReader
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
from langchain_core.documents import Document
from typing import List, Tuple
import io
import os


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from PDF"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def extract_images_from_pdf(pdf_path: str, output_dir: str = None) -> List[Image.Image]:
    """
    Extract embedded images from PDF pages
    
    Parameters:
    - pdf_path: Path to PDF file
    - output_dir: Optional directory to save extracted images
    
    Returns:
    - List of PIL Image objects
    """
    images = []
    try:
        reader = PdfReader(pdf_path)
        
        for page_num, page in enumerate(reader.pages):
            if '/XObject' in page['/Resources']:
                xObject = page['/Resources']['/XObject'].get_object()
                
                for obj in xObject:
                    if xObject[obj]['/Subtype'] == '/Image':
                        try:
                            size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                            data = xObject[obj].get_data()
                            
                            if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                                mode = "RGB"
                            else:
                                mode = "P"
                            
                            image = Image.frombytes(mode, size, data)
                            images.append(image)
                            
                            if output_dir:
                                os.makedirs(output_dir, exist_ok=True)
                                image.save(f"{output_dir}/page_{page_num}_img_{len(images)}.png")
                        except Exception as e:
                            print(f"Error extracting image from page {page_num}: {e}")
    except Exception as e:
        print(f"Error processing PDF for images: {e}")
    
    return images


def convert_pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """
    Convert PDF pages to images (useful for scanned PDFs)
    
    Parameters:
    - pdf_path: Path to PDF file
    
    Returns:
    - List of PIL Image objects (one per page)
    """
    try:
        images = convert_from_path(pdf_path)
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []


def process_pdf_with_mixed_content(pdf_path: str, use_ocr: bool = True) -> List[Document]:
    """
    Process PDF with text, images, or mixed content
    
    Parameters:
    - pdf_path: Path to PDF file
    - use_ocr: Whether to use OCR on images
    
    Returns:
    - List of Document objects with extracted content
    """
    documents = []
    
    # Extract text
    text_content = extract_text_from_pdf(pdf_path)
    
    # Check if PDF has meaningful text
    has_text = len(text_content.strip()) > 50
    
    if has_text:
        # Create document from text
        doc = Document(
            page_content=text_content,
            metadata={
                "source": os.path.basename(pdf_path),
                "type": "pdf_text",
                "content_type": "text"
            }
        )
        documents.append(doc)
    
    # Extract embedded images
    embedded_images = extract_images_from_pdf(pdf_path)
    
    if embedded_images:
        for idx, img in enumerate(embedded_images):
            if use_ocr:
                # Extract text from image using OCR
                img_text = pytesseract.image_to_string(img)
                if img_text.strip():
                    doc = Document(
                        page_content=img_text,
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "type": "pdf_image",
                            "content_type": "image_ocr",
                            "image_index": idx
                        }
                    )
                    documents.append(doc)
    
    # If no text and no embedded images, treat as scanned PDF
    if not has_text and not embedded_images:
        page_images = convert_pdf_to_images(pdf_path)
        
        if use_ocr and page_images:
            for page_num, img in enumerate(page_images):
                page_text = pytesseract.image_to_string(img)
                if page_text.strip():
                    doc = Document(
                        page_content=page_text,
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "type": "pdf_scanned",
                            "content_type": "scanned_page",
                            "page_number": page_num + 1
                        }
                    )
                    documents.append(doc)
    
    return documents
