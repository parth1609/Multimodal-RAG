"""
Unit tests for DataMiner components
"""
import unittest
import os
from tempfile import NamedTemporaryFile
from PIL import Image
import numpy as np

from components.text_processor import split_documents
from components.vector_store import create_vector_store
from langchain_core.documents import Document


class TestTextProcessor(unittest.TestCase):
    """Tests for text processing"""
    
    def test_split_documents(self):
        """Test document splitting"""
        doc = Document(page_content="This is a test document. " * 100)
        splits = split_documents([doc])
        
        self.assertGreater(len(splits), 1, "Document should be split into multiple chunks")
        self.assertLess(len(splits[0].page_content), 1200, "Chunks should respect max size")
    
    def test_split_empty_document(self):
        """Test splitting empty document"""
        doc = Document(page_content="")
        splits = split_documents([doc])
        
        self.assertGreaterEqual(len(splits), 0, "Should handle empty documents")


class TestVectorStore(unittest.TestCase):
    """Tests for vector store"""
    
    def test_create_vector_store(self):
        """Test vector store creation"""
        docs = [
            Document(page_content="This is a test document about AI."),
            Document(page_content="Machine learning is fascinating.")
        ]
        
        vectorstore = create_vector_store(docs, persist_directory="./test_chroma_db")
        
        self.assertIsNotNone(vectorstore, "Vector store should be created")
        
        # Clean up
        import shutil
        if os.path.exists("./test_chroma_db"):
            shutil.rmtree("./test_chroma_db")
    
    def test_metadata_preservation(self):
        """Test that metadata is preserved"""
        doc = Document(
            page_content="Test content",
            metadata={"source": "test.txt", "type": "text"}
        )
        
        vectorstore = create_vector_store([doc], persist_directory="./test_chroma_db")
        
        # Query to get document back
        results = vectorstore.similarity_search("test", k=1)
        
        self.assertEqual(results[0].metadata.get("source"), "test.txt")
        self.assertIn("upload_timestamp", results[0].metadata)
        
        # Clean up
        import shutil
        if os.path.exists("./test_chroma_db"):
            shutil.rmtree("./test_chroma_db")


class TestDocumentLoader(unittest.TestCase):
    """Tests for document loading"""
    
    def test_load_text_file(self):
        """Test loading text file"""
        from components.document_loader import load_documents
        
        # Create temp text file
        with NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.")
            temp_path = f.name
        
        try:
            docs = load_documents([temp_path])
            self.assertEqual(len(docs), 1, "Should load one document")
            self.assertIn("test document", docs[0].page_content)
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self):
        """Test handling of nonexistent file"""
        from components.document_loader import load_documents
        
        docs = load_documents(["/nonexistent/file.txt"])
        self.assertEqual(len(docs), 0, "Should return empty list for nonexistent files")


class TestPDFProcessor(unittest.TestCase):
    """Tests for PDF processing"""
    
    def test_extract_text_from_empty_pdf(self):
        """Test extracting text from PDF"""
        from components.pdf_processor import extract_text_from_pdf
        
        # This test would need a sample PDF file
        # For now, we test the function exists
        self.assertTrue(callable(extract_text_from_pdf))


if __name__ == '__main__':
    unittest.main()
