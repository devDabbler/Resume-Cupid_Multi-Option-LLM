import re
import logging
import io
import PyPDF2
from docx import Document
import numpy as np
import streamlit as st
import sys
import traceback
import threading
import json
import os

# Set up logging
def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger

# Create a root logger
root_logger = get_logger(__name__)

# Environment-based logging level
def setup_logger():
    environment = os.getenv('ENVIRONMENT', 'development')  # Default to 'development' if not set
    if environment == 'production':
        logging_level = logging.INFO  # Only log important information
    else:
        logging_level = logging.DEBUG  # Verbose logging for development
    
    logging.basicConfig(level=logging_level)
    root_logger.setLevel(logging_level)

# Call the logger setup
setup_logger()

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the root logger
root_logger.addHandler(console_handler)

# Custom exception handler
def exception_handler(exc_type, exc_value, exc_traceback):
    root_logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = exception_handler

# Thread-safe logger
class ThreadSafeLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self._lock = threading.Lock()

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        with self._lock:
            super()._log(level, msg, args, exc_info, extra, stack_info)

logging.setLoggerClass(ThreadSafeLogger)

# Custom formatter for structured logging
class StructuredFormatter(logging.Formatter):
    def format(self, record):
        if isinstance(record.msg, dict):
            return json.dumps({
                'timestamp': self.formatTime(record),
                'name': record.name,
                'level': record.levelname,
                'message': record.msg
            })
        return super().format(record)

# Set up structured logging
structured_handler = logging.StreamHandler(sys.stdout)
structured_handler.setLevel(logging.DEBUG)
structured_handler.setFormatter(StructuredFormatter())
root_logger.addHandler(structured_handler)

@st.cache_data
def extract_text_from_pdf(file_content: bytes) -> str:
    root_logger.debug("Extracting text from PDF...")
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        if not text.strip():
            root_logger.warning("Extracted PDF text is empty or contains only whitespace.")
        
        return text
    except Exception as e:
        root_logger.error(f"Error extracting text from PDF: {str(e)}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

@st.cache_data
def extract_text_from_docx(file_content: bytes) -> str:
    root_logger.debug("Extracting text from DOCX...")
    text = ""
    try:
        doc = Document(io.BytesIO(file_content))
        for para in doc.paragraphs:
            text += para.text + "\n"
        
        if not text.strip():
            root_logger.warning("Extracted DOCX text is empty or contains only whitespace.")
        
        return text
    except Exception as e:
        root_logger.error(f"Error extracting text from DOCX: {str(e)}")
        raise ValueError(f"Failed to extract text from DOCX: {str(e)}")

def extract_text_from_file(file) -> str:
    root_logger.debug(f"Extracting text from file: {file.name}")
    file_content = file.read()
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(file_content)
    elif file_extension in ['docx', 'doc']:
        return extract_text_from_docx(file_content)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    try:
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return similarity
    except Exception as e:
        root_logger.error(f"Error calculating similarity: {str(e)}")
        raise ValueError(f"Error calculating similarity: {str(e)}")
    
def preprocess_text(text: str) -> str:
    # Remove special characters and extra whitespace
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text.lower()

__all__ = [
    'extract_text_from_file',
    'calculate_similarity',
    'preprocess_text',
    'get_logger'
]
