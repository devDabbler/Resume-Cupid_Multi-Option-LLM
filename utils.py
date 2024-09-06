import re
import logging
import io
import PyPDF2
from docx import Document
import numpy as np
import streamlit as st

# Set up logging
def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger

# Create a root logger
root_logger = get_logger(__name__)

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the root logger
root_logger.addHandler(console_handler)

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
        return ""
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