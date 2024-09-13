import re
import logging
import io
import PyPDF2
from docx import Document
import numpy as np
import streamlit as st
import sys
import threading
import json
import os
import sqlite3
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import pandas as pd
from llama_analyzer import process_resume
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Initialize logger with environment-based settings
def setup_logger():
    environment = os.getenv('ENVIRONMENT', 'development')
    logging_level = logging.INFO if environment == 'production' else logging.DEBUG
    logging.basicConfig(level=logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging_level)
    return logger

logger = setup_logger()

# Custom exception handler
def exception_handler(exc_type, exc_value, exc_traceback):
    logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = exception_handler

# Thread-safe logger class
class ThreadSafeLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self._lock = threading.Lock()

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        with self._lock:
            super()._log(level, msg, args, exc_info, extra, stack_info)

logging.setLoggerClass(ThreadSafeLogger)

# Function to extract text from PDF files
@st.cache_data
def extract_text_from_pdf(file_content: bytes) -> str:
    logger.debug("Extracting text from PDF...")
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = "".join([page.extract_text() for page in pdf_reader.pages])
        if not text.strip():
            logger.warning("Extracted PDF text is empty or contains only whitespace.")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

# Function to extract text from DOCX files
@st.cache_data
def extract_text_from_docx(file_content: bytes) -> str:
    logger.debug("Extracting text from DOCX...")
    try:
        doc = Document(io.BytesIO(file_content))
        text = "\n".join([para.text for para in doc.paragraphs])
        if not text.strip():
            logger.warning("Extracted DOCX text is empty or contains only whitespace.")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        raise ValueError(f"Failed to extract text from DOCX: {str(e)}")

# Extract text based on file type (PDF or DOCX)
def extract_text_from_file(file) -> str:
    logger.debug(f"Extracting text from file: {file.name}")
    file_content = file.read()
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(file_content)
    elif file_extension in ['docx', 'doc']:
        return extract_text_from_docx(file_content)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

# Cosine similarity calculation between two embeddings
def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    try:
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return similarity
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        raise ValueError(f"Error calculating similarity: {str(e)}")

# Text preprocessing (remove special characters and extra spaces)
def preprocess_text(text: str) -> str:
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', cleaned_text).strip().lower()

# Split files into batches
def split_into_batches(files: List, batch_size: int) -> List[List]:
    return [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

# Asynchronous batch processing of resumes
async def process_batch(batch: List, resume_processor, job_description: str, importance_factors: Dict[str, float], candidate_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                executor,
                process_resume,
                file,
                resume_processor,
                job_description,
                importance_factors,
                candidate_data
            )
            for file, candidate_data in zip(batch, candidate_data_list)
        ]
        return await asyncio.gather(*tasks)

# Process all batches with progress tracking
async def process_all_batches(batches: List[List], resume_processor, job_description: str, importance_factors: Dict[str, float], candidate_data_list: List[Dict[str, Any]], progress_bar):
    results = []
    for batch in batches:
        batch_results = await process_batch(batch, resume_processor, job_description, importance_factors, candidate_data_list[len(results):len(results) + len(batch)])
        results.extend(batch_results)
        progress_bar.progress(len(results) / sum(len(batch) for batch in batches))
    return results

# Generate error results for failed resume processing
def _generate_error_result(file_name: str, error_message: str) -> dict:
    return {
        'file_name': file_name,
        'error': error_message,
        'match_score': 0,
        'recommendation': 'Unable to provide a recommendation due to an error.',
        'analysis': 'Unable to complete analysis',
        'brief_summary': 'Error occurred during processing',
        'experience_and_project_relevance': 'Unable to assess due to an error',
        'strengths': [],
        'areas_for_improvement': [],
        'skills_gap': [],
        'recruiter_questions': [],
        'project_relevance': ''
    }

# Process resumes in parallel using threading
def process_resumes_in_parallel(resume_files, resume_processor, job_description, importance_factors, candidate_data_list, job_title):
    logger.debug("Starting parallel resume processing...")
    def process_with_context(file, candidate_data):
        try:
            resume_text = extract_text_from_file(file)
            result = resume_processor.analyze_match(resume_text, job_description, candidate_data, job_title)
            result['file_name'] = file.name
            return result
        except Exception as e:
            logger.error(f"Error processing resume {file.name}: {str(e)}", exc_info=True)
            return _generate_error_result(file.name, str(e))

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_with_context, file, candidate_data)
            for file, candidate_data in zip(resume_files, candidate_data_list)
        ]
        add_script_run_ctx(futures)
        results = [future.result() for future in as_completed(futures)]
    
    return results

# Display results in a sorted table
def display_results(evaluation_results: List[Dict[str, Any]], run_id: str, save_feedback_func):
    st.subheader("Stack Ranking of Candidates")
    for result in evaluation_results:
        result['match_score'] = result.get('match_score', 0)
        result['recommendation'] = result.get('recommendation', 'No recommendation available')
    
    evaluation_results.sort(key=lambda x: x['match_score'], reverse=True)
    
    df = pd.DataFrame(evaluation_results)
    df['Rank'] = range(1, len(df) + 1)
    df = df[['Rank', 'file_name', 'match_score', 'recommendation']]

    df.columns = ['Rank', 'Candidate', 'Match Score (%)', 'Recommendation']
    st.table(df)

    for i, result in enumerate(evaluation_results, 1):
        with st.expander(f"Rank {i}: {result['file_name']} - Detailed Analysis"):
            if 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                st.markdown("### Brief Summary")
                st.write(result.get('brief_summary', 'No brief summary available'))
                st.markdown("### Match Score")
                st.write(f"{result.get('match_score', 'N/A')}%")
                st.markdown("### Recommendation")
                st.write(result.get('recommendation', 'No recommendation available'))

# Extract job description from Fractal's Workday job posting URL
def extract_job_description(url):
    if not is_valid_fractal_job_link(url):
        raise ValueError("Invalid job link. Please use a link from Fractal's career site.")
    
    options = Options()
    options.add_argument("--headless")

    try:
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        WebDriverWait(driver, 20).until(lambda d: d.execute_script('return document.readyState') == 'complete')
        job_description_element = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-automation-id='jobPostingDescription']"))
        )
        job_description = job_description_element.text
        return job_description
    except Exception as e:
        logger.error(f"Failed to extract job description: {str(e)}")
        raise
    finally:
        driver.quit()

# Validate Fractal job link format
def is_valid_fractal_job_link(url):
    pattern = r'^https?://fractal\.wd1\.myworkdayjobs\.com/.*Careers/.*'
    return re.match(pattern, url) is not None
