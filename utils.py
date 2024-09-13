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
import sqlite3
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Create a root logger
root_logger = logging.getLogger(__name__)

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

def get_logger(name):
    return logging.getLogger(name)

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

def split_into_batches(files: List, batch_size: int) -> List[List]:
    """Split the list of files into batches of the specified size."""
    return [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

async def process_batch(batch: List, resume_processor, job_description: str, importance_factors: Dict[str, float], candidate_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of resumes asynchronously."""
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

async def process_all_batches(batches: List[List], resume_processor, job_description: str, importance_factors: Dict[str, float], candidate_data_list: List[Dict[str, Any]], progress_bar):
    """Process all batches of resumes and update the progress bar."""
    results = []
    for batch in batches:
        batch_results = await process_batch(batch, resume_processor, job_description, importance_factors, candidate_data_list[len(results):len(results)+len(batch)])
        results.extend(batch_results)
        progress_bar.progress((len(results) / sum(len(batch) for batch in batches)))
    return results

@st.cache_data
def process_resume(resume_file, _resume_processor, job_description, importance_factors, candidate_data, job_title):
    logger = get_logger(__name__)
    logger.debug(f"Processing resume: {resume_file.name} with {_resume_processor.backend} backend")
    try:
        logger.debug("Extracting text from file")
        resume_text = extract_text_from_file(resume_file)
        logger.debug(f"Extracted text length: {len(resume_text)}")
        
        logger.debug(f"Arguments for analyze_match: resume_text={resume_text[:20]}..., job_description={job_description[:20]}..., candidate_data={candidate_data}, job_title={job_title}")
        logger.debug(f"Number of arguments: {len([resume_text, job_description, candidate_data, job_title])}")
        result = _resume_processor.analyze_match(resume_text, job_description, candidate_data, job_title)
        logger.debug(f"Analysis result: {result}")
        
        if 'error' in result:
            logger.error(f"Error in resume analysis: {result['error']}")
            return _generate_error_result(resume_file.name, result['error'])
        
        # Post-process the match score
        raw_score = result.get('match_score', 0)
        adjusted_score = max(0, min(100, raw_score * 1.2))  # Scales up scores by 20%
        result['match_score'] = round(adjusted_score)
        
        result['file_name'] = resume_file.name
        result['brief_summary'] = result.get('brief_summary', 'No brief summary available')
        result['recommendation'] = _get_recommendation(result['match_score'])
        result['experience_and_project_relevance'] = result.get('experience_and_project_relevance', 'No experience and project relevance data available')
        result['skills_gap'] = result.get('skills_gap', [])
        result['recruiter_questions'] = result.get('recruiter_questions', [])
        
        # Calculate strengths and areas for improvement
        result['strengths'] = _calculate_strengths(result)
        result['areas_for_improvement'] = _calculate_areas_for_improvement(result)
        
        return result
    except Exception as e:
        logger.error(f"Error processing resume {resume_file.name}: {str(e)}", exc_info=True)
        return _generate_error_result(resume_file.name, str(e))

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

def _get_recommendation(match_score: int) -> str:
    if match_score < 50:
        return "Do not recommend for interview"
    elif 50 <= match_score < 65:
        return "Recommend for interview with reservations"
    elif 65 <= match_score < 80:
        return "Recommend for interview"
    else:
        return "Highly recommend for interview"

def _calculate_strengths(result: dict) -> List[str]:
    strengths = []
    if result['match_score'] >= 80:
        strengths.append("Excellent overall match to job requirements")
    elif result['match_score'] >= 65:
        strengths.append("Good overall match to job requirements")
    if not result['skills_gap']:
        strengths.append("No significant skills gaps identified")
    if 'experience' in result['experience_and_project_relevance'].lower():
        strengths.append("Relevant work experience")
    if 'project' in result['experience_and_project_relevance'].lower():
        strengths.append("Relevant project experience")
    return strengths

def _calculate_areas_for_improvement(result: dict) -> List[str]:
    areas_for_improvement = []
    if result['skills_gap']:
        areas_for_improvement.append("Address identified skills gaps")
    if result['match_score'] < 65:
        areas_for_improvement.append("Improve overall alignment with job requirements")
    if 'lack' in result['experience_and_project_relevance'].lower() or 'missing' in result['experience_and_project_relevance'].lower():
        areas_for_improvement.append("Gain more relevant experience")
    return areas_for_improvement

def process_resumes_in_parallel(resume_files, resume_processor, job_description, importance_factors, candidate_data_list, job_title):
    logger = get_logger(__name__)
    def process_with_context(file, candidate_data):
        try:
            resume_text = extract_text_from_file(file)
            logger.debug(f"Calling analyze_match with args: {resume_text[:20]}..., {job_description[:20]}..., {candidate_data}, {job_title}")
            result = resume_processor.analyze_match(
                resume_text,
                job_description,
                candidate_data,
                job_title
            )
            result['file_name'] = file.name
            return result
        except Exception as e:
            logger.error(f"Error processing resume {file.name}: {str(e)}", exc_info=True)
            return _generate_error_result(file.name, str(e))

    with ThreadPoolExecutor() as executor:
        futures = []
        for file, candidate_data in zip(resume_files, candidate_data_list):
            future = executor.submit(process_with_context, file, candidate_data)
            add_script_run_ctx(future)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            results.append(future.result())
    
    return results

def display_results(evaluation_results: List[Dict[str, Any]], run_id: str, save_feedback_func):
    st.subheader("Stack Ranking of Candidates")

    # Ensure all required keys exist in each result, with default values if missing
    for result in evaluation_results:
        result['match_score'] = result.get('match_score', 0)
        result['recommendation'] = result.get('recommendation', 'No recommendation available')
    
    # Sort results by match_score
    evaluation_results.sort(key=lambda x: x['match_score'], reverse=True)

    # Create a DataFrame for the main results
    df = pd.DataFrame(evaluation_results)
    df['Rank'] = range(1, len(df) + 1)

    # Select only the columns we want to display
    display_columns = ['Rank', 'file_name', 'match_score', 'recommendation']
    df = df[[col for col in display_columns if col in df.columns]]

    # Rename columns for display
    df.columns = ['Rank', 'Candidate', 'Match Score (%)', 'Recommendation']

    # Display the main results table
    st.table(df)

    # Display detailed results for each candidate
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

                st.markdown("### Experience and Project Relevance")
                st.write(result.get('experience_and_project_relevance', 'No experience and project relevance data available'))
                
                st.markdown("### Skills Gap")
                skills_gap = result.get('skills_gap', [])
                if isinstance(skills_gap, list) and skills_gap:
                    for skill in skills_gap:
                        st.write(f"- {skill}")
                else:
                    st.write("No skills gap identified" if isinstance(skills_gap, list) else skills_gap)
                
                st.markdown("### Recruiter Questions")
                recruiter_questions = result.get('recruiter_questions', [])
                if isinstance(recruiter_questions, list) and recruiter_questions:
                    for question in recruiter_questions:
                        st.write(f"- {question}")
                else:
                    st.write("No recruiter questions available" if isinstance(recruiter_questions, list) else recruiter_questions)

                st.markdown("### Strengths")
                strengths = result.get('strengths', [])
                if strengths:
                    for strength in strengths:
                        st.write(f"- {strength}")
                else:
                    st.write("No specific strengths identified")

                st.markdown("### Areas for Improvement")
                areas_for_improvement = result.get('areas_for_improvement', [])
                if areas_for_improvement:
                    for area in areas_for_improvement:
                        st.write(f"- {area}")
                else:
                    st.write("No specific areas for improvement identified")

            # Feedback form with unique key for each result
            with st.form(key=f'feedback_form_{run_id}_{i}'):
                st.subheader("Provide Feedback")
                accuracy_rating = st.select_slider("Accuracy of the evaluation:", options=[1, 2, 3, 4, 5])
                content_rating = st.select_slider("Quality of the report content:", options=[1, 2, 3, 4, 5])
                suggestions = st.text_area("Please provide any suggestions for improvement:")
                submit_feedback = st.form_submit_button("Submit Feedback")

                if submit_feedback:
                    if save_feedback_func(run_id, result['file_name'], accuracy_rating, content_rating, suggestions):
                        st.success("Thank you for your feedback!")
                    else:
                        st.error("Failed to save feedback. Please try again.")

    # Add a progress bar for the analysis process
    progress_bar = st.progress(0)
    for i, result in enumerate(evaluation_results):
        progress_bar.progress((i + 1) / len(evaluation_results))

def is_valid_fractal_job_link(url):
    # Pattern to match Fractal's Workday job posting URLs
    pattern = r'^https?://fractal\.wd1\.myworkdayjobs\.com/.*Careers/.*'
    return re.match(pattern, url) is not None

def extract_job_description(url):
    if not is_valid_fractal_job_link(url):
        raise ValueError("Invalid job link. Please use a link from Fractal's career site.")
    
    options = Options()
    options.add_argument("--headless")
    
    try:
        driver = webdriver.Chrome(options=options)
        
        st.write("Opening the job description URL...")
        driver.get(url)
        
        st.write("Waiting for the page to load completely...")
        WebDriverWait(driver, 20).until(lambda d: d.execute_script('return document.readyState') == 'complete')
        
        st.write("Page loaded. Capturing screenshot...")
        driver.save_screenshot("/app/screenshot.png")
        
        st.write("Extracting page source for debugging...")
        page_source = driver.page_source
        print(page_source)
        
        st.write("Waiting for the job description element to be located...")
        wait = WebDriverWait(driver, 20)
        job_description_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-automation-id='jobPostingDescription']")))
        
        st.write("Job description element found. Extracting text...")
        job_description = job_description_element.text
        
        st.write("Job description extraction successful.")
        return job_description
    except Exception as e:
        error_message = f"Failed to extract job description: {str(e)}"
        st.error(error_message)
        print(error_message)
        return None
    finally:
        st.write("Closing the browser...")
        if 'driver' in locals():
            driver.quit()

def get_available_api_keys() -> Dict[str, str]:
    api_keys = {}
    backend = "llama"
    key = os.getenv(f'{backend.upper()}_API_KEY')
    if key:
        api_keys[backend] = key
    return api_keys

def clear_cache():
    if 'resume_processor' in st.session_state and hasattr(st.session_state.resume_processor, 'clear_cache'):
        st.session_state.resume_processor.clear_cache()
        logger = get_logger(__name__)
        logger.debug(f"Cleared resume analysis cache for backend: {st.session_state.backend}")
    else:
        logger = get_logger(__name__)
        logger.warning("Unable to clear cache: resume_processor not found or doesn't have clear_cache method")