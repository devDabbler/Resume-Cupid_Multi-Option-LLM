import re
import logging
from logger import get_logger
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
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from streamlit.runtime.scriptrunner import add_script_run_ctx
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Create a root logger
root_logger = logging.getLogger(__name__)
logger = get_logger(__name__)

# Initialize logger with environment-based settings
def setup_logger():
    environment = os.getenv('ENVIRONMENT', 'development')
    logging_level = logging.INFO if environment == 'production' else logging.DEBUG
    logging.basicConfig(level=logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging_level)
    return logger

logger = setup_logger()

# Set up the root_logger
root_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# Custom exception handler
def exception_handler(exc_type, exc_value, exc_traceback):
    logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
    root_logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

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
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Error retrieving result from future: {str(e)}", exc_info=True)
    
    return results

def process_resumes_sequentially(resume_files, resume_processor, job_description, importance_factors, candidate_data_list, job_title):
    logger.debug("Starting sequential resume processing...")
    results = []
    for file, candidate_data in zip(resume_files, candidate_data_list):
        try:
            resume_text = extract_text_from_file(file)
            result = resume_processor.analyze_match(resume_text, job_description, candidate_data, job_title)
            result['file_name'] = file.name
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing resume {file.name}: {str(e)}", exc_info=True)
            results.append(_generate_error_result(file.name, str(e)))
    return results

def display_results(evaluation_results: List[Dict[str, Any]], run_id: str, save_feedback_func):
    st.header("Stack Ranking of Candidates")
    
    df = pd.DataFrame(evaluation_results)
    df['Rank'] = range(1, len(df) + 1)
    df = df[['Rank', 'file_name', 'match_score', 'recommendation']]
    df.columns = ['Rank', 'Candidate', 'Match Score (%)', 'Recommendation']
    
    # Custom color scale function
    def color_scale(val):
        if val <= 39:
            return 'background-color: #FFCCCB'  # Light red
        elif val <= 54:
            return 'background-color: #FFFF99'  # Light yellow
        elif val <= 70:
            return 'background-color: #FFFFCC'  # Pale yellow
        elif val <= 81:
            return 'background-color: #CCFFCC'  # Light green
        else:
            return 'background-color: #90EE90'  # Pale green

    st.dataframe(df.style.applymap(color_scale, subset=['Match Score (%)']))

    for i, result in enumerate(evaluation_results, 1):
        with st.expander(f"Rank {i}: {result['file_name']} - Detailed Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Match Score", f"{result['match_score']}%")
            with col2:
                st.info(f"Recommendation: {result['recommendation']}")

            st.subheader("Brief Summary")
            st.write(result['brief_summary'])

            st.subheader("Experience and Project Relevance")
            display_nested_content(result['experience_and_project_relevance'])

            st.subheader("Skills Gap")
            display_nested_content(result['skills_gap'])

            st.subheader("Recruiter Questions")
            display_nested_content(result['recruiter_questions'])

            with st.form(key=f'feedback_form_{run_id}_{i}'):
                st.subheader("Provide Feedback")
                accuracy_rating = st.slider("Accuracy of the evaluation:", 1, 5, 3)
                content_rating = st.slider("Quality of the report content:", 1, 5, 3)
                suggestions = st.text_area("Please provide any suggestions for improvement:")
                submit_feedback = st.form_submit_button("Submit Feedback")

                if submit_feedback:
                    if save_feedback_func(run_id, result['file_name'], accuracy_rating, content_rating, suggestions):
                        st.success("Thank you for your feedback!")
                    else:
                        st.error("Failed to save feedback. Please try again.")

    st.progress(1.0)  # Show completion of analysis

def display_nested_content(content):
    if isinstance(content, dict):
        for key, value in content.items():
            st.write(f"**{key.capitalize()}:**")
            display_nested_content(value)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                display_nested_content(item)
            else:
                st.write(f"- {item}")
    elif isinstance(content, (int, float)):
        st.write(f"{content}")
    else:
        st.write(content)

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

# Function to get available API keys
def get_available_api_keys() -> Dict[str, str]:
    api_keys = {}
    backend = "llama"
    key = os.getenv(f'{backend.upper()}_API_KEY')
    if key:
        api_keys[backend] = key
    return api_keys

# Function to clear cache
def clear_cache():
    if 'resume_processor' in st.session_state and hasattr(st.session_state.resume_processor, 'clear_cache'):
        st.session_state.resume_processor.clear_cache()
        logger.debug(f"Cleared resume analysis cache for backend: {st.session_state.backend}")
    else:
        logger.warning("Unable to clear cache: resume_processor not found or doesn't have clear_cache method")

def process_resume(resume_file, _resume_processor, job_description, importance_factors, candidate_data, job_title, key_skills):
    logger = get_logger(__name__)
    logger.debug(f"Processing resume: {resume_file.name} with {_resume_processor.backend} backend")
    try:
        resume_text = extract_text_from_file(resume_file)
        result = _resume_processor.analyze_match(resume_text, job_description, candidate_data, job_title)
        
        processed_result = {
            'file_name': resume_file.name,
            'brief_summary': result.get('brief_summary', 'No brief summary available'),
            'match_score': _calculate_match_score(resume_text, job_description, importance_factors, key_skills),
            'experience_and_project_relevance': _assess_relevance(resume_text, job_description),
            'skills_gap': _identify_skills_gap(resume_text, job_description, key_skills),
            'key_strengths': _extract_key_points(resume_text, job_description, key_skills, is_strength=True),
            'key_weaknesses': _extract_key_points(resume_text, job_description, key_skills, is_strength=False),
            'recruiter_questions': _generate_questions(resume_text, job_description, key_skills)
        }
        
        processed_result['recommendation'] = _get_recommendation(processed_result['match_score'])
        
        return processed_result
    except Exception as e:
        logger.error(f"Error processing resume {resume_file.name}: {str(e)}", exc_info=True)
        return _generate_error_result(resume_file.name, str(e))

def _calculate_match_score(resume_text: str, job_description: str, importance_factors: Dict[str, float], key_skills: List[str]) -> int:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    base_score = cosine_sim * 100

    # Calculate relevance score
    relevance_data = _assess_relevance(resume_text, job_description)
    relevance_score = relevance_data["relevance_score"]

    # Calculate skills match
    skills_match = sum(1 for skill in key_skills if skill.lower() in resume_text.lower()) / len(key_skills) if key_skills else 0

    # Apply importance factors
    weighted_score = (
        base_score * importance_factors['technical_skills'] +
        relevance_score * importance_factors['experience'] +
        skills_match * 100 * importance_factors['industry_knowledge']
    ) / sum(importance_factors.values())

    return int(min(max(weighted_score, 0), 100))

def _assess_relevance(resume_text: str, job_description: str) -> Dict[str, Any]:
    job_doc = nlp(job_description)
    resume_doc = nlp(resume_text)
    
    job_phrases = [chunk.text.lower() for chunk in job_doc.noun_chunks]
    resume_phrases = [chunk.text.lower() for chunk in resume_doc.noun_chunks]
    
    relevant_experiences = []
    for phrase in job_phrases:
        if phrase in resume_phrases:
            relevant_experiences.append(phrase)
    
    relevance_score = len(relevant_experiences) / len(job_phrases) * 100
    
    return {
        "relevant_experiences": relevant_experiences,
        "relevance_score": relevance_score,
        "total_job_phrases": len(job_phrases),
        "matched_phrases": len(relevant_experiences)
    }

def _identify_skills_gap(resume_text: str, job_description: str, key_skills: List[str]) -> List[str]:
    job_doc = nlp(job_description)
    job_skills = set([token.text.lower() for token in job_doc if token.pos_ in ["NOUN", "PROPN"]] + key_skills)
    
    resume_doc = nlp(resume_text)
    resume_skills = set([token.text.lower() for token in resume_doc if token.pos_ in ["NOUN", "PROPN"]])
    
    missing_skills = list(job_skills - resume_skills)
    
    return missing_skills

def _extract_key_points(resume_text: str, job_description: str, is_strength: bool = True) -> List[str]:
    resume_doc = nlp(resume_text)
    job_doc = nlp(job_description)
    
    resume_phrases = [chunk.text for chunk in resume_doc.noun_chunks]
    job_phrases = [chunk.text for chunk in job_doc.noun_chunks]
    
    if is_strength:
        return [phrase for phrase in resume_phrases if phrase.lower() in job_description.lower()]
    else:
        return [phrase for phrase in job_phrases if phrase.lower() not in resume_text.lower()]

def _generate_questions(resume_text: str, job_description: str) -> List[str]:
    skills_gap = _identify_skills_gap(resume_text, job_description)
    key_strengths = _extract_key_points(resume_text, job_description, is_strength=True)
    
    questions = []
    for skill in skills_gap[:3]:  # Limit to top 3 missing skills
        questions.append(f"Can you tell me about your experience with {skill}?")
    
    for strength in key_strengths[:2]:  # Limit to top 2 strengths
        questions.append(f"Could you elaborate on your experience with {strength}?")
    
    return questions

def _get_recommendation(match_score: int) -> str:
    if match_score < 30:
        return "Do not recommend for interview"
    elif 30 <= match_score < 50:
        return "Consider for interview with significant reservations"
    elif 50 <= match_score < 70:
        return "Recommend for interview with minor reservations"
    elif 70 <= match_score < 85:
        return "Recommend for interview"
    else:
        return "Highly recommend for interview"

def _generate_error_result(file_name: str, error_message: str) -> Dict[str, Any]:
    return {
        'file_name': file_name,
        'brief_summary': f"Error occurred during analysis: {error_message}",
        'match_score': 0,
        'recommendation': 'Unable to provide a recommendation due to an error',
        'experience_and_project_relevance': 'Unable to assess due to an error',
        'skills_gap': 'Unable to determine skills gap due to an error',
        'recruiter_questions': ['Unable to generate recruiter questions due to an error']
    }

def _calculate_strengths(result: dict) -> List[str]:
    strengths = []
    if result['match_score'] >= 80:
        strengths.append("Excellent overall match to job requirements")
    elif result['match_score'] >= 65:
        strengths.append("Good overall match to job requirements")
    if not result['skills_gap']:
        strengths.append("No significant skills gaps identified")
    if 'experience' in str(result['experience_and_project_relevance']).lower():
        strengths.append("Relevant work experience")
    if 'project' in str(result['experience_and_project_relevance']).lower():
        strengths.append("Relevant project experience")
    return strengths

def _calculate_areas_for_improvement(result: dict) -> List[str]:
    areas_for_improvement = []
    if result['skills_gap']:
        areas_for_improvement.append("Address identified skills gaps")
    if result['match_score'] < 65:
        areas_for_improvement.append("Improve overall alignment with job requirements")
    if 'lack' in str(result['experience_and_project_relevance']).lower() or 'missing' in str(result['experience_and_project_relevance']).lower():
        areas_for_improvement.append("Gain more relevant experience")
    return areas_for_improvement

def _clean_content(content):
    if isinstance(content, str):
        content = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', content)
        content = re.sub(r'[^\w\s.,!?:;()\[\]{}\-]', '', content)
    elif isinstance(content, dict):
        return {k: _clean_content(v) for k, v in content.items()}
    elif isinstance(content, list):
        return [_clean_content(item) for item in content]
    return content

def format_nested_content(content, indent=0):
    formatted = ""
    if isinstance(content, dict):
        for key, value in content.items():
            formatted += "  " * indent + f"**{key.capitalize()}:**\n"
            formatted += format_nested_content(value, indent + 1)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                formatted += format_nested_content(item, indent)
            else:
                formatted += "  " * indent + f"- {item}\n"
    else:
        formatted += "  " * indent + f"{content}\n"
    return formatted

def ensure_required_fields(result):
    required_fields = [
        'file_name', 'brief_summary', 'match_score', 'recommendation',
        'experience_and_project_relevance', 'skills_gap', 'recruiter_questions'
    ]
    for field in required_fields:
        if field not in result:
            result[field] = f"No {field.replace('_', ' ')} available"
    return result