import re
import logging
from logger import get_logger
import io
import PyPDF2
from docx import Document
import numpy as np
import ssl
from collections import Counter
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
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from functools import lru_cache

# Load spaCy model lazily using a singleton pattern
@lru_cache(maxsize=1)
def get_spacy_model():
    return spacy.load("en_core_web_md")

# Create a root logger
logger = get_logger(__name__)

# Initialize logger with environment-based settings
def setup_logger():
    environment = os.getenv('ENVIRONMENT', 'development')
    logging_level = logging.INFO if environment == 'production' else logging.DEBUG
    logging.basicConfig(level=logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging_level)
    return logger

setup_logger()

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

@st.cache_data
def extract_text_from_pdf(file_content: bytes) -> str:
    logger.debug("Extracting text from PDF...")
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = "".join([page.extract_text() for page in pdf_reader.pages])
        
        if not text.strip():
            logger.warning("Extracted PDF text is empty or contains only whitespace.")
        else:
            logger.debug(f"Successfully extracted {len(text)} characters from PDF")
        
        return text
    except PyPDF2.errors.PdfReadError as e:
        logger.error(f"PyPDF2 error reading PDF: {str(e)}")
        raise ValueError(f"Failed to read PDF: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error extracting text from PDF: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

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

def extract_text_from_file(file) -> str:
    logger.debug(f"Extracting text from file: {file.name}")
    
    if file is None:
        raise ValueError("File object is None.")
    
    file.seek(0)
    file_content = file.read()

    if len(file_content) == 0:
        raise ValueError(f"File {file.name} is empty (0 bytes)")

    file_extension = file.name.split('.')[-1].lower()

    try:
        if file_extension == 'pdf':
            return extract_text_from_pdf(file_content)
        elif file_extension in ['docx', 'doc']:
            return extract_text_from_docx(file_content)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        logger.error(f"Error extracting text from file {file.name}: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to extract text from {file.name}: {str(e)}")

def preprocess_text(text: str) -> str:
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', cleaned_text).strip().lower()

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    try:
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return similarity
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        raise ValueError(f"Error calculating similarity: {str(e)}")

# Async batch processing of resumes
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

def process_resume(resume_file, resume_processor, job_description, importance_factors, candidate_data, job_title, key_skills, llm):
    logger = get_logger(__name__)
    logger.debug(f"Processing resume: {resume_file.name} with {resume_processor.backend} backend")
    try:
        resume_text = extract_text_from_file(resume_file)
        if not resume_text.strip():
            logger.warning(f"Empty content extracted from {resume_file.name}")
            return _generate_error_result(resume_file.name, "Empty content extracted")
        
        result = resume_processor.analyze_match(resume_text, job_description, candidate_data, job_title)
        
        # Adjust match score based on importance factors
        adjusted_score = adjust_match_score(result['match_score'], result, importance_factors)
        
        # Format experience and project relevance
        exp_relevance = result.get('experience_and_project_relevance', {})
        if isinstance(exp_relevance, dict):
            formatted_exp_relevance = "\n".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in exp_relevance.items()])
        elif isinstance(exp_relevance, list):
            formatted_exp_relevance = "\n".join([str(item) for item in exp_relevance])
        else:
            formatted_exp_relevance = str(exp_relevance)
        
        # Format skills gap
        skills_gap = result.get('skills_gap', {})
        if isinstance(skills_gap, dict):
            formatted_skills_gap = "\n".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in skills_gap.items()])
        elif isinstance(skills_gap, list):
            formatted_skills_gap = "\n".join([str(item) for item in skills_gap])
        else:
            formatted_skills_gap = str(skills_gap)
        
        # Extract key strengths and areas for improvement using LLM
        strengths_and_improvements = get_strengths_and_improvements(resume_text, job_description, llm)
        
        # Format recruiter questions
        formatted_questions = []
        for q in result.get('recruiter_questions', []):
            if isinstance(q, dict):
                formatted_questions.append(q.get('question', ''))
            else:
                formatted_questions.append(q)
        
        processed_result = {
            'file_name': resume_file.name,
            'brief_summary': result.get('brief_summary', 'No summary available'),
            'match_score': adjusted_score,
            'experience_and_project_relevance': formatted_exp_relevance,
            'skills_gap': formatted_skills_gap,
            'key_strengths': strengths_and_improvements['strengths'],
            'areas_for_improvement': strengths_and_improvements['improvements'],
            'recruiter_questions': formatted_questions,
            'recommendation': get_recommendation(adjusted_score)
        }
        
        logger.debug(f"Processed result: {processed_result}")
        return processed_result
    except Exception as e:
        logger.error(f"Error processing resume {resume_file.name}: {str(e)}", exc_info=True)
        return _generate_error_result(resume_file.name, str(e))
    
def generate_pdf_report(evaluation_results: List[Dict[str, Any]], run_id: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph(f"Evaluation Report (Run ID: {run_id})", styles['Heading1']))

    # Create summary table
    data = [['Rank', 'Candidate', 'Match Score (%)', 'Recommendation']]
    for i, result in enumerate(evaluation_results, 1):
        data.append([
            i,
            result['file_name'],
            result['match_score'],
            result['recommendation']
        ])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)

    for result in evaluation_results:
        elements.append(Paragraph(f"\n\n{result['file_name']}", styles['Heading2']))
        elements.append(Paragraph(f"Match Score: {result['match_score']}%", styles['Normal']))
        elements.append(Paragraph(f"Recommendation: {result['recommendation']}", styles['Normal']))

        elements.append(Paragraph("Brief Summary:", styles['Heading3']))
        brief_summary = dict_to_string(result['brief_summary']) if isinstance(result['brief_summary'], dict) else str(result['brief_summary'])
        elements.append(Paragraph(brief_summary, styles['Normal']))

        elements.append(Paragraph("Experience and Project Relevance:", styles['Heading3']))
        exp_relevance = dict_to_string(result['experience_and_project_relevance']) if isinstance(result['experience_and_project_relevance'], dict) else str(result['experience_and_project_relevance'])
        elements.append(Paragraph(exp_relevance, styles['Normal']))

        elements.append(Paragraph("Skills Gap:", styles['Heading3']))
        skills_gap = dict_to_string(result['skills_gap']) if isinstance(result['skills_gap'], dict) else str(result['skills_gap'])
        elements.append(Paragraph(skills_gap, styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# Utility Functions
def dict_to_string(d):
    return '\n'.join(f"{k}: {v}" for k, v in d.items())

def download_nltk_data():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {str(e)}")

download_nltk_data()

def clear_cache():
    if 'resume_processor' in st.session_state and hasattr(st.session_state.resume_processor, 'clear_cache'):
        st.session_state.resume_processor.clear_cache()
        logger.debug(f"Cleared resume analysis cache for backend: {st.session_state.backend}")
    else:
        logger.warning("Unable to clear cache: resume_processor not found or doesn't have clear_cache method")

def _generate_error_result(file_name: str, error_message: str) -> Dict[str, Any]:
    return {
        'file_name': file_name,
        'brief_summary': f"Error occurred during analysis: {error_message}",
        'match_score': 0,
        'recommendation': 'Unable to provide a recommendation due to an error',
        'experience_and_project_relevance': 'Unable to assess due to an error',
        'skills_gap': 'Unable to determine skills gap due to an error',
        'key_strengths': 'Unable to identify key strengths due to an error',
        'key_weaknesses': 'Unable to identify key weaknesses due to an error',
        'recruiter_questions': ['Unable to generate recruiter questions due to an error']
    }

# Handle validation for Fractal job links
def is_valid_fractal_job_link(url):
    pattern = r'^https?://fractal\.wd1\.myworkdayjobs\.com/.*Careers/.*'
    return re.match(pattern, url) is not None

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
        
def adjust_match_score(original_score, result, importance_factors, job_requirements):
    skills_weight = importance_factors.get('technical_skills', 0.3)
    experience_weight = importance_factors.get('experience', 0.3)
    education_weight = importance_factors.get('education', 0.2)
    industry_weight = importance_factors.get('industry_knowledge', 0.2)
    
    skills_score = evaluate_skills(result.get('skills_gap', {}), job_requirements.get('required_skills', []))
    experience_score = evaluate_experience(result.get('experience_and_project_relevance', {}), job_requirements.get('years_of_experience', 0))
    education_score = evaluate_education(result, job_requirements.get('education_level', 'Bachelor'))
    industry_score = evaluate_industry_knowledge(result, job_requirements.get('industry_keywords', []))
    
    adjusted_score = (
        original_score * 0.4 +  # Base score
        skills_score * skills_weight +
        experience_score * experience_weight +
        education_score * education_weight +
        industry_score * industry_weight
    )
    
    return min(max(int(adjusted_score), 0), 100)  # Ensure score is between 0 and 100

def get_strengths_and_improvements(resume_text, job_description, llm):
    prompt = f"""
    Analyze the following resume and job description, then provide a structured summary of the candidate's key strengths and areas for improvement. Focus on the most impactful points relevant to the job.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Provide a JSON object with the following structure:
    {{
        "strengths": [
            {{ "category": "Technical Skills", "points": ["...", "..."] }},
            {{ "category": "Experience", "points": ["...", "..."] }},
            {{ "category": "Education", "points": ["...", "..."] }}
        ],
        "improvements": [
            {{ "category": "Skills Gap", "points": ["...", "..."] }},
            {{ "category": "Experience", "points": ["...", "..."] }},
            {{ "category": "Industry Knowledge", "points": ["...", "..."] }}
        ]
    }}

    Each category should have 2-3 concise points (1-2 sentences each).
    """
    
    response = llm.analyze_match(resume_text, prompt, {}, "Structured Strengths and Improvements Analysis")
    
    try:
        strengths_and_improvements = json.loads(response['brief_summary'])
        return strengths_and_improvements
    except:
        return {
            'strengths': [{'category': 'General', 'points': ['Unable to extract key strengths']}],
            'improvements': [{'category': 'General', 'points': ['Unable to extract areas for improvement']}]
        }

def adjust_match_score(original_score, result, importance_factors, job_requirements):
    skills_weight = importance_factors.get('technical_skills', 0.3)
    experience_weight = importance_factors.get('experience', 0.3)
    education_weight = importance_factors.get('education', 0.2)
    industry_weight = importance_factors.get('industry_knowledge', 0.2)
    
    skills_score = evaluate_skills(result.get('skills_gap', {}), job_requirements.get('required_skills', []))
    experience_score = evaluate_experience(result.get('experience_and_project_relevance', {}), job_requirements.get('years_of_experience', 0))
    education_score = evaluate_education(result, job_requirements.get('education_level', 'Bachelor'))
    industry_score = evaluate_industry_knowledge(result, job_requirements.get('industry_keywords', []))
    
    adjusted_score = (
        original_score * 0.4 +  # Base score
        skills_score * skills_weight +
        experience_score * experience_weight +
        education_score * education_weight +
        industry_score * industry_weight
    )
    
    return min(max(int(adjusted_score), 0), 100)  # Ensure score is between 0 and 100

def evaluate_skills(skills_gap, required_skills):
    total_score = 0
    for skill, proficiency in skills_gap.items():
        if skill in required_skills:
            if proficiency == 'Expert':
                total_score += 10
            elif proficiency == 'Intermediate':
                total_score += 5
            elif proficiency == 'Beginner':
                total_score += 2
    
    max_possible_score = len(required_skills) * 10
    return (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0

def evaluate_experience(experience_relevance, years_required):
    total_years = sum(years for _, years in experience_relevance.items())
    relevant_years = sum(years for relevance, years in experience_relevance.items() if relevance >= 0.7)
    
    years_factor = min(total_years / years_required, 1) if years_required > 0 else 1
    relevance_factor = relevant_years / total_years if total_years > 0 else 0
    
    return (years_factor * 0.6 + relevance_factor * 0.4) * 100

def evaluate_education(result, required_level):
    education_levels = ['High School', 'Associate', 'Bachelor', 'Master', 'PhD']
    candidate_level = result.get('highest_education_level', 'High School')
    candidate_field = result.get('field_of_study', '')
    
    level_score = education_levels.index(candidate_level) / (len(education_levels) - 1)
    field_score = 1 if candidate_field.lower() in ['computer science', 'data science'] else 0.5
    
    return (level_score * 0.7 + field_score * 0.3) * 100

def evaluate_industry_knowledge(result, industry_keywords):
    keyword_count = sum(1 for word in result.get('resume_text', '').lower().split() if word in industry_keywords)
    total_words = len(result.get('resume_text', '').split())
    
    keyword_density = keyword_count / total_words if total_words > 0 else 0
    return min(keyword_density * 200, 100)  # Cap at 100%

def get_strengths_and_improvements(resume_text, job_description, llm):
    prompt = f"""
    Analyze the following resume and job description, then provide a structured summary of the candidate's key strengths and areas for improvement. Focus on the most impactful points relevant to the job.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Provide a JSON object with the following structure:
    {{
        "strengths": [
            {{ "category": "Technical Skills", "points": ["...", "..."] }},
            {{ "category": "Experience", "points": ["...", "..."] }},
            {{ "category": "Education", "points": ["...", "..."] }}
        ],
        "improvements": [
            {{ "category": "Skills Gap", "points": ["...", "..."] }},
            {{ "category": "Experience", "points": ["...", "..."] }},
            {{ "category": "Industry Knowledge", "points": ["...", "..."] }}
        ]
    }}

    Each category should have 2-3 concise points (1-2 sentences each).
    """
    
    response = llm.analyze_match(resume_text, prompt, {}, "Structured Strengths and Improvements Analysis")
    
    try:
        strengths_and_improvements = json.loads(response['brief_summary'])
        return strengths_and_improvements
    except:
        return {
            'strengths': [{'category': 'General', 'points': ['Unable to extract key strengths']}],
            'improvements': [{'category': 'General', 'points': ['Unable to extract areas for improvement']}]
        }

def get_recommendation(match_score: int) -> str:
    if match_score < 30:
        return "Do not recommend for interview (not suitable for the role)"
    elif 30 <= match_score < 50:
        return "Do not recommend for interview (significant skill gaps)"
    elif 50 <= match_score < 65:
        return "Consider for interview only if making a career transition; review with a lead"
    elif 65 <= match_score < 80:
        return "Recommend for interview with minor reservations"
    elif 80 <= match_score < 90:
        return "Recommend for interview"
    else:
        return "Highly recommend for interview"