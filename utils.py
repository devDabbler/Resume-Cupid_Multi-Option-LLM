import re
import logging
from logger import get_logger
import io
import PyPDF2
from docx import Document
import numpy as np
import nltk
import ssl
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
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import io

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

@st.cache_data
def extract_text_from_pdf(file_content: bytes) -> str:
    logger.debug("Extracting text from PDF...")
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            logger.debug(f"Processing page {page_num + 1}")
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            logger.debug(f"Extracted text from page {page_num + 1}: {page_text[:100]}...")  # Log first 100 chars
            text += page_text

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

def extract_text_from_file(file) -> str:
    logger.debug(f"Extracting text from file: {file.name}")
    
    # Check if the file object is empty
    if file is None:
        raise ValueError(f"File object is None for {file.name}")
    
    # Reset file pointer to the beginning
    file.seek(0)
    
    # Read file content
    file_content = file.read()
    
    # Log file content details
    logger.debug(f"File content type: {type(file_content)}")
    logger.debug(f"File content length: {len(file_content)} bytes")
    
    if len(file_content) == 0:
        raise ValueError(f"File {file.name} is empty (0 bytes)")
    
    file_extension = file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'pdf':
            text = extract_text_from_pdf(file_content)
        elif file_extension in ['docx', 'doc']:
            text = extract_text_from_docx(file_content)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        if not text.strip():
            raise ValueError(f"Extracted text is empty for file: {file.name}")
        
        logger.debug(f"Successfully extracted {len(text)} characters from {file.name}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from file {file.name}: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to extract text from {file.name}: {str(e)}")

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

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from io import BytesIO

def generate_pdf_report(evaluation_results: List[Dict[str, Any]], run_id: str) -> bytes:
    # Implement this function to generate the PDF report
    # For now, we'll use a placeholder
    buffer = BytesIO()
    # PDF generation code goes here
    buffer.seek(0)
    return buffer.getvalue()

def display_results(evaluation_results: List[Dict[str, Any]], run_id: str, save_feedback_func):
    st.header("Stack Ranking of Candidates")
    
    # Sort results by match score in descending order
    sorted_results = sorted(evaluation_results, key=lambda x: x['match_score'], reverse=True)

    df = pd.DataFrame(sorted_results)
    df['Rank'] = range(1, len(df) + 1)
    df = df[['Rank', 'file_name', 'match_score', 'recommendation']]
    df.columns = ['Rank', 'Candidate', 'Match Score (%)', 'Recommendation']
    
    def color_scale(val):
        if val < 30:
            color = 'red'
        elif val < 50:
            color = 'orange'
        elif val < 70:
            color = 'yellow'
        elif val < 85:
            color = 'lightgreen'
        else:
            color = 'green'
        return f'background-color: {color}'
    
    st.dataframe(df.style.format({'Match Score (%)': '{:.0f}'}).applymap(color_scale, subset=['Match Score (%)']))

    st.download_button(
        label="Download PDF Report",
        data=generate_pdf_report(evaluation_results, run_id),
        file_name=f"evaluation_report_{run_id}.pdf",
        mime="application/pdf"
    )

    for i, result in enumerate(sorted_results, 1):
        with st.expander(f"Rank {i}: {result['file_name']} - Detailed Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Match Score", f"{result['match_score']}%")
            with col2:
                st.info(f"Recommendation: {result['recommendation']}")

            st.subheader("Brief Summary")
            st.write(result['brief_summary'])

            st.subheader("Experience and Project Relevance")
            if isinstance(result['experience_and_project_relevance'], dict):
                for key, value in result['experience_and_project_relevance'].items():
                    if isinstance(value, dict):
                        st.write(f"**{key.replace('_', ' ').title()}:**")
                        for sub_key, sub_value in value.items():
                            st.write(f"- {sub_key.replace('_', ' ').title()}: {sub_value}")
                    else:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.write(result['experience_and_project_relevance'])

            st.subheader("Skills Gap")
            if isinstance(result['skills_gap'], dict):
                for category, skills in result['skills_gap'].items():
                    st.write(f"**{category.replace('_', ' ').title()}:**")
                    if isinstance(skills, dict):
                        for skill_type, skill_list in skills.items():
                            st.write(f"- {skill_type.replace('_', ' ').title()}:")
                            for skill in skill_list:
                                st.write(f"  - {skill}")
                    else:
                        for skill in skills:
                            st.write(f"- {skill}")
            else:
                st.write(result['skills_gap'])

            st.subheader("Key Strengths")
            for strength in result.get('key_strengths', []):
                st.write(f"- {strength}")

            st.subheader("Areas for Improvement")
            for weakness in result.get('key_weaknesses', []):
                st.write(f"- {weakness}")

            st.subheader("Recruiter Questions")
            for question in result['recruiter_questions']:
                st.write(f"- {question}")

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

    st.progress(100)  # Show completion of analysis
    
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

def process_resume(resume_file, resume_processor, job_description, importance_factors, candidate_data, job_title, key_skills, llm):
    logger = get_logger(__name__)
    logger.debug(f"Processing resume: {resume_file.name} with {resume_processor.backend} backend")
    try:
        resume_text = extract_text_from_file(resume_file)
        if not resume_text.strip():
            logger.warning(f"Empty content extracted from {resume_file.name}")
            return _generate_error_result(resume_file.name, "Empty content extracted")
        
        result = resume_processor.analyze_match(resume_text, job_description, candidate_data, job_title)
        
        # Format experience and project relevance
        exp_relevance = result.get('experience_and_project_relevance', {})
        formatted_exp_relevance = "\n".join([f"{k.replace('_', ' ').title()}: {v}%" for k, v in exp_relevance.items()])
        
        # Format skills gap
        skills_gap = result.get('skills_gap', {})
        formatted_skills_gap = "\n".join([f"{k.replace('_', ' ').title()}: {v}%" for k, v in skills_gap.items()])
        
        # Extract key strengths and areas for improvement
        key_strengths = _extract_key_points(resume_text, job_description, key_skills, is_strength=True)
        areas_for_improvement = _extract_key_points(resume_text, job_description, key_skills, is_strength=False)
        
        processed_result = {
            'file_name': resume_file.name,
            'brief_summary': result.get('brief_summary', 'No summary available'),
            'match_score': result.get('match_score', 0),
            'experience_and_project_relevance': formatted_exp_relevance,
            'skills_gap': formatted_skills_gap,
            'key_strengths': key_strengths,
            'areas_for_improvement': areas_for_improvement,
            'recruiter_questions': result.get('recruiter_questions', []),
            'recommendation': result.get('recommendation', 'No recommendation available')
        }
        
        logger.debug(f"Processed result: {processed_result}")
        return processed_result
    except Exception as e:
        logger.error(f"Error processing resume {resume_file.name}: {str(e)}", exc_info=True)
        return _generate_error_result(resume_file.name, str(e))

def process_experience_relevance(experience_data):
    try:
        if isinstance(experience_data, list):
            return "\n".join([f"{exp['project']}: {exp['description']} (Relevance: {exp['relevance']}%)" for exp in experience_data])
        return str(experience_data)
    except Exception as e:
        logger.error(f"Error processing experience relevance: {str(e)}")
        return "Unable to process experience relevance data"

def _process_skills_gap(skills_gap):
    try:
        if isinstance(skills_gap, list):
            if skills_gap and isinstance(skills_gap[0], dict):
                return "\n".join([f"{skill['skill']}: {skill['description']} (Importance: {skill['importance']}%)" for skill in skills_gap])
            else:
                return ", ".join(skills_gap)
        return str(skills_gap)
    except Exception as e:
        logger.error(f"Error processing skills gap: {str(e)}")
        return "Unable to process skills gap data"

def _process_recruiter_questions(questions):
    if isinstance(questions, list):
        if questions and isinstance(questions[0], dict):
            return [q['question'] for q in questions[:3]]  # Return top 3 questions
        else:
            return questions[:3]  # Return top 3 questions if they're already strings
    elif isinstance(questions, str):
        return [questions]  # Return the string as a single-item list
    else:
        return ['No recruiter questions generated']

def _generate_brief_summary(original_summary, match_score, recommendation):
    # Extract the candidate's name from the original summary
    name_match = re.search(r'^(\w+(?:\s\w+)?)', original_summary)
    name = name_match.group(1) if name_match else "The candidate"
    
    if match_score < 40:
        return f"{name} does not appear to be a good fit for the ML Ops Engineer role. With a match score of {match_score}%, there are significant gaps in the required skills and experience."
    elif match_score < 55:
        return f"{name} may have some relevant skills, but with a match score of {match_score}%, there are considerable gaps in meeting the requirements for the ML Ops Engineer role. Further evaluation is needed."
    elif match_score < 71:
        return f"{name} shows potential for the ML Ops Engineer role with a match score of {match_score}%. While there are some areas that need improvement, the candidate has relevant skills and experience."
    elif match_score < 82:
        return f"{name} is a strong candidate for the ML Ops Engineer role, with a match score of {match_score}%. The candidate demonstrates most of the required skills and experience, with only minor gaps."
    else:
        return f"{name} is an excellent fit for the ML Ops Engineer role, with a match score of {match_score}%. The candidate exceeds expectations in terms of skills and experience for this position."

def _summarize_relevance(relevance_data):
    if isinstance(relevance_data, dict):
        relevance_score = relevance_data.get('relevance_rating', 'N/A')
        matched_phrases = relevance_data.get('alignment', 'N/A')
        explanation = relevance_data.get('relevance_rating_explanation', 'No explanation provided')
        return f"Relevance Score: {relevance_score}, Alignment: {matched_phrases}, Explanation: {explanation}"
    return str(relevance_data)

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

    # Calculate experience match
    experience_match = _calculate_experience_match(resume_text, job_description)

    # Apply importance factors
    weighted_score = (
        base_score * importance_factors.get('technical_skills', 0.25) +
        relevance_score * importance_factors.get('experience', 0.25) +
        (skills_match * 100) * importance_factors.get('industry_knowledge', 0.25) +
        experience_match * importance_factors.get('years_of_experience', 0.25)
    ) / sum(importance_factors.values() or [1])

    final_score = int(min(max(weighted_score * 0.8, 0), 100))  # Adjust overall score
    return final_score

def _assess_relevance(resume_text: str, job_description: str) -> Dict[str, Any]:
    job_doc = nlp(job_description)
    resume_doc = nlp(resume_text)
    
    job_phrases = [chunk.text.lower() for chunk in job_doc.noun_chunks]
    resume_phrases = [chunk.text.lower() for chunk in resume_doc.noun_chunks]
    
    relevant_experiences = []
    for phrase in job_phrases:
        if phrase in resume_phrases:
            relevant_experiences.append(phrase)
    
    relevance_score = len(relevant_experiences) / len(job_phrases) * 100 if job_phrases else 0
    
    return {
        "relevant_experiences": relevant_experiences,
        "relevance_score": relevance_score,
        "total_job_phrases": len(job_phrases),
        "matched_phrases": len(relevant_experiences)
    }

def _calculate_experience_match(resume_text: str, job_description: str) -> float:
    required_years = re.search(r'(\d+)\+?\s*years?', job_description)
    if required_years:
        required_years = int(required_years.group(1))
        candidate_years = re.search(r'(\d+)\+?\s*years?', resume_text)
        if candidate_years:
            candidate_years = int(candidate_years.group(1))
            return min(candidate_years / required_years, 1.0)
    return 0.5  # Default to middle score if can't determine

def _identify_skills_gap(resume_text: str, job_description: str, key_skills: List[str]) -> List[str]:
    job_doc = nlp(job_description)
    resume_doc = nlp(resume_text)
    
    stop_words = set(nlp.Defaults.stop_words).union({'experience', 'skill', 'ability'})
    
    def is_valid_skill(word):
        return len(word) > 3 and word.lower() not in stop_words and word.isalpha()
    
    specific_skills = [
        "machine learning", "model risk management", "model governance",
        "risk management", "compliance", "mlops", "python", "r", "scala",
        "nlp models", "financial services", "insurance"
    ]
    
    job_skills = set([token.text.lower() for token in job_doc if token.pos_ in ["NOUN", "PROPN"] and is_valid_skill(token.text)] + specific_skills + [skill.lower() for skill in key_skills])
    resume_skills = set([token.text.lower() for token in resume_doc if token.pos_ in ["NOUN", "PROPN"] and is_valid_skill(token.text)])
    
    missing_skills = list(job_skills - resume_skills)
    return sorted(missing_skills, key=lambda x: job_description.lower().count(x), reverse=True)[:10]

# Add this function to download NLTK data
def download_nltk_data():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {str(e)}")

# Call this function at the beginning of your script or in the main function
download_nltk_data()

def _extract_key_points(resume_text, job_description, key_skills, is_strength=True):
    try:
        # Combine resume text and job description
        combined_text = resume_text + " " + job_description
        
        # Create a set of words to ignore
        ignore_words = set(['work', 'learning', 'learn', 'role', 'company', 'business', 'data', 'set', 'machine'])
        try:
            ignore_words.update(nltk.corpus.stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords not available. Proceeding without them.")
        
        # Tokenize and process the text
        words = nltk.word_tokenize(combined_text.lower())
        words = [word for word in words if word.isalnum() and word not in ignore_words]
        
        # Get word frequencies
        freq_dist = nltk.FreqDist(words)
        
        # Filter words based on whether we're looking for strengths or areas for improvement
        if is_strength:
            relevant_words = [word for word in freq_dist.keys() if word in resume_text.lower()]
        else:
            relevant_words = [word for word in freq_dist.keys() if word in job_description.lower() and word not in resume_text.lower()]
        
        # Sort by frequency and return top 5
        sorted_words = sorted(relevant_words, key=freq_dist.get, reverse=True)
        return sorted_words[:5]
    except Exception as e:
        logger.error(f"Error in _extract_key_points: {str(e)}")
        return ["Unable to extract key points"]

def generate_questions(resume_text: str, job_description: str, key_skills: List[str]) -> List[str]:
    skills_gap = _identify_skills_gap(resume_text, job_description, key_skills)
    key_strengths = _extract_key_points(resume_text, job_description, key_skills, is_strength=True)
    
    questions = []
    for skill in skills_gap[:3]:  # Limit to top 3 missing skills
        questions.append(f"Can you tell me about your experience with {skill}?")
    
    for strength in key_strengths[:2]:  # Limit to top 2 strengths
        questions.append(f"Could you elaborate on your experience with {strength}?")
    
    return questions

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

def calculate_strengths(result: dict) -> List[str]:
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

def calculate_areas_for_improvement(result: dict) -> List[str]:
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

# Assuming you have an LLM class or function to interact with your LLM
# Replace `llm.generate` with the actual method you use to interact with your LLM

def extract_key_features(text, llm):
    if not isinstance(text, str):
        text = str(text)
    prompt = f"Extract key features from the following text:\n{text}"
    response = llm.analyze_match(text, prompt, {}, "Feature Extraction")
    return response

def compare_features(features1, features2, llm):
    features1_str = json.dumps(features1) if isinstance(features1, dict) else str(features1)
    features2_str = json.dumps(features2) if isinstance(features2, dict) else str(features2)
    
    prompt = f"Compare the following features and provide a similarity score (0-100):\nFeatures 1: {features1_str}\nFeatures 2: {features2_str}"
    response = llm.analyze_match(features1_str + "\n" + features2_str, prompt, {}, "Feature Comparison")
    similarity_score = int(response.get('match_score', 0))
    return similarity_score

def calculate_combined_score(resume_text, job_description, importance_factors, key_skills, llm):
    if not isinstance(resume_text, str):
        resume_text = str(resume_text)
    if not isinstance(job_description, str):
        job_description = str(job_description)
    
    tfidf_score = calculate_tfidf_score(resume_text, job_description)
    keyword_match = calculate_keyword_match(resume_text, job_description, key_skills)
    
    resume_features = extract_key_features(resume_text, llm)
    job_features = extract_key_features(job_description, llm)
    
    llm_similarity_score = compare_features(resume_features, job_features, llm)
    
    combined_score = (
        tfidf_score * importance_factors.get('technical_skills', 0.5) +
        keyword_match * importance_factors.get('industry_knowledge', 0.5) +
        llm_similarity_score * importance_factors.get('experience', 0.5)
    ) / sum(importance_factors.values() or [1])
    
    return int(min(max(combined_score, 0), 100))

def calculate_tfidf_score(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return cosine_sim

def calculate_keyword_match(resume_text, job_description, key_skills):
    job_skills = set([skill.lower() for skill in key_skills])
    resume_skills = set([word.lower() for word in resume_text.split()])
    matched_skills = job_skills & resume_skills
    return len(matched_skills) / len(job_skills) if job_skills else 0

def dict_to_string(d):
    return '\n'.join(f"{k}: {v}" for k, v in d.items())

def generate_pdf_report(evaluation_results: List[Dict[str, Any]], run_id: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph(f"Evaluation Report (Run ID: {run_id})", styles['Heading1']))

    # Create the data for the summary table
    data = [['Rank', 'Candidate', 'Match Score (%)', 'Recommendation']]
    for i, result in enumerate(evaluation_results, 1):
        data.append([
            i,
            result['file_name'],
            result['match_score'],
            result['recommendation']
        ])

    # Create the summary table
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)

    # Add detailed information for each candidate
    for result in evaluation_results:
        elements.append(Paragraph(f"\n\n{result['file_name']}", styles['Heading2']))
        elements.append(Paragraph(f"Match Score: {result['match_score']}%", styles['Normal']))
        elements.append(Paragraph(f"Recommendation: {result['recommendation']}", styles['Normal']))
        elements.append(Paragraph("Brief Summary:", styles['Heading3']))
        
        # Handle 'brief_summary' whether it's a string or a dictionary
        if isinstance(result['brief_summary'], dict):
            brief_summary = dict_to_string(result['brief_summary'])
        else:
            brief_summary = str(result['brief_summary'])
        elements.append(Paragraph(brief_summary, styles['Normal']))

        # Handle 'experience_and_project_relevance'
        elements.append(Paragraph("Experience and Project Relevance:", styles['Heading3']))
        if isinstance(result['experience_and_project_relevance'], dict):
            exp_relevance = dict_to_string(result['experience_and_project_relevance'])
        else:
            exp_relevance = str(result['experience_and_project_relevance'])
        elements.append(Paragraph(exp_relevance, styles['Normal']))

        # Handle 'skills_gap'
        elements.append(Paragraph("Skills Gap:", styles['Heading3']))
        if isinstance(result['skills_gap'], list):
            skills_gap = '\n'.join(f"- {skill}" for skill in result['skills_gap'])
        elif isinstance(result['skills_gap'], dict):
            skills_gap = dict_to_string(result['skills_gap'])
        else:
            skills_gap = str(result['skills_gap'])
        elements.append(Paragraph(skills_gap, styles['Normal']))

        # Add 'key_strengths' if available
        if 'key_strengths' in result:
            elements.append(Paragraph("Key Strengths:", styles['Heading3']))
            strengths = '\n'.join(f"- {strength}" for strength in result['key_strengths'])
            elements.append(Paragraph(strengths, styles['Normal']))

        # Add 'key_weaknesses' if available
        if 'key_weaknesses' in result:
            elements.append(Paragraph("Areas for Improvement:", styles['Heading3']))
            weaknesses = '\n'.join(f"- {weakness}" for weakness in result['key_weaknesses'])
            elements.append(Paragraph(weaknesses, styles['Normal']))

        # Add 'recruiter_questions'
        elements.append(Paragraph("Recruiter Questions:", styles['Heading3']))
        questions = '\n'.join(f"- {question}" for question in result['recruiter_questions'])
        elements.append(Paragraph(questions, styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()