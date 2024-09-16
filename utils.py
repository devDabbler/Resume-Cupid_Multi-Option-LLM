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
    
    st.dataframe(df.style.applymap(color_scale, subset=['Match Score (%)']))

    for i, result in enumerate(evaluation_results, 1):
        with st.expander(f"Rank {i}: {result['file_name']} - Detailed Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Match Score", f"{result['match_score']}%")
            with col2:
                st.info(f"Recommendation: {result['recommendation']}")

            st.subheader("Brief Summary")
            if isinstance(result['brief_summary'], dict):
                for key, value in result['brief_summary'].items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            elif isinstance(result['brief_summary'], list):
                for item in result['brief_summary']:
                    st.write(f"- {item}")
            else:
                st.write(result['brief_summary'])

            st.subheader("Experience and Project Relevance")
            if isinstance(result['experience_and_project_relevance'], dict):
                for key, value in result['experience_and_project_relevance'].items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            elif isinstance(result['experience_and_project_relevance'], list):
                for item in result['experience_and_project_relevance']:
                    st.write(f"- **{item['project']}:** {item['comments']} (Relevance: {item['relevance']}%)")
            else:
                st.write(result['experience_and_project_relevance'])

            st.subheader("Skills Gap")
            if isinstance(result['skills_gap'], list):
                for skill in result['skills_gap']:
                    st.write(f"- {skill}")
            elif isinstance(result['skills_gap'], dict):
                for key, value in result['skills_gap'].items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.write(result['skills_gap'])

            st.subheader("Key Strengths")
            for strength in result['key_strengths']:
                st.write(f"- {strength}")

            st.subheader("Areas for Improvement")
            for weakness in result['key_weaknesses']:
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

def process_resume(resume_file, _resume_processor, job_description, importance_factors, candidate_data, job_title, key_skills, llm):
    logger = get_logger(__name__)
    logger.debug(f"Processing resume: {resume_file.name} with {_resume_processor.backend} backend")
    try:
        resume_text = extract_text_from_file(resume_file)
        logger.debug(f"Extracted text type: {type(resume_text)}")
        logger.debug(f"Extracted text (first 100 chars): {resume_text[:100]}")
        
        result = _resume_processor.analyze_match(resume_text, job_description, candidate_data, job_title)
        logger.debug(f"Raw analysis result: {result}")
        
        processed_result = {
            'file_name': resume_file.name,
            'brief_summary': result.get('brief_summary', 'No brief summary available'),
            'match_score': result.get('match_score', 0),
            'experience_and_project_relevance': _process_experience_relevance(result.get('experience_and_project_relevance', [])),
            'skills_gap': _process_skills_gap(result.get('skills_gap', [])),
            'key_strengths': _extract_key_points(resume_text, job_description, key_skills, is_strength=True),
            'key_weaknesses': _extract_key_points(resume_text, job_description, key_skills, is_strength=False),
            'recruiter_questions': _process_recruiter_questions(result.get('recruiter_questions', [])),
            'recommendation': result.get('recommendation', 'No recommendation available')
        }
        
        logger.debug(f"Processed result: {processed_result}")
        return processed_result
    except Exception as e:
        logger.error(f"Error processing resume {resume_file.name}: {str(e)}", exc_info=True)
        return _generate_error_result(resume_file.name, str(e))

def _process_experience_relevance(experience_data):
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

    # Apply importance factors (use get() method with default value to avoid KeyError)
    weighted_score = (
        base_score * importance_factors.get('technical_skills', 0.3) +
        relevance_score * importance_factors.get('experience', 0.3) +
        (skills_match * 100) * importance_factors.get('industry_knowledge', 0.4)
    ) / sum(importance_factors.values() or [1])  # Use [1] as fallback if the dict is empty

    # Apply a penalty for lack of specific experience
    experience_penalty = 0.5 if "5+ years" not in resume_text.lower() else 1

    final_score = int(min(max(weighted_score * experience_penalty * 0.5, 0), 100))  # Multiply by 0.5 to further reduce scores
    return final_score

def _assess_relevance(resume_text: str, job_description: str) -> Dict[str, Any]:
    job_doc = nlp(job_description)
    resume_doc = nlp(resume_text)
    
    relevant_terms = [
        "machine learning", "model risk", "model governance", "risk management",
        "compliance", "mlops", "model validation", "model monitoring",
        "documentation", "regulatory requirements", "financial services"
    ]
    
    job_phrases = [chunk.text.lower() for chunk in job_doc.noun_chunks if any(term in chunk.text.lower() for term in relevant_terms)]
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

def _extract_key_points(resume_text: str, job_description: str, key_skills: List[str], is_strength: bool = True) -> List[str]:
    resume_doc = nlp(resume_text)
    job_doc = nlp(job_description)
    
    # Filter out common words and short words
    stop_words = set(nlp.Defaults.stop_words).union({'experience', 'skill', 'ability'})
    
    def is_valid_word(word):
        return len(word) > 3 and word.lower() not in stop_words and word.isalpha()
    
    resume_phrases = [chunk.text.lower() for chunk in resume_doc.noun_chunks if is_valid_word(chunk.root.text)]
    job_phrases = [chunk.text.lower() for chunk in job_doc.noun_chunks if is_valid_word(chunk.root.text)]
    
    if is_strength:
        strengths = list(set([phrase for phrase in resume_phrases if phrase in job_phrases or any(skill.lower() in phrase for skill in key_skills)]))
        return sorted(strengths, key=lambda x: job_description.lower().count(x), reverse=True)[:5]
    else:
        weaknesses = list(set([phrase for phrase in job_phrases if phrase not in resume_phrases and not any(skill.lower() in phrase for skill in key_skills)]))
        return sorted(weaknesses, key=lambda x: job_description.lower().count(x), reverse=True)[:5]

def _generate_questions(resume_text: str, job_description: str, key_skills: List[str]) -> List[str]:
    skills_gap = _identify_skills_gap(resume_text, job_description, key_skills)
    key_strengths = _extract_key_points(resume_text, job_description, key_skills, is_strength=True)
    
    questions = []
    for skill in skills_gap[:3]:  # Limit to top 3 missing skills
        questions.append(f"Can you tell me about your experience with {skill}?")
    
    for strength in key_strengths[:2]:  # Limit to top 2 strengths
        questions.append(f"Could you elaborate on your experience with {strength}?")
    
    return questions

def _get_recommendation(match_score: int) -> str:
    if match_score < 40:
        return "Do not recommend for interview"
    elif 40 <= match_score < 55:
        return "Review profile again and/or conduct indepth recruiter screen focusing on the questions provided. Recommend for interview with significant reservations"
    elif 55 <= match_score < 71:
        return "Recommend for interview with minor reservations - be sure to focus on skill gaps etc."
    elif 71 <= match_score < 82:
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

def generate_pdf_report(evaluation_results: List[Dict[str, Any]], run_id: str) -> bytes:
    buffer = io.BytesIO()
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
    for i, result in enumerate(evaluation_results, 1):
        elements.append(Paragraph(f"\n\nRank {i}: {result['file_name']}", styles['Heading2']))
        elements.append(Paragraph(f"Match Score: {result['match_score']}%", styles['Normal']))
        elements.append(Paragraph(f"Recommendation: {result['recommendation']}", styles['Normal']))
        elements.append(Paragraph("Brief Summary:", styles['Heading3']))
        elements.append(Paragraph(result['brief_summary'], styles['Normal']))
        elements.append(Paragraph("Experience and Project Relevance:", styles['Heading3']))
        elements.append(Paragraph(str(result['experience_and_project_relevance']), styles['Normal']))
        elements.append(Paragraph("Skills Gap:", styles['Heading3']))
        elements.append(Paragraph(", ".join(result['skills_gap']), styles['Normal']))
        elements.append(Paragraph("Recruiter Questions:", styles['Heading3']))
        for question in result['recruiter_questions']:
            elements.append(Paragraph(f"- {question}", styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()