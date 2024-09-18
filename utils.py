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
from config_settings import Config
import plotly.graph_objects as go

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

def get_db_connection():
    conn = sqlite3.connect(Config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

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

def generate_job_requirements(job_description):
    # Process the job description with spaCy
    nlp = get_spacy_model()
    doc = nlp(job_description)
    
    # Extract required skills
    required_skills = set()
    for ent in doc.ents:
        if ent.label_ in ["SKILL", "PRODUCT", "ORG"]:
            required_skills.add(ent.text.lower())
    
    # Extract years of experience
    experience_pattern = r'\b(\d+)(?:\+)?\s*(?:years?|yrs?)\b.*?experience'
    experience_matches = re.findall(experience_pattern, job_description, re.IGNORECASE)
    years_of_experience = max(map(int, experience_matches)) if experience_matches else 0
    
    # Extract education level
    education_levels = ["high school", "associate", "bachelor", "master", "phd", "doctorate"]
    education_level = "Not specified"
    for level in education_levels:
        if level in job_description.lower():
            education_level = level.title()
            break
    
    # Extract industry keywords
    industry_keywords = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2]
    industry_keywords = [kw for kw, count in Counter(industry_keywords).most_common(10)]
    
    return {
        'required_skills': list(required_skills),
        'years_of_experience': years_of_experience,
        'education_level': education_level,
        'industry_keywords': industry_keywords
    }

def display_results(evaluation_results: List[Dict[str, Any]], run_id: str, save_feedback_func):
    st.header("Candidate Evaluation Results")

    # Sort results by match score in descending order
    sorted_results = sorted(evaluation_results, key=lambda x: x.get('match_score', 0), reverse=True)

    # Create a summary dataframe
    df = pd.DataFrame(sorted_results)
    df['Rank'] = range(1, len(df) + 1)
    df = df[['Rank', 'file_name', 'match_score', 'recommendation']]
    df.columns = ['Rank', 'Candidate', 'Match Score (%)', 'Recommendation']
    df = df.set_index('Rank')

    # Display summary table
    st.subheader("Candidate Summary")
    st.dataframe(df.style.format({'Match Score (%)': '{:.0f}'}))

    # Display detailed results for each candidate
    for i, result in enumerate(sorted_results, 1):
        with st.expander(f"Rank {i}: {result.get('file_name', 'Unknown')} - Detailed Analysis", expanded=i == 1):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Match Score", f"{result.get('match_score', 0)}%")
            with col2:
                st.write("**Recommendation:**", format_output(result.get('recommendation', 'N/A')))

            st.write("**Fit Summary:**", format_output(result.get('fit_summary', 'No fit summary available.')))

            st.subheader("Experience and Project Relevance")
            st.write(display_nested_content(result.get('experience_and_project_relevance', 'Not provided')))

            st.subheader("Skills Gap")
            st.write(display_nested_content(result.get('skills_gap', 'Not provided')))

            st.subheader("Key Strengths")
            for strength in result.get('key_strengths', []):
                st.write(f"**{strength['category']}:** {', '.join(format_output(strength['points']))}")

            st.subheader("Areas for Improvement")
            for area in result.get('areas_for_improvement', []):
                st.write(f"**{area['category']}:** {', '.join(format_output(area['points']))}")

            st.subheader("Recommended Interview Questions")
            questions = result.get('recruiter_questions', [])
            if isinstance(questions, list) and len(questions) > 0:
                for question in questions:
                    st.write(f"- {format_output(question)}")
            else:
                st.write("No specific questions generated.")

            # Feedback form
            st.subheader("Provide Feedback")
            with st.form(key=f'feedback_form_{run_id}_{i}'):
                accuracy_rating = st.slider("Accuracy of the evaluation:", 1, 5, 3)
                content_rating = st.slider("Quality of the report content:", 1, 5, 3)
                suggestions = st.text_area("Please provide any suggestions for improvement:")
                submit_feedback = st.form_submit_button("Submit Feedback")

                if submit_feedback:
                    if save_feedback_func(run_id, result['file_name'], accuracy_rating, content_rating, suggestions):
                        st.success("Thank you for your feedback!")
                    else:
                        st.error("Failed to save feedback. Please try again.")

    # Add the PDF download button to the Streamlit UI
    try:
        pdf_data = generate_pdf_report(evaluation_results, run_id)

        st.download_button(
            label="Download Detailed PDF Report",
            data=pdf_data,
            file_name=f"evaluation_report_{run_id}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}", exc_info=True)
        st.error("Unable to generate PDF report due to an error. Please try again later.")

    st.success("Evaluation complete!")

# Utility function to format nested data
def display_nested_content(data):
    """
    Simplifies nested structures (dictionaries/lists) into human-readable strings.
    Ensures the output is always a string.
    """
    if isinstance(data, dict):
        return '\n'.join(f"{k.capitalize()}: {v}" for k, v in data.items())
    elif isinstance(data, list):
        # Ensure list is converted to a string of items separated by commas
        return ', '.join(str(item) for item in data)
    else:
        # For any other types (including strings), return as string
        return str(data)

def format_output(content):
    if isinstance(content, str):
        # Capitalize the first letter of each sentence
        sentences = content.split('. ')
        formatted_sentences = [s.capitalize() for s in sentences]
        return '. '.join(formatted_sentences)
    elif isinstance(content, list):
        return [format_output(item) for item in content]
    elif isinstance(content, dict):
        return {k: format_output(v) for k, v in content.items()}
    else:
        return content
        
# Function to get available API keys
def get_available_api_keys() -> Dict[str, str]:
    api_keys = {}
    backend = "llama"
    key = os.getenv(f'{backend.upper()}_API_KEY')
    if key:
        api_keys[backend] = key
    return api_keys

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

def process_resume(resume_file, resume_processor, job_description, importance_factors, candidate_data, job_title, key_skills, llm, job_requirements):
    logger = get_logger(__name__)
    logger.debug(f"Processing resume: {resume_file.name} with {resume_processor.backend} backend")
    try:
        resume_text = extract_text_from_file(resume_file)
        if not resume_text.strip():
            logger.warning(f"Empty content extracted from {resume_file.name}")
            return _generate_error_result(resume_file.name, "Empty content extracted")
        
        result = resume_processor.analyze_match(resume_text, job_description, candidate_data, job_title)
        logger.debug(f"Initial analysis result: {json.dumps(result, indent=2)}")
        
        if not isinstance(result, dict):
            logger.error(f"Unexpected result type: {type(result)}")
            return _generate_error_result(resume_file.name, "Unexpected result type")
        
        # Adjust the match score less aggressively
        original_score = max(1, result.get('match_score', 1))
        adjusted_score = max(1, min(100, int(original_score * 0.98)))  # Reduce by 2% instead of 5%
        
        logger.debug(f"Original score: {original_score}, Adjusted score: {adjusted_score}")
        
        result['match_score'] = adjusted_score
        result['brief_summary'] = generate_brief_summary(adjusted_score, job_title)
        result['fit_summary'] = generate_fit_summary(result)
        
        # Ensure experience_and_project_relevance is not empty
        if not result.get('experience_and_project_relevance') or result['experience_and_project_relevance'] == "Unable to complete analysis due to an error":
            result['experience_and_project_relevance'] = analyze_experience_relevance(resume_text, job_description, llm)
        
        # Ensure skills_gap is not empty and contains actual skill gaps
        if not result.get('skills_gap') or isinstance(result['skills_gap'], str):
            result['skills_gap'] = analyze_skills_gap(resume_text, job_description, key_skills, llm)
        
        strengths_and_improvements = get_strengths_and_improvements(resume_text, job_description, llm)
        
        # Generate recruiter questions if they're missing or empty
        if not result.get('recruiter_questions') or len(result.get('recruiter_questions', [])) == 0:
            result['recruiter_questions'] = generate_specific_questions(resume_text, job_description, job_title, llm)
        
        formatted_questions = format_recruiter_questions(result.get('recruiter_questions', []))
        
        processed_result = {
            'file_name': resume_file.name,
            'brief_summary': result['brief_summary'],
            'fit_summary': result['fit_summary'],
            'match_score': adjusted_score,
            'experience_and_project_relevance': result['experience_and_project_relevance'],
            'skills_gap': result['skills_gap'],
            'key_strengths': strengths_and_improvements['strengths'],
            'areas_for_improvement': strengths_and_improvements['improvements'],
            'recruiter_questions': formatted_questions,
            'recommendation': get_recommendation(adjusted_score)
        }
        
        logger.debug(f"Final processed result: {json.dumps(processed_result, indent=2)}")
        return processed_result
    except Exception as e:
        logger.error(f"Error processing resume {resume_file.name}: {str(e)}", exc_info=True)
        return _generate_error_result(resume_file.name, str(e))

def analyze_experience_relevance(resume_text, job_description, llm):
    prompt = f"""
    Analyze the candidate's experience and project relevance based on the following resume and job description:

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Provide a detailed analysis of how the candidate's experience and projects align with the job requirements. 
    Focus on specific skills, technologies, and achievements mentioned in the resume that are relevant to the job.
    Do not mention any match score or overall fit in this analysis.

    Return the result as a string.
    """
    response = llm.analyze(prompt)
    return response.get('experience_and_project_relevance', "Unable to analyze experience relevance")

def analyze_skills_gap(resume_text, job_description, key_skills, llm):
    prompt = f"""
    Identify the skills gap between the candidate's resume and the job requirements:

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Key Skills Required:
    {', '.join(key_skills)}

    List the important skills or qualifications mentioned in the job description that the candidate lacks or hasn't explicitly demonstrated in their resume. 
    Be sure to cross-check with the resume to avoid listing skills that the candidate has clearly mentioned.
    Return the result as a JSON array of strings.
    """
    response = llm.analyze(prompt)
    skills_gap = response.get('skills_gap', [])
    if isinstance(skills_gap, list):
        return skills_gap
    else:
        return ["Unable to determine specific skills gap"]

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
            {{ "category": "Soft Skills", "points": ["...", "..."] }}
        ],
        "improvements": [
            {{ "category": "Skills Gap", "points": ["...", "..."] }},
            {{ "category": "Experience", "points": ["...", "..."] }},
            {{ "category": "Industry Knowledge", "points": ["...", "..."] }}
        ]
    }}

    Each category should have 2-3 specific points (1-2 sentences each) directly related to the candidate's resume and the job requirements.
    """
    
    try:
        response = llm.analyze(prompt)
        strengths_and_improvements = json.loads(response.get('strengths_and_improvements', '{}'))
    except:
        # Fallback data if analysis fails
        strengths_and_improvements = {
            'strengths': [
                {'category': 'Technical Skills', 'points': ['Candidate possesses relevant technical skills for the role']},
                {'category': 'Experience', 'points': ['Candidate has experience in related fields']},
                {'category': 'Soft Skills', 'points': ['Candidate likely has essential soft skills for the position']}
            ],
            'improvements': [
                {'category': 'Skills Gap', 'points': ['Consider assessing any potential skill gaps during the interview']},
                {'category': 'Experience', 'points': ['Explore depth of experience in specific areas during the interview']},
                {'category': 'Industry Knowledge', 'points': ['Evaluate industry-specific knowledge in the interview process']}
            ]
        }
    
    return strengths_and_improvements

def generate_specific_questions(resume_text, job_description, job_title, llm):
    prompt = f"""
    Generate 5 specific interview questions for a {job_title} role based on the following resume and job description:

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Create questions that explore the candidate's experience, skills, and potential gaps related to this specific role.
    Return the result as a JSON array of strings.
    """
    response = llm.analyze(prompt)
    questions = response.get('recruiter_questions', [])
    if isinstance(questions, list) and len(questions) > 0:
        return questions
    else:
        return generate_generic_questions(job_title)

def format_recruiter_questions(questions):
    if isinstance(questions, list):
        return [q.get('question', q) if isinstance(q, dict) else q for q in questions[:5]]
    elif isinstance(questions, str):
        return [questions]
    else:
        return ["No specific questions generated"]

def generate_brief_summary(score: int, job_title: str) -> str:
    if score < 30:
        return f"The candidate shows limited alignment with the {job_title} role requirements. With a match score of {score}%, there are significant areas for improvement."
    elif 30 <= score < 50:
        return f"The candidate demonstrates some potential for the {job_title} role. With a match score of {score}%, there are areas that require further evaluation."
    elif 50 <= score < 70:
        return f"The candidate shows good potential for the {job_title} role. With a match score of {score}%, they meet many of the key requirements, though some areas may need development."
    elif 70 <= score < 85:
        return f"The candidate is a strong fit for the {job_title} role. With a match score of {score}%, they demonstrate solid alignment with the required skills and experience."
    else:
        return f"The candidate is an excellent match for the {job_title} role. With a match score of {score}%, they exceed expectations in meeting the required skills and experience."

def generate_fit_summary(result: Dict[str, Any]) -> str:
    score = result['match_score']
    if score < 50:
        return "The candidate shows potential but has significant areas for development relative to the job requirements."
    elif 50 <= score < 70:
        return "The candidate is a good fit, meeting many of the job requirements with some areas for growth."
    elif 70 <= score < 85:
        return "The candidate is a strong fit, aligning well with most job requirements and showing potential in others."
    else:
        return "The candidate is an excellent fit, meeting or exceeding most job requirements."

def get_recommendation(match_score: int) -> str:
    if match_score < 30:
        return "Consider for different roles that better match the candidate's current skill set"
    elif 30 <= match_score < 50:
        return "Potential for interview, but prepare to discuss skill gaps and development areas"
    elif 50 <= match_score < 70:
        return "Recommend for interview with some reservations"
    elif 70 <= match_score < 85:
        return "Strongly recommend for interview"
    else:
        return "Highly recommend for interview as a top candidate"

def generate_generic_questions(job_title):
    return [
        f"Can you describe your experience that's most relevant to the {job_title} role?",
        f"What challenges have you faced in previous roles similar to {job_title}, and how did you overcome them?",
        "How do you stay updated with the latest technologies and best practices in your field?",
        "Can you give an example of a complex problem you've solved and the approach you took?",
        "How do you handle high-pressure situations or tight deadlines?"
    ]

def format_nested_structure(data):
    if isinstance(data, dict):
        return "\n".join([f"{k}: {format_nested_structure(v)}" for k, v in data.items()])
    elif isinstance(data, list):
        return "\n".join([f"- {format_nested_structure(item)}" for item in data])
    else:
        return str(data)

def generate_pdf_report(evaluation_results: List[Dict[str, Any]], run_id: str) -> bytes:
    try:
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
                result.get('file_name', 'Unknown'),
                result.get('match_score', 0),
                result.get('recommendation', 'N/A')
            ])

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)

        # Adding details for each result
        for result in evaluation_results:
            elements.append(Paragraph(f"\n\n{result.get('file_name', 'Unknown')}", styles['Heading2']))
            elements.append(Paragraph(f"Match Score: {result.get('match_score', 0)}%", styles['Normal']))
            elements.append(Paragraph(f"Recommendation: {result.get('recommendation', 'N/A')}", styles['Normal']))

            # Add the fit summary for each candidate
            fit_summary = result.get('fit_summary', 'No fit summary available.')
            elements.append(Paragraph(f"Fit Summary: {fit_summary}", styles['Normal']))

            # Experience and project relevance
            elements.append(Paragraph("Experience and Project Relevance:", styles['Heading3']))
            exp_relevance = result.get('experience_and_project_relevance', 'Not provided')
            elements.append(Paragraph(str(exp_relevance), styles['Normal']))

            # Skills gap
            elements.append(Paragraph("Skills Gap:", styles['Heading3']))
            skills_gap = result.get('skills_gap', 'Not provided')
            elements.append(Paragraph(str(skills_gap), styles['Normal']))

            # Recruiter questions
            elements.append(Paragraph("Recommended Interview Questions:", styles['Heading3']))
            recruiter_questions = result.get('recruiter_questions', ['No questions generated'])
            for question in recruiter_questions:
                elements.append(Paragraph(f"• {question}", styles['Normal']))

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}", exc_info=True)
        raise
    
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
    logger.debug(f"Adjusting match score. Original score: {original_score}")
    
    # Weightings
    weight_experience = 0.35
    weight_skills = 0.45
    weight_education = 0.1
    weight_industry = 0.1

    skills_weight = importance_factors.get('technical_skills', weight_skills)
    experience_weight = importance_factors.get('experience', weight_experience)
    education_weight = importance_factors.get('education', weight_education)
    industry_weight = importance_factors.get('industry_knowledge', weight_industry)
    
    try:
        skills_score = evaluate_skills(result.get('skills_gap', []), job_requirements.get('required_skills', []))
        
        # Split experience into professional and academic
        professional_experience = result.get('professional_experience', {})
        academic_experience = result.get('academic_experience', {})
        experience_score = evaluate_experience(professional_experience, academic_experience, job_requirements.get('years_of_experience', 0))
        
        education_score = evaluate_education(result, job_requirements.get('education_level', 'Bachelor'))
        industry_score = evaluate_industry_knowledge(result, job_requirements.get('industry_keywords', []))
    
        logger.debug(f"Component scores - Skills: {skills_score}, Experience: {experience_score}, Education: {education_score}, Industry: {industry_score}")
    
        adjusted_score = (
            original_score * 0.4 +  # Reduced base score weight
            skills_score * skills_weight * 0.2 +
            experience_score * experience_weight * 0.2 +
            education_score * education_weight * 0.1 +
            industry_score * industry_weight * 0.1
        )
    
        # Cap the maximum adjustment
        max_adjustment = 20  # Increased maximum adjustment
        final_score = min(original_score + max_adjustment, adjusted_score)
    
        logger.debug(f"Adjusted score: {final_score}")
    except Exception as e:
        logger.error(f"Error in adjust_match_score: {str(e)}")
        final_score = original_score  # Use original score if there's an error
    
        logger.debug(f"Adjusted score: {adjusted_score}")
        return min(max(int(adjusted_score), 0), 100)  # Ensure score is between 0 and 100
    
def evaluate_skills(skills_gap, required_skills):
    logger.debug(f"Evaluating skills. Skills gap: {skills_gap}, Required skills: {required_skills}")
    
    # Handle both list and dictionary inputs for skills_gap
    if isinstance(skills_gap, dict):
        candidate_skills = skills_gap.get('key_skills', [])
        if isinstance(candidate_skills, str):
            candidate_skills = [candidate_skills]
    elif isinstance(skills_gap, list):
        candidate_skills = skills_gap
    else:
        candidate_skills = []
    
    logger.debug(f"Candidate skills: {candidate_skills}")
    
    # Ensure required_skills is a list
    if isinstance(required_skills, str):
        required_skills = [required_skills]
    elif not isinstance(required_skills, list):
        required_skills = []
    
    # Check how many required skills are missing
    missing_skills = [skill for skill in required_skills if skill.lower() not in [s.lower() for s in candidate_skills]]
    
    logger.debug(f"Missing skills: {missing_skills}")
    
    # Max score based on total required skills
    max_possible_score = len(required_skills) * 10  # e.g., 10 points per skill
    
    if max_possible_score == 0:
        logger.debug("No required skills listed, returning perfect score")
        return 100  # No required skills listed means perfect score
    
    score = max(100 - (len(missing_skills) / len(required_skills)) * 100, 0)
    logger.debug(f"Skills evaluation score: {score}")
    return score
def evaluate_experience(professional_experience, academic_experience, years_required):
    try:
        # Calculate total years of experience
        total_professional_years = sum(float(years) for _, years in professional_experience.items() if isinstance(years, (int, float)))
        total_academic_years = sum(float(years) for _, years in academic_experience.items() if isinstance(years, (int, float)))
        
        # Calculate relevant years of experience (relevance >= 0.7)
        relevant_professional_years = sum(float(years) for relevance, years in professional_experience.items() if isinstance(relevance, (int, float)) and relevance >= 0.7)
        relevant_academic_years = sum(float(years) for relevance, years in academic_experience.items() if isinstance(relevance, (int, float)) and relevance >= 0.7)
        
        # Weight professional experience more heavily
        weighted_total_years = total_professional_years * 1.5 + total_academic_years * 0.5
        weighted_relevant_years = relevant_professional_years * 1.5 + relevant_academic_years * 0.5
        
        # Calculate factors
        years_factor = min(weighted_total_years / years_required, 1) if years_required > 0 else 1
        relevance_factor = weighted_relevant_years / weighted_total_years if weighted_total_years > 0 else 0
        
        # Calculate final score
        experience_score = (years_factor * 0.6 + relevance_factor * 0.4) * 100
        
        logger.debug(f"Experience evaluation - Professional: {total_professional_years} years, Academic: {total_academic_years} years")
        logger.debug(f"Weighted total years: {weighted_total_years}, Weighted relevant years: {weighted_relevant_years}")
        logger.debug(f"Experience score: {experience_score}")
        
        return experience_score
    except (ValueError, TypeError) as e:
        logger.error(f"Error processing experience relevance: {str(e)}")
        return 0

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
