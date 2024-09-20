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
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
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
    """
    Generate job requirements by processing the job description with spaCy and extracting relevant information.
    """
    nlp = get_spacy_model()
    doc = nlp(job_description)
    
    required_skills = {ent.text.lower() for ent in doc.ents if ent.label_ in ["SKILL", "PRODUCT", "ORG"]}
    
    experience_pattern = r'\b(\d+)(?:\+)?\s*(?:years?|yrs?)\b.*?experience'
    experience_matches = re.findall(experience_pattern, job_description, re.IGNORECASE)
    years_of_experience = max(map(int, experience_matches)) if experience_matches else 0
    
    education_levels = ["high school", "associate", "bachelor", "master", "phd", "doctorate"]
    education_level = next((level.title() for level in education_levels if level in job_description.lower()), "Not specified")
    
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

    sorted_results = sorted(evaluation_results, key=lambda x: x.get('match_score', 0), reverse=True)

    df = pd.DataFrame(sorted_results)
    df['Rank'] = range(1, len(df) + 1)
    
    columns_to_display = ['Rank', 'file_name', 'match_score', 'recommendation']
    column_names = ['Rank', 'Candidate', 'Match Score (%)', 'Recommendation']
    
    if 'confidence_score' in df.columns:
        columns_to_display.append('confidence_score')
        column_names.append('Confidence Score')
    
    df = df[columns_to_display]
    df.columns = column_names
    df = df.set_index('Rank')

    st.subheader("Candidate Summary")
    st.dataframe(df.style.format({'Match Score (%)': '{:.0f}', 'Confidence Score': '{:.2f}'}))

    for i, result in enumerate(sorted_results, 1):
        with st.expander(f"Rank {i}: {result.get('file_name', 'Unknown')} - Detailed Analysis", expanded=i == 1):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Match Score", f"{result.get('match_score', 0)}%")
            with col2:
                st.metric("Confidence Score", f"{result.get('confidence_score', 0):.2f}")
            with col3:
                st.write("**Recommendation:**", result.get('recommendation', 'N/A'))

            st.write("**Brief Summary:**", result.get('brief_summary', 'No summary available.'))
            st.write("**Fit Summary:**", result.get('fit_summary', 'No fit summary available.'))

            st.subheader("Experience and Project Relevance")
            relevance = result.get('experience_and_project_relevance', {})
            if isinstance(relevance, dict):
                st.write(f"**Overall Relevance:** {relevance.get('overall_relevance', 0)}")
                st.write(f"**Relevant Experience:** {relevance.get('relevant_experience', 0)}")
                st.write(f"**Project Relevance:** {relevance.get('project_relevance', 0)}")
                st.write(f"**Technical Skills Relevance:** {relevance.get('technical_skills_relevance', 0)}")
                st.write(relevance.get('description', 'No description available.'))
            else:
                st.write(relevance)

            st.subheader("Skills Gap")
            skills_gap = result.get('skills_gap', {})
            st.write("**Missing Skills:**")
            for skill in skills_gap.get('missing_skills', []):
                st.write(f"- {skill}")
            st.write(skills_gap.get('description', 'No description available.'))

            st.subheader("Key Strengths")
            for strength in result.get('key_strengths', []):
                st.write(f"- {strength}")

            st.subheader("Areas for Improvement")
            for area in result.get('areas_for_improvement', []):
                st.write(f"- {area}")

            st.subheader("Recommended Interview Questions")
            for question in result.get('recruiter_questions', []):
                st.write(f"- {question}")

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

def display_nested_content(data):
    """
    Simplifies nested structures (dictionaries/lists) into human-readable strings.
    Ensures the output is always a string.
    """
    if isinstance(data, dict):
        return '\n'.join(f"{k.capitalize()}: {v}" for k, v in data.items())
    elif isinstance(data, list):
        return ', '.join(str(item) for item in data)
    else:
        return str(data)

def format_output(content):
    if isinstance(content, str):
        sentences = content.split('. ')
        formatted_sentences = [s.capitalize() for s in sentences]
        return '. '.join(formatted_sentences)
    elif isinstance(content, list):
        return [format_output(item) for item in content]
    elif isinstance(content, dict):
        return {k: format_output(v) for k, v in content.items()}
    else:
        return content

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

def process_resume(resume_file, resume_processor, job_description, importance_factors, job_title, key_skills, llm, job_requirements):
    logger = get_logger(__name__)
    logger.debug(f"Processing resume: {resume_file.name} with {resume_processor.backend} backend")
    try:
        resume_text = extract_text_from_file(resume_file)
        if not resume_text.strip():
            logger.warning(f"Empty content extracted from {resume_file.name}")
            return _generate_error_result(resume_file.name, "Empty content extracted")
        
        result = resume_processor.analyze_match(resume_text, job_description, {}, job_title)
        logger.debug(f"Initial analysis result: {json.dumps(result, indent=2)}")
        
        if not isinstance(result, dict):
            logger.error(f"Unexpected result type: {type(result)}")
            return _generate_error_result(resume_file.name, "Unexpected result type")
        
        result = {
        'file_name': resume_file.name,
        'brief_summary': result.get('brief_summary', "No summary available"),
        'match_score': int(result.get('match_score', 0)),
        'recommendation': result.get('recommendation', "No recommendation available"),
        'experience_and_project_relevance': result.get('experience_and_project_relevance', "No relevance information available"),
        'skills_gap': result.get('skills_gap', []),
        'key_strengths': result.get('key_strengths', []),
        'areas_for_improvement': result.get('areas_for_improvement', []),
        'recruiter_questions': result.get('recruiter_questions', []),
        'confidence_score': result.get('confidence_score', 0) 
    }
        
        if not result['brief_summary']:
            result['brief_summary'] = generate_brief_summary(result['match_score'], job_title)
        if not result['recommendation']:
            result['recommendation'] = get_recommendation(result['match_score'])
        
        logger.debug(f"Final processed result: {json.dumps(result, indent=2)}")
        return result
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
        return (f"The candidate shows limited alignment with the {job_title} role requirements. "
                f"With a match score of {score}%, there are significant areas for improvement.")
    elif 30 <= score < 50:
        return (f"The candidate demonstrates some potential for the {job_title} role. "
                f"With a match score of {score}%, there are notable gaps in required skills.")
    elif 50 <= score < 70:
        return (f"The candidate shows moderate potential for the {job_title} role. "
                f"With a match score of {score}%, they meet some key requirements but may need development in others.")
    elif 70 <= score < 85:
        return (f"The candidate is a strong fit for the {job_title} role. "
                f"With a match score of {score}%, they demonstrate solid alignment with the required skills and experience.")
    else:
        return (f"The candidate is an excellent match for the {job_title} role. "
                f"With a match score of {score}%, they exceed expectations in meeting the required skills and experience.")

def generate_fit_summary(result: Dict[str, Any]) -> str:
    score = result['match_score']
    if score < 50:
        return ("The candidate does not sufficiently meet the job requirements and is not recommended for further consideration.")
    elif 50 <= score < 70:
        return ("The candidate meets some job requirements but has areas for growth.")
    elif 70 <= score < 85:
        return ("The candidate aligns well with most job requirements and shows potential in others.")
    else:
        return ("The candidate is an excellent fit, meeting or exceeding most job requirements.")

def get_recommendation(match_score: int) -> str:
    if match_score < 50:
        return ("Do not recommend for interview; the candidate does not sufficiently meet the role requirements.")
    elif 50 <= match_score < 70:
        return ("Consider for interview with reservations; be prepared to discuss areas for improvement.")
    elif 70 <= match_score < 85:
        return ("Recommend for interview; the candidate is a strong fit.")
    else:
        return ("Highly recommend for interview; the candidate is an excellent match.")

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
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    elements = []

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

    elements.append(Paragraph(f"Evaluation Report (Run ID: {run_id})", styles['Heading1']))
    elements.append(Spacer(1, 12))

    # Summary table
    data = [['Rank', 'Candidate', 'Match Score (%)', 'Recommendation']]
    for i, result in enumerate(sorted(evaluation_results, key=lambda x: x.get('match_score', 0), reverse=True), 1):
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
    elements.append(Spacer(1, 12))

    for result in evaluation_results:
        elements.append(Paragraph(f"{result.get('file_name', 'Unknown')}", styles['Heading2']))
        elements.append(Paragraph(f"Match Score: {result.get('match_score', 0)}%", styles['Normal']))
        elements.append(Paragraph(f"Recommendation: {result.get('recommendation', 'N/A')}", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Brief Summary:", styles['Heading3']))
        elements.append(Paragraph(result.get('brief_summary', 'No summary available.'), styles['Justify']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Fit Summary:", styles['Heading3']))
        elements.append(Paragraph(result.get('fit_summary', 'No fit summary available.'), styles['Justify']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Experience and Project Relevance:", styles['Heading3']))
        exp_relevance = result.get('experience_and_project_relevance', 'Not provided')
        if isinstance(exp_relevance, list):
            for item in exp_relevance:
                elements.append(Paragraph(f"- {item.get('experience', '')} (Relevance: {item.get('relevance', 'N/A')})", styles['Normal']))
                elements.append(Paragraph(f"  {item.get('description', '')}", styles['Justify']))
        else:
            elements.append(Paragraph(str(exp_relevance), styles['Justify']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Skills Gap:", styles['Heading3']))
        skills_gap = result.get('skills_gap', 'Not provided')
        if isinstance(skills_gap, list):
            for item in skills_gap:
                elements.append(Paragraph(f"- {item.get('skill', '')} (Relevance: {item.get('relevance', 'N/A')})", styles['Normal']))
                elements.append(Paragraph(f"  {item.get('description', '')}", styles['Justify']))
        else:
            elements.append(Paragraph(str(skills_gap), styles['Justify']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Key Strengths:", styles['Heading3']))
        strengths = result.get('key_strengths', [])
        if strengths:
            for strength in strengths:
                elements.append(Paragraph(f"- {strength}", styles['Normal']))
        else:
            elements.append(Paragraph("No key strengths provided.", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Areas for Improvement:", styles['Heading3']))
        improvements = result.get('areas_for_improvement', [])
        if improvements:
            for area in improvements:
                elements.append(Paragraph(f"- {area}", styles['Normal']))
        else:
            elements.append(Paragraph("No areas for improvement provided.", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Recommended Interview Questions:", styles['Heading3']))
        questions = result.get('recruiter_questions', [])
        if questions:
            for question in questions:
                if isinstance(question, dict):
                    elements.append(Paragraph(f"- {question.get('question', '')}", styles['Normal']))
                    elements.append(Paragraph(f"  Reason: {question.get('description', '')}", styles['Justify']))
                else:
                    elements.append(Paragraph(f"- {question}", styles['Normal']))
        else:
            elements.append(Paragraph("No specific questions generated.", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("", styles['Normal']))  # Add a blank line between candidates

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
    
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
    
    return min(max(int(final_score), 0), 100)  # Ensure score is between 0 and 100

def evaluate_skills(skills_gap, required_skills):
    logger.debug(f"Evaluating skills. Skills gap: {skills_gap}, Required skills: {required_skills}")
    
    if isinstance(skills_gap, dict):
        candidate_skills = skills_gap.get('key_skills', [])
        if isinstance(candidate_skills, str):
            candidate_skills = [candidate_skills]
    elif isinstance(skills_gap, list):
        candidate_skills = skills_gap
    else:
        candidate_skills = []
    
    logger.debug(f"Candidate skills: {candidate_skills}")
    
    if isinstance(required_skills, str):
        required_skills = [required_skills]
    elif not isinstance(required_skills, list):
        required_skills = []
    
    missing_skills = [skill for skill in required_skills if skill.lower() not in [s.lower() for s in candidate_skills]]
    
    logger.debug(f"Missing skills: {missing_skills}")
    
    max_possible_score = len(required_skills) * 10  # e.g., 10 points per skill
    
    if max_possible_score == 0:
        logger.debug("No required skills listed, returning perfect score")
        return 100  # No required skills listed means perfect score
    
    score = max(100 - (len(missing_skills) / len(required_skills)) * 100, 0)
    logger.debug(f"Skills evaluation score: {score}")
    return score

def evaluate_experience(professional_experience, academic_experience, years_required):
    try:
        total_professional_years = sum(float(years) for _, years in professional_experience.items() if isinstance(years, (int, float)))
        total_academic_years = sum(float(years) for _, years in academic_experience.items() if isinstance(years, (int, float)))
        
        relevant_professional_years = sum(float(years) for relevance, years in professional_experience.items() if isinstance(relevance, (int, float)) and relevance >= 0.7)
        relevant_academic_years = sum(float(years) for relevance, years in academic_experience.items() if isinstance(relevance, (int, float)) and relevance >= 0.7)
        
        weighted_total_years = total_professional_years * 1.5 + total_academic_years * 0.5
        weighted_relevant_years = relevant_professional_years * 1.5 + relevant_academic_years * 0.5
        
        years_factor = min(weighted_total_years / years_required, 1) if years_required > 0 else 1
        relevance_factor = weighted_relevant_years / weighted_total_years if weighted_total_years > 0 else 0
        
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