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
    sorted_results = sorted(evaluation_results, key=lambda x: x['match_score'], reverse=True)

    # Create a summary dataframe
    df = pd.DataFrame(sorted_results)
    df['Rank'] = range(1, len(df) + 1)
    df = df[['Rank', 'file_name', 'match_score', 'recommendation']]
    df.columns = ['Rank', 'Candidate', 'Match Score (%)', 'Recommendation']

    # Display summary table with custom styling
    st.subheader("Candidate Summary")

    # Custom styling function
    def color_match_score(val):
        color = 'red' if val < 50 else 'yellow' if val < 75 else 'green'
        return f'background-color: {color}'

    # Apply custom styling
    styled_df = df.style.format({'Match Score (%)': '{:.0f}'}) \
        .applymap(color_match_score, subset=['Match Score (%)']) \
        .set_properties(**{'text-align': 'left'}) \
        .set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

    st.dataframe(styled_df)

    # Create a download button for the PDF report
    st.download_button(
        label="Download Detailed PDF Report",
        data=generate_pdf_report(evaluation_results, run_id),
        file_name=f"evaluation_report_{run_id}.pdf",
        mime="application/pdf"
    )

    # Display detailed results for each candidate
    for i, result in enumerate(sorted_results, 1):
        with st.expander(f"Rank {i}: {result['file_name']} - Detailed Analysis", expanded=i == 1):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Candidate Overview")
                st.write(result['brief_summary'])
                
                st.subheader("Key Strengths")
                for strength in result['key_strengths']:
                    st.write(f"- {strength}")
                
                st.subheader("Areas for Improvement")
                for area in result['areas_for_improvement']:
                    st.write(f"- {area}")
            
            with col2:
                # Create a gauge chart for the match score
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = result['match_score'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Match Score", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': 'red'},
                            {'range': [50, 75], 'color': 'yellow'},
                            {'range': [75, 100], 'color': 'green'}
                        ],
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"Recommendation: {result['recommendation']}")

            st.subheader("Experience and Project Relevance")
            relevance = result['experience_and_project_relevance']
            if isinstance(relevance, dict):
                st.write(f"**Relevance Score:** {relevance.get('relevance_score', 'N/A')}")
                st.write("**Strengths:**")
                for strength in relevance.get('strengths', []):
                    st.write(f"- {strength}")
                if relevance.get('weaknesses'):
                    st.write("**Weaknesses:**")
                    for weakness in relevance['weaknesses']:
                        st.write(f"- {weakness}")
            else:
                st.write(relevance)  # Display the relevance information as is if it's not a dictionary

            st.subheader("Skills Gap")
            for skill in result['skills_gap']:
                st.write(f"- {skill}")

            st.subheader("Recommended Interview Questions")
            for question in result['recruiter_questions']:
                st.write(f"- {question}")

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

    st.success("Evaluation complete!")

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
        
        # Slightly lower the original match score
        original_score = result['match_score']
        adjusted_score = max(0, min(100, int(original_score * 0.9)))  # Reduce by 10% and ensure it's between 0 and 100
        
        logger.debug(f"Original score: {original_score}, Adjusted score: {adjusted_score}")
        
        # Update the result with the adjusted score
        result['match_score'] = adjusted_score
        
        # Generate a new brief summary based on the adjusted score
        result['brief_summary'] = generate_brief_summary(adjusted_score, job_title)
        
        # Format experience and project relevance
        exp_relevance = result.get('experience_and_project_relevance', {})
        formatted_exp_relevance = format_nested_structure(exp_relevance)
        
        # Format skills gap
        skills_gap = result.get('skills_gap', {})
        formatted_skills_gap = format_nested_structure(skills_gap)
        
        # Extract key strengths and areas for improvement using LLM
        strengths_and_improvements = get_strengths_and_improvements(resume_text, job_description, llm)
        
        # Format recruiter questions
        formatted_questions = format_recruiter_questions(result.get('recruiter_questions', []))
        
        processed_result = {
            'file_name': resume_file.name,
            'brief_summary': result['brief_summary'],
            'match_score': adjusted_score,
            'experience_and_project_relevance': formatted_exp_relevance,
            'skills_gap': formatted_skills_gap,
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

def format_nested_structure(data):
    if isinstance(data, dict):
        return {k: format_nested_structure(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [format_nested_structure(item) for item in data]
    elif isinstance(data, str):
        return data.replace('\n', ' ').strip()
    else:
        return str(data)

def format_recruiter_questions(questions):
    formatted = []
    for q in questions:
        if isinstance(q, dict):
            formatted.append(q.get('question', ''))
        else:
            formatted.append(str(q))
    return formatted[:5]  # Limit to 5 questions

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

    Each category should have 2-3 concise points (1-2 sentences each).
    """
    
    try:
        response = llm.analyze_match(resume_text, prompt, {}, "Structured Strengths and Improvements Analysis")
        strengths_and_improvements = json.loads(response['brief_summary'])
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
    
    skills_weight = importance_factors.get('technical_skills', 0.3)
    experience_weight = importance_factors.get('experience', 0.3)
    education_weight = importance_factors.get('education', 0.2)
    industry_weight = importance_factors.get('industry_knowledge', 0.2)
    
    try:
        skills_score = evaluate_skills(result.get('skills_gap', []), job_requirements.get('required_skills', []))
        experience_score = evaluate_experience(result.get('experience_and_project_relevance', {}), job_requirements.get('years_of_experience', 0))
        education_score = evaluate_education(result, job_requirements.get('education_level', 'Bachelor'))
        industry_score = evaluate_industry_knowledge(result, job_requirements.get('industry_keywords', []))
    
        logger.debug(f"Component scores - Skills: {skills_score}, Experience: {experience_score}, Education: {education_score}, Industry: {industry_score}")
    
        adjusted_score = (
            original_score * 0.6 +  # Base score weight
            skills_score * skills_weight * 0.1 +
            experience_score * experience_weight * 0.1 +
            education_score * education_weight * 0.1 +
            industry_score * industry_weight * 0.1
        )
    
        # Cap the maximum adjustment
        max_adjustment = 15  # Maximum 15 point increase
        final_score = min(original_score + max_adjustment, adjusted_score)
    
        logger.debug(f"Adjusted score: {final_score}")
    except Exception as e:
        logger.error(f"Error in adjust_match_score: {str(e)}")
        final_score = original_score  # Use original score if there's an error
    
    return min(max(int(final_score), 0), 100)  # Ensure score is between 0 and 100

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

def evaluate_experience(experience_relevance, years_required):
    try:
        total_years = sum(float(years) for _, years in experience_relevance.items() if isinstance(years, (int, float)))
        
        relevant_years = sum(float(years) for relevance, years in experience_relevance.items() if isinstance(relevance, (int, float)) and relevance >= 0.7)
        
        years_factor = min(total_years / years_required, 1) if years_required > 0 else 1
        relevance_factor = relevant_years / total_years if total_years > 0 else 0
        
        # Ensure balanced contribution from both factors
        return (years_factor * 0.6 + relevance_factor * 0.4) * 100
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
    
def generate_brief_summary(score, job_title):
    if score < 30:
        return f"The candidate is not a strong fit for the {job_title} role. With a match score of {score}%, there are significant gaps in required skills and experience for this position."
    elif 30 <= score < 50:
        return f"The candidate shows limited potential for the {job_title} role. With a match score of {score}%, there are considerable gaps in meeting the requirements. Further evaluation is needed."
    elif 50 <= score < 65:
        return f"The candidate shows some potential for the {job_title} role, but with a match score of {score}%, there are gaps in meeting the requirements. Further evaluation is recommended."
    elif 65 <= score < 80:
        return f"The candidate is a good fit for the {job_title} role. With a match score of {score}%, they demonstrate alignment with many of the required skills and experience for this position."
    else:
        return f"The candidate is an excellent fit for the {job_title} role. With a match score of {score}%, they demonstrate strong alignment with the required skills and experience for this position."

# Add any additional utility functions here as needed