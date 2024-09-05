import os
from config import Config, ENV_TYPE
import re
import uuid
from typing import List, Dict, Any
from datetime import datetime
import time
import logging
from logging.handlers import RotatingFileHandler
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import spacy
import numpy as np
import concurrent.futures
from utils import extract_text_from_file, preprocess_text
from database import init_db, insert_run_log, save_role, get_saved_roles, delete_saved_role, save_feedback
from resume_processor import create_resume_processor
from streamlit.runtime.scriptrunner import add_script_run_ctx
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from candidate_data import get_candidate_data

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Create logger
logger = logging.getLogger(__name__)

# Create RotatingFileHandler
os.makedirs(Config.LOG_DIR, exist_ok=True)
log_file = os.path.join(Config.LOG_DIR, "resume_cupid.log")
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Initialize SpaCy
nlp = spacy.load("en_core_web_md")

USER_CREDENTIALS = {
    "username": os.getenv('LOGIN_USERNAME'),
    "password": os.getenv('LOGIN_PASSWORD')
}

DB_PATH = os.getenv('SQLITE_DB_PATH')

# Session State Initialization
if 'importance_factors' not in st.session_state:
    st.session_state.importance_factors = {
        'education': 0.5,
        'experience': 0.5,
        'skills': 0.5
    }

if 'backend' not in st.session_state:
    st.session_state.backend = None
if 'resume_processor' not in st.session_state:
    st.session_state.resume_processor = None

# Custom CSS for branding and UI enhancement
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f2f6;
    }
    
    .main-title {
        font-size: 3.5em;
        font-weight: 700;
        color: #2C3E50;
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2em;
        color: #34495E;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .login-form {
        background-color: white;
        padding: 2.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        width: 100%;
        max-width: 350px;
        margin: 0 auto;
    }
    
    .login-button {
        background-color: #3498DB;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5em 1em;
        margin-top: 1.5rem;
        width: 100%;
    }
    .footer-text {
            text-align: center;
            width: 100%;
            max-width: 800px;
            margin: 1rem auto;
        }
        h3 {
            margin-bottom: 1rem !important;
        }
    .stButton>button {
        width: 100%;
    }
</style>
"""

def check_login(username: str, password: str) -> bool:
    """Check login credentials."""
    return USER_CREDENTIALS.get("username") == username and USER_CREDENTIALS.get("password") == password

def login_page():
    """Display the login form and manage user authentication."""
    st.markdown(custom_css, unsafe_allow_html=True)
    
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-title'>Resume Cupid 💘</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Login to access the resume evaluation tool.</p>", unsafe_allow_html=True)

    with st.form(key='login_form', clear_on_submit=False):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.current_user = username
                st.success("Login successful! Redirecting to main app...")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    st.markdown("<div class='footer-text'>Need access? <a href='mailto:hello@resumecupid.ai'>Contact us</a> to get started!</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


BATCH_SIZE = 3  # Number of resumes to process in each batch

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
        
        result['file_name'] = resume_file.name
        return result
    except Exception as e:
        logger.error(f"Error processing resume {resume_file.name}: {str(e)}", exc_info=True)
        return {
            'file_name': resume_file.name,
            'error': str(e),
            'match_score': 0,
            'recommendation': 'Unable to provide a recommendation due to an error.',
            'analysis': 'Unable to complete analysis',
            'strengths': [],
            'areas_for_improvement': [],
            'skills_gap': [],
            'interview_questions': [],
            'project_relevance': ''
        }

def process_resumes_in_parallel(resume_files, resume_processor, job_description, importance_factors, candidate_data_list, job_title):
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
            return {
                'file_name': file.name,
                'error': str(e),
                'match_score': 0,
                'recommendation': 'Error occurred during processing',
                'analysis': f'An error occurred: {str(e)}',
                'strengths': [],
                'areas_for_improvement': [],
                'skills_gap': [],
                'interview_questions': [],
                'project_relevance': ''
            }

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for file, candidate_data in zip(resume_files, candidate_data_list):
            future = executor.submit(process_with_context, file, candidate_data)
            add_script_run_ctx(future)
            futures.append(future)
        
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    return results

def display_results(evaluation_results: List[Dict[str, Any]], run_id: str):
    st.subheader("Stack Ranking of Candidates")

    # Ensure all required keys exist in each result, with default values if missing
    for result in evaluation_results:
        result['match_score'] = result.get('match_score', 0)
        result['recommendation'] = result.get('recommendation', result.get('recommendation_for_interview', 'No recommendation available'))
    
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
                st.write(result.get('summary', 'No brief summary available'))

                st.markdown("### Match Score")
                st.write(f"{result.get('match_score', 'N/A')}%")

                st.markdown("### Recommendation")
                st.write(result.get('recommendation', 'No recommendation available'))

                st.markdown("### Experience and Project Relevance")
                st.write(result.get('analysis', 'No experience and project relevance data available'))
                
                st.markdown("### Skills Gap")
                skills_gap = result.get('skills_gap', [])
                if isinstance(skills_gap, list) and skills_gap:
                    for skill in skills_gap:
                        st.write(f"- {skill}")
                else:
                    st.write("No skills gap identified" if isinstance(skills_gap, list) else skills_gap)
                
                st.markdown("### Recruiter Questions")
                recruiter_questions = result.get('interview_questions', [])
                if isinstance(recruiter_questions, list) and recruiter_questions:
                    for question in recruiter_questions:
                        st.write(f"- {question}")
                else:
                    st.write("No recruiter questions available" if isinstance(recruiter_questions, list) else recruiter_questions)

            # Feedback form with unique key for each result
            with st.form(key=f'feedback_form_{run_id}_{i}'):
                st.subheader("Provide Feedback")
                accuracy_rating = st.select_slider("Accuracy of the evaluation:", options=[1, 2, 3, 4, 5])
                content_rating = st.select_slider("Quality of the report content:", options=[1, 2, 3, 4, 5])
                suggestions = st.text_area("Please provide any suggestions for improvement:")
                submit_feedback = st.form_submit_button("Submit Feedback")

                if submit_feedback:
                    if save_feedback(run_id, result['file_name'], accuracy_rating, content_rating, suggestions):
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
    
    options = Config.get_chrome_options()
    
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
    for backend in ["claude", "llama", "gpt4o_mini"]:
        key = os.getenv(f'{backend.upper()}_API_KEY')
        if key:
            api_keys[backend] = key
    return api_keys

def clear_cache():
    if 'resume_processor' in st.session_state and hasattr(st.session_state.resume_processor, 'clear_cache'):
        st.session_state.resume_processor.clear_cache()
        logger.debug(f"Cleared resume analysis cache for backend: {st.session_state.backend}")
    else:
        logger.warning("Unable to clear cache: resume_processor not found or doesn't have clear_cache method")
        
def main_app():
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown("<h1 class='main-title'>Resume Cupid 💘</h1>", unsafe_allow_html=True)
    st.markdown("Resume Cupid is an intelligent resume evaluation tool designed to streamline the hiring process. Upload one or multiple resumes to evaluate and rank candidates for a specific role.")

    # Initialize session state variables
    if 'roles_updated' not in st.session_state:
        st.session_state.roles_updated = False

    for key in ['role_name_input', 'job_description', 'current_role_name', 'job_description_link', 'importance_factors', 'backend', 'resume_processor', 'last_backend']:
        if key not in st.session_state:
            st.session_state[key] = '' if key != 'importance_factors' else {'education': 0.5, 'experience': 0.5, 'skills': 0.5}

    if 'saved_roles' not in st.session_state:
        st.session_state.saved_roles = get_saved_roles(st.session_state.get("current_user", ""))

    llm_descriptions = {
        "claude": "Highly effective for natural language understanding and generation. Developed by Anthropic.",
        "llama": "Large language model with strong performance on various NLP tasks. Created by Meta AI.",
        "gpt4o_mini": "Compact version of GPT-4 with impressive capabilities. Open-source alternative."
    }

    api_keys = get_available_api_keys()
    
    if not api_keys:
        st.error("No API keys found. Please set at least one API key in the environment variables.")
        return

    available_backends = list(api_keys.keys())
    selected_backend = st.selectbox(
        "Select AI backend:",
        [f"{backend.capitalize()}: {llm_descriptions[backend]}" for backend in available_backends],
        index=0 if "claude" in available_backends else 0
    )
    selected_backend = selected_backend.split(":")[0].strip().lower()

    # Check if the backend has changed and clear cache if necessary
    if selected_backend != st.session_state.get('last_backend'):
        selected_api_key = api_keys[selected_backend]
        st.session_state.resume_processor = create_resume_processor(selected_api_key, selected_backend)
        st.session_state.resume_processor.clear_cache()  # Clear the cache when switching backends
        st.session_state.last_backend = selected_backend
        logger.debug(f"Switched to {selected_backend} backend and cleared cache")

    st.session_state.backend = selected_backend

    logger.debug(f"Using ResumeProcessor with backend: {selected_backend}")

    resume_processor = st.session_state.resume_processor

    if not resume_processor:
        st.error("Failed to initialize resume processor. Please check your configuration.")
        return

    # Add this line to check if the analyze_match method exists
    if not hasattr(resume_processor, 'analyze_match'):
        raise AttributeError(f"ResumeProcessor for backend '{selected_backend}' does not have the 'analyze_match' method")

    # Add dropdown for selecting saved roles
    saved_role_options = ["Select a saved role"] + [f"{role['role_name']} - {role['client']}" for role in st.session_state.saved_roles]
    selected_saved_role = st.selectbox("Select a saved role:", options=saved_role_options)

    if selected_saved_role != "Select a saved role":
        selected_role = next(role for role in st.session_state.saved_roles if f"{role['role_name']} - {role['client']}" == selected_saved_role)
        st.session_state.job_description = selected_role['job_description']
        st.session_state.job_description_link = selected_role['job_description_link']
        st.session_state.role_name_input = selected_role['role_name']
        st.session_state.client = selected_role['client']

    # Add job title input field
    st.session_state.job_title = st.text_input("Enter the job title:", value=st.session_state.get('job_title', ''))

    jd_option = st.radio("Job Description Input Method:", ("Paste Job Description", "Provide Link to Fractal Job Posting"))

    if jd_option == "Paste Job Description":
        st.session_state.job_description = st.text_area(
            "Paste the Job Description here:", 
            value=st.session_state.get('job_description', ''), 
            placeholder="Job description. This field is required."
        )
        st.session_state.job_description_link = ""
    else:
        st.session_state.job_description_link = st.text_input(
            "Enter the link to the Fractal job posting:", 
            value=st.session_state.get('job_description_link', '')
        )
    if 'job_description' not in st.session_state or not st.session_state.job_description:
        if st.session_state.job_description_link and is_valid_fractal_job_link(st.session_state.job_description_link):
            with st.spinner('Extracting job description...'):
                st.session_state.job_description = extract_job_description(st.session_state.job_description_link)
        else:
            st.session_state.job_description = ""

    # Display the current job description
    if st.session_state.job_description:
        st.text_area("Current Job Description:", value=st.session_state.job_description, height=200, key='current_jd', disabled=True)

    # Move the client name input field here
    st.session_state.client = st.text_input("Enter the client name:", value=st.session_state.get('client', ''))

    st.subheader("Customize Importance Factors")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.importance_factors['education'] = st.slider("Education Importance", 0.0, 1.0, st.session_state.importance_factors['education'], 0.1)
    with col2:
        st.session_state.importance_factors['experience'] = st.slider("Experience Importance", 0.0, 1.0, st.session_state.importance_factors['experience'], 0.1)
    with col3:
        st.session_state.importance_factors['skills'] = st.slider("Skills Importance", 0.0, 1.0, st.session_state.importance_factors['skills'], 0.1)

    st.write("Upload your resume(s) (Maximum 3 files allowed):")
    resume_files = st.file_uploader("Upload resumes (max 3)", type=['pdf', 'docx'], accept_multiple_files=True)

    if resume_files:
        if len(resume_files) > 3:
            st.warning("You've uploaded more than 3 resumes. Only the first 3 will be processed.")
            resume_files = resume_files[:3]  # Truncate to first 3 files
        st.write(f"Number of resumes uploaded: {len(resume_files)}")

    if st.button('Process Resumes', key='process_resumes'):
        if resume_files and st.session_state.job_description and st.session_state.job_title:
            if len(resume_files) > 3:
                st.warning("You've uploaded more than 3 resumes. Only the first 3 will be processed.")
                resume_files = resume_files[:3]
        
            run_id = str(uuid.uuid4())
            insert_run_log(run_id, "start_analysis", f"Starting analysis for {len(resume_files)} resumes")
            logger.info("Processing resumes...")

            # Get candidate data
            candidates = get_candidate_data()
            candidate_data_list = []
            for file in resume_files:
                matching_candidate = next((c for c in candidates if c['candidate'] == file.name), None)
                candidate_data_list.append(matching_candidate or {})

            with st.spinner('Processing resumes...'):
                evaluation_results = process_resumes_in_parallel(resume_files, resume_processor, st.session_state.job_description, st.session_state.importance_factors, candidate_data_list, st.session_state.job_title)
            
            logger.debug(f"Processed resumes with {selected_backend} backend. Results: {evaluation_results}")
            insert_run_log(run_id, "end_analysis", f"Completed analysis for {len(resume_files)} resumes")

            if evaluation_results:
                st.success("Evaluation complete!")
                logger.info("Evaluation complete, displaying results.")
                display_results(evaluation_results, run_id)
            else:
                st.warning("No resumes were successfully processed. Please check the uploaded files and try again.")
        else:
            if not st.session_state.job_description:
                st.error("Please ensure a job description is provided.")
            if not st.session_state.job_title:
                st.error("Please enter a job title.")
            if not resume_files:
                st.error("Please upload at least one resume.")

    st.session_state.role_name_input = st.session_state.role_name_input
    st.session_state.job_description_link = st.session_state.job_description_link

    save_role_option = st.checkbox("Save this role for future use")
    if save_role_option:
        with st.form(key='save_role_form'):
            saved_role_name = st.text_input("Save role as (e.g., Job Title):", value=st.session_state.role_name_input)
            client = st.text_input("Client (required):", value=st.session_state.get('client', ''))
            save_button = st.form_submit_button('Save Role')

        if save_button:
            if not saved_role_name or not client:
                st.error("Please enter both a name for the role and a client before saving.")
            elif not st.session_state.job_description:
                st.error("Job description is required to save a role.")
            else:
                full_role_name = f"{saved_role_name} - {client}"
                try:
                    if save_role(st.session_state.get("current_user", ""), full_role_name, client, st.session_state.job_description, st.session_state.job_description_link):
                        st.success(f"Role '{full_role_name}' saved successfully!")
                        st.session_state.saved_roles = get_saved_roles(st.session_state.get("current_user", ""))
                    else:
                        st.error("Failed to save the role. Please try again.")
                except ValueError as e:
                    st.error(str(e))

    # Initialize delete_role_options as an empty list outside the checkbox condition
    delete_role_options = []
    
    if st.checkbox("Delete a saved role"):
        # Populate delete_role_options
        delete_role_options = [f"{role['role_name']} - {role['client']}" for role in st.session_state.saved_roles]

    if delete_role_options:
        delete_role_name = st.selectbox("Select a role to delete:", options=delete_role_options)

        if delete_role_name:
            if st.button(f"Delete {delete_role_name}"):
                # Split the role name and take everything up to the second occurrence of " - "
                role_parts = delete_role_name.rsplit(" - ", 2)
                role_name_to_delete = " - ".join(role_parts[:-1])
                
                # Call delete_saved_role with error handling
                try:
                    if delete_saved_role(st.session_state.get("current_user", ""), role_name_to_delete):
                        st.success(f"Role '{delete_role_name}' deleted successfully!")
                        st.session_state.saved_roles = get_saved_roles(st.session_state.get("current_user", ""))
                    else:
                        raise Exception("Deletion failed")
                except Exception as e:
                    st.error(f"Failed to delete role '{delete_role_name}'. Error: {str(e)}")

if __name__ == "__main__":
    init_db()
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()
