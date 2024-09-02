import os
import re
import uuid
from typing import List, Dict, Any
from datetime import datetime
import logging
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
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize SpaCy
nlp = spacy.load("en_core_web_md")

USER_CREDENTIALS = {
    "username": os.getenv('LOGIN_USERNAME'),
    "password": os.getenv('LOGIN_PASSWORD')
}

DB_PATH = os.getenv('SQLITE_DB_PATH')

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

@st.cache_data
def process_resume(resume_file, _resume_processor, job_description, importance_factors):
    try:
        resume_text = extract_text_from_file(resume_file)
        result = _resume_processor.analyze_resume(resume_text, job_description, importance_factors)
        result['file_name'] = resume_file.name
        return result
    except Exception as e:
        error_msg = f"Error processing resume {resume_file.name}: {str(e)}"
        logging.error(error_msg)
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

def process_resumes_in_parallel(resume_files: List, resume_processor, job_description: str, importance_factors: Dict[str, float]) -> List[Dict[str, Any]]:
    def process_with_context(file):
        return process_resume(file, resume_processor, job_description, importance_factors)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for file in resume_files:
            future = executor.submit(process_with_context, file)
            add_script_run_ctx(future)
            futures.append(future)
        
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results

def display_results(evaluation_results: List[Dict[str, Any]], run_id: str):
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
            st.write("Summary:", result.get('summary', 'No summary available'))
            st.write("Analysis:", result.get('analysis', 'No analysis available'))
            st.write("Strengths:", ", ".join(result.get('strengths', ['No strengths identified'])))
            st.write("Areas for Improvement:", ", ".join(result.get('areas_for_improvement', ['No areas for improvement identified'])))
            st.write("Skills Gap:", ", ".join(result.get('skills_gap', ['No skills gap identified'])))
            st.write("Interview Questions:", ", ".join(result.get('interview_questions', ['No interview questions available'])))
            st.write("Project Relevance:", result.get('project_relevance', 'No project relevance analysis available'))

            with st.form(key=f'feedback_form_{i}'):
                st.subheader("Provide Feedback")
                accuracy_rating = st.select_slider("Accuracy of the evaluation:", options=[1, 2, 3, 4, 5])
                content_rating = st.select_slider("Quality of the report content:", options=[1, 2, 3, 4, 5])
                suggestions = st.text_area("Please provide any suggestions for improvement:")
                submit_feedback = st.form_submit_button("Submit Feedback")

                if submit_feedback:
                    if save_feedback(run_id, result['file_name'], accuracy_rating, content_rating, suggestions, DB_PATH):
                        st.success("Thank you for your feedback!")
                    else:
                        st.error("Failed to save feedback. Please try again.")

def is_valid_fractal_job_link(url):
    # Pattern to match Fractal's Workday job posting URLs
    pattern = r'^https?://fractal\.wd1\.myworkdayjobs\.com/.*Careers/.*'
    return re.match(pattern, url) is not None

def extract_job_description(url):
    if not is_valid_fractal_job_link(url):
        raise ValueError("Invalid job link. Please use a link from Fractal's career site.")
    
    options = Options()
    options.add_argument("--headless")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        driver.get(url)
        
        # Wait for the job description to load
        wait = WebDriverWait(driver, 10)
        job_description_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-automation-id='jobPostingDescription']")))
        
        # Extract the text
        job_description = job_description_element.text
        
        return job_description
    except Exception as e:
        st.error(f"Failed to extract job description: {str(e)}")
        return None
    finally:
        driver.quit()

def main_app():
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown("<h1 class='main-title'>Resume Cupid 💘</h1>", unsafe_allow_html=True)
    st.markdown("Resume Cupid is an intelligent resume evaluation tool designed to streamline the hiring process. Upload one or multiple resumes to evaluate and rank candidates for a specific role.")

    # Initialize session state variables
    if 'roles_updated' not in st.session_state:
        st.session_state.roles_updated = False

    for key in ['role_name_input', 'job_description', 'current_role_name', 'job_description_link']:
        if key not in st.session_state:
            st.session_state[key] = ''

    if 'saved_roles' not in st.session_state:
        st.session_state.saved_roles = get_saved_roles(st.session_state.get("current_user", ""))

    llm_descriptions = {
        "Claude": "Highly effective for natural language understanding and generation. Developed by Anthropic.",
        "Llama": "Large language model with strong performance on various NLP tasks. Created by Meta AI.",
        "GPT-4o-mini": "Compact version of GPT-4 with impressive capabilities. Open-source alternative."
    }
    
    backend = st.selectbox(
        "Select LLM backend:",
        [f"{llm}: {description}" for llm, description in llm_descriptions.items()]
    )
    backend = backend.split(":")[0].strip()

    resume_processor = create_resume_processor(backend)

    if not resume_processor:
        return
    
    # Dropdown for saved roles
    role_options = [""] + [f"{role['role_name']} - {role['client']}" for role in st.session_state.saved_roles]

    selected_saved_role = st.selectbox("Select a saved role:", options=role_options, key="selected_saved_role")

    if selected_saved_role and selected_saved_role != st.session_state.get('last_selected_role'):
        selected_role_data = next((role for role in st.session_state.saved_roles if f"{role['role_name']} - {role['client']}" == selected_saved_role), None)
        if selected_role_data:
            st.session_state.current_role_name = selected_role_data['role_name']
            st.session_state.role_name_input = selected_role_data['role_name']
            st.session_state.client = selected_role_data['client']
            st.session_state.job_description = selected_role_data['job_description']
            st.session_state.job_description_link = selected_role_data['job_description_link']
            st.session_state.last_selected_role = selected_saved_role

    role_name_input = st.text_input("Enter the role title:", value=st.session_state.get('role_name_input', ''))

    # New text input for the client name
    client_input = st.text_input("Enter the client name:", value=st.session_state.get('client', ''))

    jd_option = st.radio("Job Description Input Method:", ("Paste Job Description", "Provide Link to Fractal Job Posting"))

    with st.form(key='job_description_form'):
        if jd_option == "Paste Job Description":
            job_description = st.text_area(
                "Paste the Job Description here:", 
                value=st.session_state.get('job_description', ''), 
                placeholder="Job description. This field is required."
            )
            job_description_link = ""
            submit_label = "Submit Job Description"
        else:
            job_description = ""
            job_description_link = st.text_input(
                "Enter the link to the Fractal job posting:", 
                value=st.session_state.get('job_description_link', '')
            )
            submit_label = "Extract Job Description"

        submit_button = st.form_submit_button(submit_label)

    if submit_button:
        if jd_option == "Paste Job Description":
            if job_description:
                st.session_state.job_description = job_description
                st.success("Job description submitted successfully!")
            else:
                st.error("Please paste a job description before submitting.")
        else:
            if job_description_link:
                if is_valid_fractal_job_link(job_description_link):
                    with st.spinner('Extracting job description...'):
                        extracted_job_description = extract_job_description(job_description_link)
                    if extracted_job_description:
                        st.session_state.job_description = extracted_job_description
                        st.success("Job description extracted successfully!")
                    else:
                        st.error("Failed to extract job description. Please try pasting it manually.")
                else:
                    st.error("Invalid job link. Please use a link from Fractal's career site.")
            else:
                st.error("Please enter a job posting link before submitting.")

    # Display the current job description
    if st.session_state.job_description:
        st.text_area("Current Job Description:", value=st.session_state.job_description, height=200, key='current_jd')

    with st.form(key='importance_factors_form'):
        st.subheader("Customize Importance Factors")
        col1, col2, col3 = st.columns(3)
        with col1:
            education_importance = st.slider("Education Importance", 0.0, 1.0, 0.5, 0.1)
        with col2:
            experience_importance = st.slider("Experience Importance", 0.0, 1.0, 0.5, 0.1)
        with col3:
            skills_importance = st.slider("Skills Importance", 0.0, 1.0, 0.5, 0.1)

        importance_factors = {
            "education": education_importance,
            "experience": experience_importance,
            "skills": skills_importance
        }
        
        update_factors = st.form_submit_button("Update Importance Factors")

    if update_factors:
        st.success("Importance factors updated successfully!")

    resume_files = st.file_uploader("Upload your resume(s)", type=['pdf', 'docx'], accept_multiple_files=True)

    if st.button('Process Resumes', key='process_resumes'):
        if resume_files and st.session_state.job_description:
            run_id = str(uuid.uuid4())
            insert_run_log(run_id, "start_analysis", f"Starting analysis for {len(resume_files)} resumes")
            
            with st.spinner('Processing resumes...'):
                evaluation_results = process_resumes_in_parallel(resume_files, resume_processor, st.session_state.job_description, importance_factors)
            
            insert_run_log(run_id, "end_analysis", f"Completed analysis for {len(resume_files)} resumes")

            if evaluation_results:
                st.success("Evaluation complete!")
                display_results(evaluation_results, run_id)
            else:
                st.warning("No resumes were successfully processed. Please check the uploaded files and try again.")
        else:
            st.error("Please upload at least one resume and ensure a job description is provided.")

    st.session_state.role_name_input = role_name_input
    st.session_state.client = client_input
    st.session_state.job_description_link = job_description_link

    save_role_option = st.checkbox("Save this role for future use")
    if save_role_option:
        with st.form(key='save_role_form'):
            saved_role_name = st.text_input("Save role as (e.g., Job Title):", value=role_name_input)
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
    else:
        st.warning("No roles available to delete.")

if __name__ == "__main__":
    init_db()
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()


