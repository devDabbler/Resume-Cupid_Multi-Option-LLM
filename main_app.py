from config_settings import Config
import logging
import uuid
import streamlit as st
from utils import (
    extract_text_from_file, preprocess_text,
    split_into_batches, process_all_batches, process_resumes_in_parallel,
    display_results, is_valid_fractal_job_link, extract_job_description,
    get_available_api_keys, clear_cache
)
from database import (
    init_db, insert_run_log, save_role, get_saved_roles, delete_saved_role, save_feedback, get_logger
)
from resume_processor import create_resume_processor
from candidate_data import get_candidate_data
import os
from claude_analyzer import ClaudeAPI
from gpt4o_mini_analyzer import GPT4oMiniAPI
from llama_analyzer import LlamaAPI

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = get_logger(__name__)

# Get the BASE_URL from the Config
BASE_URL = Config.BASE_URL

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load API keys from Config
claude_api_key = Config.CLAUDE_API_KEY
gpt4o_mini_api_key = Config.GPT4O_MINI_API_KEY
llama_api_key = Config.LLAMA_API_KEY

# Initialize API clients
try:
    claude_api = ClaudeAPI(claude_api_key) if claude_api_key else None
    llama_api = LlamaAPI(llama_api_key) if llama_api_key else None
    gpt4o_mini_api = GPT4oMiniAPI(gpt4o_mini_api_key) if gpt4o_mini_api_key else None
    
    logger.info("API clients initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize API clients: {str(e)}")
    st.error("Failed to initialize API clients. Please check your configuration.")

# Environment type (e.g., development or production)
ENVIRONMENT = Config.ENVIRONMENT

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

BATCH_SIZE = 3  # Number of resumes to process in each batch

def main_app():
    init_db()
    """Main application interface that should only be accessible after login."""
    
    st.markdown("",unsafe_allow_html=True)
    st.markdown("<h1 class='main-title'>Resume Cupid 💘</h1>", unsafe_allow_html=True)
    st.markdown("Resume Cupid is an intelligent resume evaluation tool designed to streamline the hiring process. Upload one or multiple resumes to evaluate and rank candidates for a specific role.")

    # Initialize session state variables
    if 'roles_updated' not in st.session_state:
        st.session_state.roles_updated = False

    for key in ['role_name_input', 'job_description', 'current_role_name', 'job_description_link', 'importance_factors', 'backend', 'resume_processor', 'last_backend']:
        if key not in st.session_state:
            st.session_state[key] = '' if key != 'importance_factors' else {'education': 0.5, 'experience': 0.5, 'skills': 0.5}

    if 'saved_roles' not in st.session_state:
        st.session_state.saved_roles = get_saved_roles()

    llm_descriptions = {
        "claude": "3.5 Sonnet is Anthropic's most recent model that is highly effective for natural language understanding and generation. It comes at a premium but very accurate.",
        "llama": "Created by Meta AI, this is the llama-3.1, 8 billion parameter model. This is the latest and most capable large language model with strong performance on various NLP tasks.",
        "gpt4o_mini": "This is the compact version of GPT-4 with impressive capabilities. Open-source alternative."
    }

    api_keys = get_available_api_keys()

    if not api_keys:
        st.error("No API keys found. Please set at least one API key in the environment variables.")
        return

    available_backends = list(api_keys.keys())
    backend_options = []
    for backend in available_backends:
        if backend in llm_descriptions:
            backend_options.append(f"{backend}: {llm_descriptions[backend]}")
        else:
            backend_options.append(f"{backend}: No description available")

    if not backend_options:
        st.error("No compatible backends found. Please check your API key configuration.")
        return

    selected_backend = st.selectbox(
        "Select AI backend:",
        backend_options,
        index=0
    )
    selected_backend = selected_backend.split(":")[0].strip()

    # Check if the backend has changed and clear cache if necessary
    if selected_backend != st.session_state.get('last_backend'):
        selected_api_key = api_keys[selected_backend]
        st.session_state.resume_processor = create_resume_processor(selected_api_key, selected_backend)
        if hasattr(st.session_state.resume_processor, 'clear_cache'):
            clear_cache()  # Clear the cache when switching backends
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
                display_results(evaluation_results, run_id, save_feedback)
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
                    if save_role(full_role_name, client, st.session_state.job_description, st.session_state.job_description_link):
                        st.success(f"Role '{full_role_name}' saved successfully!")
                        st.session_state.saved_roles = get_saved_roles()
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
                    if delete_saved_role(role_name_to_delete):
                        st.success(f"Role '{delete_role_name}' deleted successfully!")
                        st.session_state.saved_roles = get_saved_roles()
                    else:
                        raise Exception("Deletion failed")
                except Exception as e:
                    st.error(f"Failed to delete role '{delete_role_name}'. Error: {str(e)}")

    # Add a logout button
    if st.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

if __name__ == "__main__":
    pass