from config_settings import Config
import logging
import uuid
import streamlit as st
from utils import extract_job_description, is_valid_fractal_job_link, get_available_api_keys, clear_cache, process_resume, display_results, extract_text_from_file, generate_job_requirements
from database import init_db, insert_run_log, save_role, delete_saved_role, get_saved_roles, save_feedback
from resume_processor import create_resume_processor
from candidate_data import get_candidate_data
import os
from claude_analyzer import ClaudeAPI
from gpt4o_mini_analyzer import GPT4oMiniAPI
from llama_analyzer import LlamaAPI, initialize_llm
from logger import get_logger

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

def _generate_error_result(file_name, error_message):
    return {
        "file_name": file_name,
        "error": error_message,
        "status": "error"
    }

def main_app():
    init_db()  # Initialize the database
    st.markdown("", unsafe_allow_html=True)
    st.markdown("<h1 class='main-title'>Resume Cupid 💘</h1>", unsafe_allow_html=True)
    st.markdown("Resume Cupid is an intelligent resume evaluation tool designed to streamline the hiring process. Upload one or multiple resumes to evaluate and rank candidates for a specific role.")

    # Initialize session state variables
    if 'roles_updated' not in st.session_state:
        st.session_state.roles_updated = False

    for key in ['role_name_input', 'job_description', 'current_role_name', 'job_description_link', 'backend', 'resume_processor', 'last_backend', 'key_skills']:
        st.session_state.setdefault(key, [] if key == 'key_skills' else '')

    if 'saved_roles' not in st.session_state:
        st.session_state.saved_roles = get_saved_roles()

    llm_descriptions = {
        "llama": "Created by Meta AI, this is the llama-3.1, 8 billion parameter model. This is the latest and most capable large language model with strong performance on various NLP tasks.",
    }

    api_keys = get_available_api_keys()

    if not api_keys or 'llama' not in api_keys:
        st.error("No API key found for Llama. Please check your configuration.")
        return

    available_backends = ['llama']
    backend_options = [f"{backend}: {llm_descriptions.get(backend, '')}" for backend in available_backends]

    if not backend_options:
        st.error("No compatible backends found. Please check your API key configuration.")
        return

    selected_backend = st.selectbox("Select AI backend:", backend_options, index=0).split(":")[0].strip()

    if selected_backend != st.session_state.get('last_backend'):
        selected_api_key = api_keys.get(selected_backend)
        if not selected_api_key:
            st.error(f"No API key found for {selected_backend}. Please check your configuration.")
            return
    
        st.session_state.resume_processor = create_resume_processor(selected_api_key, selected_backend)
        if hasattr(st.session_state.resume_processor, 'clear_cache'):
            clear_cache()
        st.session_state.last_backend = selected_backend
        logger.debug(f"Switched to {selected_backend} backend and cleared cache")
    else:
        selected_api_key = api_keys.get(selected_backend)
        if not selected_api_key:
            st.error(f"No API key found for {selected_backend}. Please check your configuration.")
            return

    st.session_state.backend = selected_backend
    resume_processor = st.session_state.resume_processor

    # Initialize the LlamaAPI instance here
    llm = initialize_llm()

    if not hasattr(resume_processor, 'analyze_match'):
        raise AttributeError(f"ResumeProcessor for backend '{selected_backend}' does not have the 'analyze_match' method")

    saved_role_options = ["Select a saved role"] + [f"{role['role_name']} - {role['client']}" for role in st.session_state.saved_roles]
    selected_saved_role = st.selectbox("Select a saved role:", options=saved_role_options)

    if selected_saved_role != "Select a saved role":
        selected_role = next(role for role in st.session_state.saved_roles if f"{role['role_name']} - {role['client']}" == selected_saved_role)
        st.session_state.update({
            'job_description': selected_role['job_description'],
            'job_description_link': selected_role['job_description_link'],
            'role_name_input': selected_role['role_name'],
            'client': selected_role['client']
        })

    st.session_state.job_title = st.text_input("Enter the job title:", value=st.session_state.get('job_title', ''))

    jd_option = st.radio("Job Description Input Method:", ("Paste Job Description", "Provide Link to Fractal Job Posting"))

    if jd_option == "Paste Job Description":
        st.session_state.job_description = st.text_area("Paste the Job Description here:", value=st.session_state.get('job_description', ''), placeholder="Job description. This field is required.")
        st.session_state.job_description_link = ""
    else:
        st.session_state.job_description_link = st.text_input("Enter the link to the Fractal job posting:", value=st.session_state.get('job_description_link', ''))

    if not st.session_state.job_description:
        if st.session_state.job_description_link and is_valid_fractal_job_link(st.session_state.job_description_link):
            with st.spinner('Extracting job description...'):
                st.session_state.job_description = extract_job_description(st.session_state.job_description_link)

    if st.session_state.job_description:
        st.text_area("Current Job Description:", value=st.session_state.job_description, height=200, key='current_jd', disabled=True)

    st.session_state.client = st.text_input("Enter the client name:", value=st.session_state.get('client', ''))

    st.subheader("Customize Importance Factors")
    
    # Define the required factors and their defaults
    required_factors = {
        'technical_skills': 0.5,
        'experience': 0.5,
        'education': 0.5,
        'soft_skills': 0.5,
        'industry_knowledge': 0.5
    }

    # Initialize or update importance_factors in session state
    if 'importance_factors' not in st.session_state:
        # Initialize with default values if not present
        st.session_state.importance_factors = {factor: default_value for factor, default_value in required_factors.items()}
    else:
        # Ensure all required factors are present, set missing ones to their default values
        for factor, default_value in required_factors.items():
            st.session_state.importance_factors.setdefault(factor, default_value)

    # Create sliders for importance factors
    cols = st.columns(len(required_factors))
    for i, (factor, default_value) in enumerate(required_factors.items()):
        with cols[i]:
            st.session_state.importance_factors[factor] = st.slider(
                f"{factor.replace('_', ' ').title()}",
                0.0, 1.0,
                st.session_state.importance_factors[factor],
                0.1,
                key=f"slider_{factor}"  # Add a unique key for each slider
            )

    st.write("Current Importance Factors:", st.session_state.importance_factors)

    key_skills = st.text_area("Enter key skills or requirements (one per line):", help="These will be used to assess the candidate's fit for the role.")

    if key_skills:
        st.session_state.key_skills = [skill.strip() for skill in key_skills.split('\n') if skill.strip()]

    if st.session_state.key_skills:
        st.write("Key Skills/Requirements Entered:")
        for skill in st.session_state.key_skills:
            st.write(f"- {skill}")

    st.write("Upload your resume(s) (Maximum 3 files allowed):")
    resume_files = st.file_uploader("Upload resumes (max 3)", type=['pdf', 'docx'], accept_multiple_files=True)

    if resume_files:
        if len(resume_files) > 3:
            st.warning("You've uploaded more than 3 resumes. Only the first 3 will be processed.")
            resume_files = resume_files[:3]
        st.write(f"Number of resumes uploaded: {len(resume_files)}")

    if st.button('Process Resumes', key='process_resumes'):
        if resume_files and st.session_state.job_description and st.session_state.job_title:
            process_resumes_logic(resume_files, resume_processor, llm)
        else:
            if not st.session_state.job_description:
                st.error("Please ensure a job description is provided.")
            if not st.session_state.job_title:
                st.error("Please enter a job title.")
            if not resume_files:
                st.error("Please upload at least one resume.")

    handle_save_role_logic()
    handle_delete_role_logic()

    if st.button("Logout"):
        st.session_state.clear()
        st.experimental_rerun()

def process_resumes_logic(resume_files, resume_processor, llm):
    run_id = str(uuid.uuid4())
    insert_run_log(run_id, "start_analysis", f"Starting analysis for {len(resume_files)} resumes")
    logger.info("Processing resumes...")

    candidates = get_candidate_data()
    candidate_data_list = [next((c for c in candidates if c['candidate'] == file.name), {}) for file in resume_files]

    progress_bar = st.progress(0)
    status_text = st.empty()

    job_requirements = generate_job_requirements(st.session_state.job_description)

    evaluation_results = []
    try:
        with st.spinner('Processing resumes...'):
            for i, (resume_file, candidate_data) in enumerate(zip(resume_files, candidate_data_list)):
                status_text.text(f"Processing resume {i+1} of {len(resume_files)}: {resume_file.name}")
                result = process_resume(
                    resume_file, resume_processor, st.session_state.job_description, st.session_state.importance_factors,
                    candidate_data, st.session_state.job_title, st.session_state.key_skills, llm, job_requirements
                )
                evaluation_results.append(result)
                progress_bar.progress((i + 1) / len(resume_files))

        insert_run_log(run_id, "end_analysis", f"Completed analysis for {len(resume_files)} resumes")

        if evaluation_results:
            st.success("Evaluation complete!")
            display_results(evaluation_results, run_id, save_feedback)
        else:
            st.warning("No resumes were successfully processed. Please check the uploaded files and try again.")
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        logger.error(f"Error during resume processing: {str(e)}", exc_info=True)
    finally:
        progress_bar.empty()
        status_text.empty()


def handle_save_role_logic():
    save_button = False
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
            try:
                if save_role(saved_role_name, client, st.session_state.job_description, st.session_state.job_description_link):
                    st.success(f"Role '{saved_role_name} - {client}' saved successfully!")
                    st.session_state.saved_roles = get_saved_roles()
                else:
                    st.error("Failed to save the role. Please try again.")
            except Exception as e:
                st.error(f"An error occurred while saving the role: {str(e)}")


def handle_delete_role_logic():
    if st.checkbox("Delete a saved role"):
        delete_role_options = [f"{role['role_name']} - {role['client']}" for role in st.session_state.saved_roles]

        if delete_role_options:
            delete_role_name = st.selectbox("Select a role to delete:", options=delete_role_options)

            if delete_role_name:
                if st.button(f"Delete {delete_role_name}"):
                    role_parts = delete_role_name.rsplit(" - ", 2)
                    role_name_to_delete = " - ".join(role_parts[:-1])

                    try:
                        if delete_saved_role(role_name_to_delete):
                            st.success(f"Role '{delete_role_name}' deleted successfully!")
                            st.session_state.saved_roles = get_saved_roles()
                        else:
                            raise Exception("Deletion failed")
                    except Exception as e:
                        st.error(f"Failed to delete role '{delete_role_name}'. Error: {str(e)}")


if __name__ == "__main__":
    main_app()
