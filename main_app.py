import streamlit as st
import uuid
import pandas as pd
from typing import List, Dict, Any
from auth import Authenticator, require_auth, init_auth_state, auth_page
from resume_processor import resume_processor
import logging
import time
import re
import os
import textwrap
from dotenv import load_dotenv
from llm_orchestrator import llm_orchestrator
from llama_service import LlamaService
import plotly.graph_objects as go
from utils import generate_recommendation, generate_fit_summary, extract_text_from_file, generate_pdf_report, custom_notification
from database import (
    get_saved_roles, save_role, save_evaluation_result, get_evaluation_results,
    get_user_profile, update_user_profile, get_user_resumes, save_user_resume, get_job_recommendations, get_latest_evaluation,
    get_all_verification_tokens, verify_user_email, reset_user_password, delete_role, get_user_by_reset_token, update_password_with_token)


# Load environment variables from the specified .env file
dotenv_file = os.getenv('DOTENV_FILE', '.env.development')
load_dotenv(dotenv_file)

# Initialize logger
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Resume Cupid", page_icon="💘")

# Initialize Authenticator
authenticator = Authenticator()
    
def load_css():
    """Load custom CSS for styling Streamlit elements."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        color: #3366cc;
        margin-bottom: 0;
    }
    .welcome-message {
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
        background-color: #e6f2ff;
        padding: 1rem;
        border-radius: 5px;
    }
    .section-title {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f4e79;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e6f2ff;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #3366cc;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def display_dynamic_welcome_message(page):
    """Display dynamic welcome messages based on the selected page."""
    if page == "Evaluate Resumes":
        return """<h4>Getting Started</h4>
<ul>
    <li>Select an existing job role from the dropdown.</li>
    <li>Or add new job roles via the 'Manage Job Roles' page.</li>
    <li>Use the 'Evaluate Resumes' feature to assess candidates against your job descriptions.</li>
    <li>View past evaluations and manage your job roles.</li>
</ul>
<p>Let's get started by adding or selecting your job roles!</p>
"""
    elif page == "Manage Job Roles":
        return """<h4>Getting Started</h4>
<ul>
    <li>Create new job roles by entering the role name, client, description, and key requirements.</li>
    <li>Update or delete existing roles as needed to keep your job listings current and relevant.</li>
</ul>
"""
    elif page == "View Past Evaluations":
        return """<h4>Getting Started</h4>
<p>Here, you can access past assessments of candidates to:</p>
<ul>
    <li>Refer back to earlier evaluations</li>
    <li>Make informed hiring decisions</li>
    <li>Revisit key candidate information when needed</li>
</ul>
"""
    return ""

def main():
    try:
        load_css()
        init_auth_state()

        query_params = st.query_params
        action = query_params.get('action', [None])[0]
        token = query_params.get('token', [None])[0]

        if action == "verify_email" and token:
            handle_email_verification(token)
        elif action == "reset_password" and token:
            reset_password_page(token)
        elif not st.session_state.get('user'):
            if st.session_state.get('password_reset_success'):
                st.success("Your password has been reset successfully. Please log in with your new password.")
                st.session_state.pop('password_reset_success', None)
            auth_page()
        else:
            logged_in_user_type = st.session_state.get('logged_in_user_type')
            display_main_app(logged_in_user_type)

    except Exception as e:
        logger.error(f"An error occurred in main(): {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please check the logs for more information.")

def handle_email_verification(token):
    st.title("Email Verification")
    if verify_user_email(token):
        st.success("Your email has been successfully verified! You can now log in.")
        if st.button("Go to Login"):
            st.query_params.clear()
            st.rerun()
    else:
        st.error("Email verification failed. The link may be invalid or expired.")

def handle_password_reset(token):
    st.title("Reset Password")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Reset Password"):
        if new_password and new_password == confirm_password:
            if reset_user_password(token, new_password):
                st.success("Your password has been successfully reset. You can now log in with your new password.")
                st.query_params.clear()
                st.rerun()
            else:
                st.error("Failed to reset password. The link may be invalid or expired.")
        else:
            st.error("Passwords do not match or are empty.")

def verify_email_page(token):
    st.title("Email Verification")
    logger.info(f"Received verification request with token: {token}")

    all_tokens = get_all_verification_tokens()
    logger.info(f"All verification tokens in database: {all_tokens}")

    if authenticator.verify_email(token):
        logger.info(f"Email verified successfully with token: {token}")
        st.success("Your email has been successfully verified! You can now log in.")
        if st.button("Go to Login"):
            st.query_params.clear()
            st.rerun()
    else:
        logger.warning(f"Email verification failed for token: {token}")
        st.error("Email verification failed. The link may be invalid or expired. Please try registering again or contact support.")

def reset_password_page(token):
    st.title("Reset Password")
    
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    if st.button("Reset Password"):
        if new_password and new_password == confirm_password:
            if update_password_with_token(token, new_password):
                st.success("Your password has been successfully reset. You can now log in with your new password.")
                st.session_state.password_reset_success = True
                time.sleep(2)
                st.experimental_rerun()
            else:
                st.error("Failed to reset password. The link may be invalid or expired.")
        else:
            st.error("Passwords do not match or are empty.")

def display_main_app(logged_in_user_type):
    st.sidebar.markdown(
        f"<h3 style='margin-bottom: 30px; font-size: 1.5rem;'>Welcome, {st.session_state.user['username']}</h3>",
        unsafe_allow_html=True
    )

    menu = ["My Profile", "Evaluate Resume", "Job Recommendations", "Improvement Suggestions"] if logged_in_user_type == 'job_seeker' else ["Evaluate Resumes", "Manage Job Roles", "View Past Evaluations"]

    st.sidebar.markdown("<label style='font-size: 1.2rem;'>Navigate to</label>", unsafe_allow_html=True)
    choice = st.sidebar.selectbox("", menu, key="navigation_menu")

    if st.sidebar.button("Logout"):
        authenticator.logout()
        st.rerun()

    content_message = display_dynamic_welcome_message(choice)
    st.markdown(f"""<div style="background-color: #f1f3f6; padding: 20px 20px 5px 20px; border-radius: 10px; margin-bottom: 10px;">
        <h1 style="color: #3f51b5; margin-bottom: 5px;">Welcome to Resume Cupid 💘!</h1>
        <h3 style="color: #3f51b5; margin-top: 0; margin-bottom: 15px;">Your AI-powered hiring assistant.</h3>
        {content_message}
        </div>
    """, unsafe_allow_html=True)

    if logged_in_user_type == 'job_seeker':
        job_seeker_menu(choice)
    else:
        employer_menu(choice)

def evaluate_resume_page():
    st.title("Evaluate Resume")
    
    # Get the current user's ID
    user_id = st.session_state['user']['id']
    
    # File uploader for resume
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])
    
    if uploaded_file is not None:
        # Extract text from the uploaded file
        resume_text = extract_text_from_file(uploaded_file)
        
        # Get saved job roles (assuming you want to evaluate against a specific job)
        saved_roles = get_saved_roles()
        role_names = [role['role_name'] for role in saved_roles]
        selected_role = st.selectbox("Select a job role to evaluate against", [""] + role_names)
        
        if selected_role:
            role = next((role for role in saved_roles if role['role_name'] == selected_role), None)
            if role:
                if st.button("Evaluate Resume"):
                    with st.spinner("Evaluating resume..."):
                        result = llm_orchestrator.analyze_resume(resume_text, role['job_description'], role['role_name'])
                        
                        # Display results (you can customize this part)
                        st.write(f"Match Score: {result.get('match_score', 'N/A')}")
                        st.write(f"Summary: {result.get('summary', 'N/A')}")
                        # Add more result display as needed
                        
                        # Save the evaluation result
                        save_evaluation_result(uploaded_file.name, role['id'], result.get('match_score', 0), result.get('summary', 'N/A'))
            else:
                st.error("Selected job role not found.")
        else:
            st.info("Please select a job role to evaluate against.")
        
def job_seeker_menu(choice):
    user_id = st.session_state['user']['id']
    if choice == "My Profile":
        display_job_seeker_profile(user_id)
    elif choice == "Evaluate Resume":
        evaluate_resume_page()
    elif choice == "Job Recommendations":
        job_recommendations_page()
    elif choice == "Improvement Suggestions":
        improvement_suggestions_page()

def employer_menu(choice):
    if choice == "Evaluate Resumes":
        evaluate_resume_page()
    elif choice == "Manage Job Roles":
        manage_job_roles_page()
    elif choice == "View Past Evaluations":
        view_past_evaluations_page()

def display_job_seeker_profile(user_id: int):
    """
    Displays the job seeker's profile information in a Streamlit app.
    
    Parameters:
    - user_id (int): The ID of the job seeker to display.
    """
    st.title("Job Seeker Profile")

    # Retrieve the user's profile data
    user_profile = get_user_profile(user_id)
    if not user_profile:
        st.error("Could not retrieve profile data. Please ensure you are logged in and your profile is complete.")
        return

    # Display user information
    st.subheader("Personal Information")
    st.write(f"**Name:** {user_profile.get('name', 'Not Available')}")
    st.write(f"**Email:** {user_profile.get('email', 'Not Available')}")
    if 'location' in user_profile:
        st.write(f"**Location:** {user_profile.get('location', 'Not Available')}")
    
    # Additional fields (skills, summary, etc.)
    if 'skills' in user_profile:
        st.subheader("Skills")
        st.write(", ".join(user_profile['skills']) if user_profile['skills'] else "No skills listed.")
    
    if 'summary' in user_profile:
        st.subheader("Profile Summary")
        st.write(user_profile['summary'] or "No summary provided.")
    
    # Display user resumes
    st.subheader("Uploaded Resumes")
    resumes = get_user_resumes(user_id)
    if resumes:
        for resume in resumes:
            st.write(f"**File Name:** {resume['file_name']}")
            st.write(f"**Uploaded On:** {resume['created_at']}")
            with st.expander("View Resume Content"):
                st.write(resume['content'][:500] + "..." if len(resume['content']) > 500 else resume['content'])
    else:
        st.write("No resumes uploaded.")
    
    # Display evaluation result if available
    st.subheader("Latest Evaluation Result")
    evaluation = get_latest_evaluation(user_id)
    if evaluation:
        st.write(f"**Resume Evaluated:** {evaluation['resume_file_name']}")
        st.write(f"**Match Score:** {evaluation['match_score']}")
        st.write(f"**Recommendation:** {evaluation['recommendation']}")
        st.write(f"**Skills Gap:** {', '.join(evaluation['skills_gap'])}")
        st.write(f"**Areas for Improvement:** {', '.join(evaluation['areas_for_improvement'])}")
    else:
        st.write("No evaluation results found.")
        
def job_recommendations_page():
    st.title("Job Recommendations")

    user_id = st.session_state['user']['id']
    evaluations = get_evaluation_results(user_id)
    
    if not evaluations:
        st.warning("No evaluations found. Please upload a resume for evaluation first.")
        return

    # Display job recommendations based on evaluation results
    recommendations = get_job_recommendations(user_id)
    if not recommendations:
        st.info("No job recommendations available at the moment. Please check back later.")
        return

    for recommendation in recommendations:
        st.subheader(f"Job Title: {recommendation['title']}")
        st.write(f"**Company:** {recommendation.get('company_name', 'N/A')}")
        st.write(f"**Location:** {recommendation.get('location', 'N/A')}")
        st.write(f"**Match Score:** {recommendation.get('match_score', 'N/A')}")
        st.markdown("---")

def improvement_suggestions_page():
    st.title("Improvement Suggestions")

    user_id = st.session_state['user']['id']
    latest_evaluation = get_latest_evaluation(user_id)

    if not latest_evaluation:
        st.warning("No evaluation found. Please upload and evaluate a resume first.")
        return

    st.write("**Areas for Improvement**")
    if 'areas_for_improvement' in latest_evaluation:
        for area in latest_evaluation['areas_for_improvement']:
            st.write(f"- {area}")
    else:
        st.write("No specific areas for improvement found.")

    st.write("**Skills Gap**")
    if 'skills_gap' in latest_evaluation:
        for skill in latest_evaluation['skills_gap']:
            st.write(f"- {skill}")
    else:
        st.write("No skills gap found.")

def process_resumes(files: List[Any], job_description: str, job_role_id: int, job_title: str) -> List[Dict[str, Any]]:
    results = []
    
    for file in files:
        try:
            logger.info(f"Starting to process resume: {file.name}")
            resume_text = extract_text_from_file(file)
            if not resume_text.strip():
                logger.warning(f"Resume {file.name} is empty or unreadable.")
                st.warning(f"Resume {file.name} is empty or unreadable.")
                continue

            result = llm_orchestrator.analyze_resume(resume_text, job_description, job_title)

            if isinstance(result, dict) and "error" in result:
                logger.error(f"Error processing {file.name}: {result['error']}")
                st.error(f"Error processing {file.name}: {result['error']}")
                continue

            result['file_name'] = file.name
            if save_evaluation_result(file.name, job_role_id, result.get('match_score', 0), result.get('summary', 'N/A')):
                logger.info(f"Saved evaluation result for {file.name}")
                results.append(result)
                st.success(f"Successfully processed resume: {file.name}")
            else:
                logger.error(f"Failed to save evaluation result for {file.name}")
                st.error(f"Failed to save evaluation result for {file.name}")
        except Exception as e:
            logger.error(f"Unexpected error processing resume {file.name}: {str(e)}", exc_info=True)
            st.error(f"Unexpected error processing resume {file.name}: {str(e)}")

    return results

def generate_recommendation(match_score: int) -> str:
    """Generate recommendation based on match score."""
    if match_score >= 90:
        return "Strongly recommend for immediate interview"
    elif 80 <= match_score < 90:
        return "Highly recommend for interview"
    elif 70 <= match_score < 80:
        return "Recommend for interview"
    elif 60 <= match_score < 70:
        return "Consider for interview with reservations"
    elif 50 <= match_score < 60:
        return "Potentially consider for interview, but significant gaps exist"
    else:
        return "Do not recommend for interview at this time"

def generate_fit_summary(match_score: int, job_title: str) -> str:
    """Generate a fit summary based on match score and job title."""
    if match_score >= 90:
        return f"The candidate is an exceptional fit for the {job_title} role, exceeding most job requirements and demonstrating outstanding qualifications."
    elif 80 <= match_score < 90:
        return f"The candidate is an excellent fit for the {job_title} role, meeting or exceeding most job requirements with minor areas for improvement."
    elif 70 <= match_score < 80:
        return f"The candidate is a good fit for the {job_title} role, meeting many of the job requirements with some areas for development."
    elif 60 <= match_score < 70:
        return f"The candidate shows potential for the {job_title} role but has notable gaps that would require further assessment and development."
    elif 50 <= match_score < 60:
        return f"The candidate has some relevant skills for the {job_title} role, but significant gaps exist that may hinder their immediate success."
    else:
        return f"The candidate is not a strong fit for the {job_title} role, with considerable gaps in required skills and experience."

def display_results(results: List[Dict[str, Any]], job_title: str):
    st.markdown("<h2 class='section-title'>Evaluation Results</h2>", unsafe_allow_html=True)

    if not results:
        st.warning("No results to display.")
        return

    # Sort results by match score in descending order
    sorted_results = sorted(results, key=lambda x: x.get('match_score', 0), reverse=True)

    # Create DataFrame with rank
    df = pd.DataFrame(sorted_results)
    df['Rank'] = range(1, len(df) + 1)
    df['Recommendation'] = df['match_score'].apply(generate_recommendation)
    
    columns_to_display = ['Rank', 'file_name', 'match_score', 'Recommendation']
    df = df[columns_to_display]
    df.columns = ['Rank', 'Resume', 'Match Score', 'Recommendation']

    st.dataframe(df)

    def extract_score(rating):
        match = re.search(r'\((\d+)', rating)
        return int(match.group(1)) if match else 0

    for result in sorted_results:
        with st.expander(f"Detailed Analysis: {result.get('file_name', 'Unknown')}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Candidate Information")
                st.write(f"**Name:** {result['file_name']}")
                st.write(f"**Recommendation:** {generate_recommendation(result['match_score'])}")
                st.write(f"**Fit Summary:** {generate_fit_summary(result['match_score'], job_title)}")
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['match_score'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Match Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 85
                        }
                    }
                ))
                st.plotly_chart(fig)

            st.subheader("Summary")
            st.write(result.get('summary', 'N/A'))

            st.subheader("Experience Relevance")
            experience = result.get('experience_relevance', {})
            relevance_data = {
                'High': [],
                'Medium': [],
                'Low': []
            }

            for job, details in experience.items():
                if isinstance(details, dict):
                    for project, rating in details.items():
                        score = extract_score(rating)
                        if score >= 8:
                            relevance_data['High'].append((f"{job} - {project}", score))
                        elif 6 <= score < 8:
                            relevance_data['Medium'].append((f"{job} - {project}", score))
                        else:
                            relevance_data['Low'].append((f"{job} - {project}", score))
                else:
                    score = extract_score(details)
                    if score >= 8:
                        relevance_data['High'].append((job, score))
                    elif 6 <= score < 8:
                        relevance_data['Medium'].append((job, score))
                    else:
                        relevance_data['Low'].append((job, score))

            for relevance, data in relevance_data.items():
                if data:
                    st.write(f"**{relevance} Relevance Experience:**")
                    for item, score in data:
                        st.write(f"- {item}")
                        st.progress(score / 10)
                    st.write("")  # Add some space between relevance categories

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Key Strengths")
                for strength in result.get('key_strengths', []):
                    st.write(f"- {strength}")

            with col2:
                st.subheader("Areas for Improvement")
                for area in result.get('areas_for_improvement', []):
                    st.write(f"- {area}")

            st.subheader("Skills Gap")
            for skill in result.get('skills_gap', []):
                st.write(f"- {skill}")

            st.subheader("Recruiter Questions")
            for i, question in enumerate(result.get('recruiter_questions', []), 1):
                st.write(f"**Question {i}:** {question['question']}")
                st.write(f"**Purpose:** {question['purpose']}")
                st.write("")  # Add a blank line for spacing

    if results:
        pdf_report = generate_pdf_report(sorted_results, str(uuid.uuid4()), job_title)
        st.download_button(
            label="Download PDF Report",
            data=pdf_report,
            file_name="resume_evaluation_report.pdf",
            mime="application/pdf"
        )

def manage_job_roles_page():
    st.markdown("<h2 class='section-title'>Manage Job Roles</h2>", unsafe_allow_html=True)

    with st.form("add_job_role"):
        role_name = st.text_input("Role Name")
        client = st.text_input("Client Name")
        job_description = st.text_area("Job Description")
        submitted = st.form_submit_button("Save Job Role")

        if submitted:
            if role_name.strip() and job_description.strip() and client.strip():
                sanitized_role_name = sanitize_text(role_name)
                sanitized_client = sanitize_text(client)
                sanitized_job_description = sanitize_text(job_description)
                if save_role(sanitized_role_name, sanitized_client, sanitized_job_description):
                    st.success("Job role saved successfully!")
                else:
                    st.error("Failed to save job role. Please try again.")
            else:
                st.warning("Please provide role name, client, and job description.")

    st.markdown("<h3 class='section-title'>Existing Job Roles</h3>", unsafe_allow_html=True)
    roles = get_saved_roles()
    for role in roles:
        with st.expander(role['role_name']):
            st.write(f"Client: {role.get('client', 'N/A')}")
            st.write(role['job_description'])
            if st.button(f"Delete Role: {role['role_name']}", key=f"delete_{uuid.uuid4()}"):
                if delete_role(role['id']):
                    st.success(f"Role '{role['role_name']}' deleted successfully.")
                else:
                    st.error(f"Failed to delete role '{role['role_name']}'. Please try again.")

def view_past_evaluations_page():
    st.markdown("<h2 class='section-title'>View Past Evaluations</h2>", unsafe_allow_html=True)

    saved_roles = get_saved_roles()  # Remove db_conn argument
    role_names = [role['role_name'] for role in saved_roles]
    selected_role = st.selectbox("Select a job role", [""] + role_names)

    if selected_role:
        role = next((role for role in saved_roles if role['role_name'] == selected_role), None)
        if role:
            results = get_evaluation_results(role['id'])  # Remove db_conn argument if it was present

            if results:
                df = pd.DataFrame(results)
                df = df[['id', 'resume_file_name', 'match_score', 'recommendation', 'created_at']]
                df.columns = ['ID', 'Resume', 'Match Score', 'Recommendation', 'Evaluation Date']
                df['Evaluation Date'] = pd.to_datetime(df['Evaluation Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
                df = df.sort_values('Evaluation Date', ascending=False)

                # Add 'Select' column as the first column
                df.insert(0, 'Select', False)

                st.markdown("""
                    <style>
                    .scrollable-table {
                        max-height: 400px;
                        overflow-y: auto;
                        overflow-x: auto;
                    }
                    </style>
                """, unsafe_allow_html=True)

                st.markdown('<div class="scrollable-table">', unsafe_allow_html=True)
                
                # Use data_editor with column_order to ensure 'Select' is the first column
                edited_df = st.data_editor(
                    df,
                    hide_index=True,
                    column_order=['Select', 'ID', 'Resume', 'Match Score', 'Recommendation', 'Evaluation Date'],
                    column_config={
                        "Select": st.column_config.CheckboxColumn("Select", default=False),
                        "ID": st.column_config.NumberColumn("ID", format="%d"),
                        "Match Score": st.column_config.NumberColumn("Match Score", format="%.2f%%"),
                    },
                    disabled=["ID", "Resume", "Match Score", "Recommendation", "Evaluation Date"],
                    width=1000
                )
                
                st.markdown('</div>', unsafe_allow_html=True)

                # Get selected rows
                selected_ids = edited_df[edited_df['Select']]['ID'].tolist()

                # Add download button for CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{selected_role}_evaluations.csv",
                    mime="text/csv",
                )

                # Add download button for PDF report
                if st.button("Generate PDF Report"):
                    if selected_ids:
                        # Create a dictionary mapping ID to Resume (file_name)
                        id_to_resume = dict(zip(df['ID'], df['Resume']))
                        
                        # Prepare selected results with file_name included
                        selected_results = []
                        for result in results:
                            if result['id'] in selected_ids:
                                result_copy = result.copy()
                                result_copy['file_name'] = id_to_resume[result['id']]
                                selected_results.append(result_copy)
                        
                        pdf_report = generate_pdf_report(selected_results, str(uuid.uuid4()), selected_role)
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_report,
                            file_name=f"{selected_role}_selected_evaluation_report.pdf",
                            mime="application/pdf",
                        )
                    else:
                        st.warning("Please select at least one evaluation to generate a PDF report.")
            else:
                st.info("No past evaluations found for this role.")
        else:
            st.error("Selected job role not found.")

def sanitize_text(text: str) -> str:
    """Sanitize text to prevent XSS attacks and unwanted HTML rendering."""
    sanitized = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    sanitized = re.sub(r'[^A-Za-z0-9 .,!?@#&()\-\'\"]+', ' ', sanitized)  # Remove special characters
    return sanitized.strip()

if __name__ == "__main__":
    logger.info("Script started")
    main()
    logger.info("Script ended")