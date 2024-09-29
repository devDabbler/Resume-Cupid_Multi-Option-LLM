import streamlit as st
import uuid
import pandas as pd
from typing import List, Dict, Any
from auth import require_auth, init_auth_state, auth_page, logout_user
from resume_processor import resume_processor
import logging
import time
import re
import os
from dotenv import load_dotenv
from llm_orchestrator import llm_orchestrator
from llama_service import LlamaService
import plotly.graph_objects as go
from utils import generate_recommendation, generate_fit_summary, extract_text_from_file, generate_pdf_report
from database import (
    get_saved_roles, save_role, save_evaluation_result, get_evaluation_results,
    get_user_profile, update_user_profile, get_job_recommendations, get_latest_evaluation
)
from auth import logout_user

# Load environment variables from the specified .env file
dotenv_file = os.getenv('DOTENV_FILE', '.env.development')
load_dotenv(dotenv_file)

# Initialize logger
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Resume Cupid", page_icon="💘")

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
        margin-bottom: 2rem;
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

def display_job_seeker_profile():
    st.markdown("<h2 class='section-title'>My Profile</h2>", unsafe_allow_html=True)
    
    # Fetch user profile from the database
    user_profile = get_user_profile(st.session_state.user['id'])
    
    # Initialize user_profile if it's None
    if user_profile is None:
        user_profile = {}
    
    # Display and edit profile information
    with st.form("edit_profile"):
        name = st.text_input("Full Name", value=user_profile.get('name', ''))
        email = st.text_input("Email", value=user_profile.get('email', ''))
        phone = st.text_input("Phone", value=user_profile.get('phone', ''))
        skills = st.text_area("Skills (comma-separated)", value=', '.join(user_profile.get('skills', [])))
        experience = st.number_input("Years of Experience", value=user_profile.get('experience', 0), min_value=0, step=1)
        education = st.text_input("Highest Education", value=user_profile.get('education', ''))
        
        if st.form_submit_button("Update Profile"):
            updated_profile = {
                'name': name,
                'email': email,
                'phone': phone,
                'skills': [skill.strip() for skill in skills.split(',')],
                'experience': experience,
                'education': education
            }
            if update_user_profile(st.session_state.user['id'], updated_profile):
                st.success("Profile updated successfully!")
            else:
                st.error("Failed to update profile. Please try again.")

def evaluate_resume_page():
    st.markdown("<h2 class='section-title'>Evaluate My Resume</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    
    if uploaded_file:
        if st.button("Evaluate Resume"):
            with st.spinner("Evaluating your resume, please wait..."):
                resume_text = extract_text_from_file(uploaded_file)
                evaluation_result = resume_processor.process_resume(resume_text, "", "General")  # Using empty job description for general evaluation
                
                st.subheader("Evaluation Results")
                st.write(f"Overall Score: {evaluation_result['match_score']}%")
                st.write(f"Summary: {evaluation_result['brief_summary']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Key Strengths")
                    for strength in evaluation_result['key_strengths']:
                        st.write(f"• {strength}")
                
                with col2:
                    st.subheader("Areas for Improvement")
                    for area in evaluation_result['areas_for_improvement']:
                        st.write(f"• {area}")
                
                st.subheader("Skill Gap Analysis")
                for skill in evaluation_result['skills_gap']:
                    st.write(f"• {skill}")
                
                # Generate and offer PDF report download
                pdf_report = generate_pdf_report([evaluation_result], "resume_evaluation", "General Evaluation")
                st.download_button(
                    label="Download Evaluation Report (PDF)",
                    data=pdf_report,
                    file_name="resume_evaluation_report.pdf",
                    mime="application/pdf"
                )

def job_recommendations_page():
    st.markdown("<h2 class='section-title'>Job Recommendations</h2>", unsafe_allow_html=True)
    
    # Fetch job recommendations based on user's profile and resume
    recommendations = get_job_recommendations(st.session_state.user['id'])
    
    if recommendations:
        for job in recommendations:
            with st.expander(f"{job['title']} - {job['company']}"):
                st.write(f"Match Score: {job['match_score']}%")
                st.write(f"Description: {job['description']}")
                st.write(f"Required Skills: {', '.join(job['required_skills'])}")
                st.write(f"Location: {job['location']}")
                st.write(f"Salary Range: {job['salary_range']}")
                if st.button("Apply", key=f"apply_{job['id']}"):
                    # Implement application logic here
                    st.success("Application submitted successfully!")
    else:
        st.info("No job recommendations available at the moment. Please check back later or update your profile and resume for better matches.")

def improvement_suggestions_page():
    st.markdown("<h2 class='section-title'>Improvement Suggestions</h2>", unsafe_allow_html=True)
    
    # Fetch the user's latest resume evaluation
    latest_evaluation = get_latest_evaluation(st.session_state.user['id'])
    
    if latest_evaluation:
        st.subheader("Based on your latest resume evaluation:")
        
        # Check if 'skills_gap' exists in the evaluation
        if 'skills_gap' in latest_evaluation and latest_evaluation['skills_gap']:
            st.write("Skills to Improve:")
            for skill in latest_evaluation['skills_gap']:
                st.write(f"• {skill}")
                # Provide resources or courses to improve each skill
                st.write("  Suggested resources:")
                st.write("  - Online course: [Course Name](course_link)")
                st.write("  - Tutorial: [Tutorial Name](tutorial_link)")
        else:
            st.write("No specific skills gap identified in your latest evaluation.")
        
        # Check if 'areas_for_improvement' exists in the evaluation
        if 'areas_for_improvement' in latest_evaluation and latest_evaluation['areas_for_improvement']:
            st.write("Experience Suggestions:")
            for area in latest_evaluation['areas_for_improvement']:
                st.write(f"• {area}")
                # Provide suggestions to gain relevant experience
                st.write("  How to gain experience:")
                st.write("  - Volunteer for projects in this area")
                st.write("  - Contribute to open-source projects")
                st.write("  - Take on freelance work to build portfolio")
        else:
            st.write("No specific areas for improvement identified in your latest evaluation.")
        
        st.write("Resume Writing Tips:")
        # Provide general resume writing tips
        tips = [
            "Use action verbs to describe your accomplishments",
            "Quantify your achievements with numbers and percentages",
            "Tailor your resume to each job application",
            "Keep your resume concise and to the point",
            "Proofread carefully for grammar and spelling errors"
        ]
        for tip in tips:
            st.write(f"• {tip}")
    else:
        st.info("No recent resume evaluation found. Please evaluate your resume to get personalized improvement suggestions.")

    # Add logging to help debug
    logger.info(f"Latest evaluation data: {latest_evaluation}")

def verify_email():
    token = st.query_params.get('token')
    if token:
        if verify_user_email(token):
            st.success("Your email has been verified successfully. You can now log in.")
        else:
            st.error("Invalid or expired verification token.")
    else:
        st.error("No verification token provided.")

def reset_password_page():
    token = st.query_params.get('token')
    if token:
        st.title("Reset Your Password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        if st.button("Set New Password"):
            if new_password == confirm_password:
                if update_password_with_token(token, new_password):
                    st.success("Your password has been reset successfully. You can now log in with your new password.")
                else:
                    st.error("Failed to reset password. The token may be invalid or expired.")
            else:
                st.error("Passwords do not match. Please try again.")
    else:
        st.error("Invalid password reset link.")

def main():
    load_css()
    
    # Check for special routes first
    query_params = st.query_params
    if 'page' in query_params:
        if query_params['page'][0] == 'verify_email':
            verify_email()
            return
        elif query_params['page'][0] == 'reset_password':
            reset_password_page()
            return

    # Continue with the regular flow
    if not st.session_state.get('user'):
        auth_page()
    else:
        st.markdown("<h1 class='main-title'>Resume Cupid 💘</h1>", unsafe_allow_html=True)
        st.markdown("<p class='info-box'>Welcome to Resume Cupid - Your AI-powered career assistant</p>", unsafe_allow_html=True)

        # Sidebar
        st.sidebar.title(f"Welcome, {st.session_state.user['username']}")
        
        logged_in_user_type = st.session_state.get('logged_in_user_type')

        if logged_in_user_type == 'job_seeker':
            menu = ["My Profile", "Evaluate Resume", "Job Recommendations", "Improvement Suggestions"]
        else:  # employer
            menu = ["Evaluate Resumes", "Manage Job Roles", "View Past Evaluations"]

        choice = st.sidebar.selectbox("Menu", menu)

        # Add logout button to sidebar
        with st.sidebar.container():
            st.markdown('<div class="logout-button">', unsafe_allow_html=True)
            if st.button("Logout"):
                logout_user()
                st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Rest of your main function remains the same
        if logged_in_user_type == 'job_seeker':
            if choice == "My Profile":
                display_job_seeker_profile()
            elif choice == "Evaluate Resume":
                evaluate_resume_page()
            elif choice == "Job Recommendations":
                job_recommendations_page()
            elif choice == "Improvement Suggestions":
                improvement_suggestions_page()
        else:  # employer
            if choice == "Evaluate Resumes":
                evaluate_resumes_page()
            elif choice == "Manage Job Roles":
                manage_job_roles_page()
            elif choice == "View Past Evaluations":
                view_past_evaluations_page()

def job_seeker_menu():
    menu = ["My Profile", "Evaluate My Resume", "Job Recommendations", "Improvement Suggestions"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "My Profile":
        display_job_seeker_profile()
    elif choice == "Evaluate My Resume":
        evaluate_resume_page()
    elif choice == "Job Recommendations":
        job_recommendations_page()
    elif choice == "Improvement Suggestions":
        improvement_suggestions_page()

def employer_menu():
    menu = ["Evaluate Resumes", "Manage Job Roles", "View Past Evaluations"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Evaluate Resumes":
        evaluate_resumes_page()
    elif choice == "Manage Job Roles":
        manage_job_roles_page()
    elif choice == "View Past Evaluations":
        view_past_evaluations_page()

def evaluate_resumes_page():
    st.markdown("<h2 class='section-title'>Evaluate Resumes</h2>", unsafe_allow_html=True)

    saved_roles = get_saved_roles()  # Remove db_conn argument
    role_names = [role['role_name'] for role in saved_roles]
    selected_role = st.selectbox("Select a job role", [""] + role_names)

    if selected_role:
        role = next((role for role in saved_roles if role['role_name'] == selected_role), None)
        if role:
            st.markdown("<p class='info-box'>Job Description:</p>", unsafe_allow_html=True)
            sanitized_job_description = sanitize_text(role['job_description'])
            st.text_area("", value=sanitized_job_description, height=200, key="job_description")

            uploaded_files = st.file_uploader("Upload Resumes", accept_multiple_files=True, type=["pdf", "docx"])
            
            if uploaded_files:
                if st.button("Evaluate Resumes"):
                    with st.spinner("Evaluating resumes, please wait..."):
                        job_description = role['job_description']
                        job_role_id = role['id']
                        job_title = role['role_name'].replace("- C3", "").strip()  # Standardize job title here
                        results = process_resumes(uploaded_files, job_description, job_role_id, job_title)
                        
                        if results:
                            display_results(results, job_title=job_title)
                        else:
                            st.warning("No valid results to display. Please check the error messages above.")
            else:
                st.warning("Please upload resumes to evaluate.")
        else:
            st.error("Selected job role not found.")

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
        job_description = st.text_area("Job Description")
        submitted = st.form_submit_button("Save Job Role")

        if submitted:
            if role_name.strip() and job_description.strip():
                sanitized_role_name = sanitize_text(role_name)
                sanitized_job_description = sanitize_text(job_description)
                if save_role(sanitized_role_name, sanitized_job_description):
                    st.success("Job role saved successfully!")
                else:
                    st.error("Failed to save job role. Please try again.")
            else:
                st.warning("Please provide both role name and job description.")

    st.markdown("<h3 class='section-title'>Existing Job Roles</h3>", unsafe_allow_html=True)
    roles = get_saved_roles()  # Remove db_conn argument
    for role in roles:
        with st.expander(role['role_name']):
            st.write(role['job_description'])

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
    init_auth_state()
    main()
