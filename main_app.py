import streamlit as st
import uuid
import pandas as pd
from typing import List, Dict, Any
from auth import require_auth, init_auth_state
from resume_processor import resume_processor
from database import save_role, get_saved_roles, save_evaluation_result, get_evaluation_results
from utils import extract_text_from_file, generate_pdf_report
import logging
import re

logger = logging.getLogger(__name__)

def load_css():
    st.markdown("""
    <style>
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

@require_auth
def main():
    load_css()
    st.markdown("<h1 class='main-title'>Resume Cupid 💘</h1>", unsafe_allow_html=True)
    st.markdown("<p class='info-box'>Welcome to Resume Cupid - Your AI-powered resume evaluation tool</p>", unsafe_allow_html=True)

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

    saved_roles = get_saved_roles()
    role_names = [role['role_name'] for role in saved_roles]
    selected_role = st.selectbox("Select a job role", [""] + role_names)

    if selected_role:
        role = next((role for role in saved_roles if role['role_name'] == selected_role), None)
        if role:
            st.markdown("<p class='info-box'>Job Description:</p>", unsafe_allow_html=True)
            sanitized_job_description = sanitize_text(role['job_description'])
            st.text_area("", value=sanitized_job_description, height=200, disabled=True)

            uploaded_files = st.file_uploader("Upload resumes (PDF or DOCX)", accept_multiple_files=True, type=['pdf', 'docx'])

            if uploaded_files:
                if st.button("Evaluate Resumes"):
                    with st.spinner("Evaluating resumes, please wait..."):
                        results = process_resumes(uploaded_files, role['job_description'], role['id'])
                    display_results(results)
        else:
            st.error("Selected job role not found.")

def display_results(results: List[Dict[str, Any]]):
    st.markdown("<h3 class='section-title'>Evaluation Results</h3>", unsafe_allow_html=True)

    if not results:
        st.warning("No results to display.")
        return

    df = pd.DataFrame(results)
    columns_to_display = ['file_name', 'match_score', 'recommendation']
    df = df[[col for col in columns_to_display if col in df.columns]]
    df.columns = ['Resume', 'Match Score', 'Recommendation']
    df = df.sort_values('Match Score', ascending=False)

    st.dataframe(df)

    for result in results:
        with st.expander(f"Detailed Analysis: {result.get('file_name', 'Unknown')}"):
            st.write(f"**Match Score:** {result.get('match_score', 'N/A')}%")
            st.write(f"**Recommendation:** {result.get('recommendation', 'N/A')}")
            st.write(f"**Brief Summary:** {result.get('brief_summary', 'N/A')}")
            st.write(f"**Fit Summary:** {result.get('fit_summary', 'N/A')}")

            if 'skills_analysis' in result:
                st.write("**Skills Analysis:**")
                for skill, score in result['skills_analysis'].items():
                    st.write(f"- {skill}: {score:.2f}")

            if 'experience_analysis' in result:
                st.write("**Experience Analysis:**")
                st.write(f"- Relevance: {result['experience_analysis'].get('relevance', 'N/A')}")
                st.write(f"- Years: {result['experience_analysis'].get('years', 'N/A')}")

            if 'project_analysis' in result:
                st.write("**Project Analysis:**")
                st.write(f"- Complexity: {result['project_analysis'].get('complexity', 'N/A')}")
                st.write(f"- Relevance: {result['project_analysis'].get('relevance', 'N/A')}")

            if 'education_analysis' in result:
                st.write("**Education Analysis:**")
                st.write(f"- Degree Level: {result['education_analysis'].get('degree_level', 'N/A')}")
                st.write(f"- Relevance: {result['education_analysis'].get('relevance', 'N/A')}")

            st.write("**Key Strengths:**")
            for strength in result.get('key_strengths', []):
                st.write(f"- {strength}")

            st.write("**Areas for Improvement:**")
            for area in result.get('areas_for_improvement', []):
                st.write(f"- {area}")

            st.write("**Recommended Interview Questions:**")
            for question in result.get('recruiter_questions', []):
                st.write(f"- {question}")

    if results:
        pdf_report = generate_pdf_report(results, str(uuid.uuid4()))
        st.download_button(
            label="Download PDF Report",
            data=pdf_report,
            file_name="resume_evaluation_report.pdf",
            mime="application/pdf"
        )

def process_resumes(files: List[Any], job_description: str, job_role_id: int) -> List[Dict[str, Any]]:
    results = []

    # Fetch the job role details
    saved_roles = get_saved_roles()
    job_role = next((role for role in saved_roles if role['id'] == job_role_id), None)

    if not job_role:
        st.error(f"Job role with ID {job_role_id} not found.")
        return results

    job_title = job_role['role_name']

    for file in files:
        try:
            logger.debug(f"Processing file: {file.name}, Size: {file.size} bytes, Type: {file.type}")
            resume_text = extract_text_from_file(file)
            if not resume_text.strip():
                st.warning(f"Resume {file.name} is empty or unreadable.")
                continue

            result = resume_processor.process_resume(resume_text, job_description, job_title)
            result['file_name'] = file.name  # Add the file name to the result
            results.append(result)
            save_evaluation_result(file.name, job_role_id, result['match_score'], result['recommendation'])
        except Exception as e:
            logger.error(f"Error processing resume {file.name}: {str(e)}", exc_info=True)
            st.error(f"Error processing resume {file.name}: {str(e)}")
    return results

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
    roles = get_saved_roles()
    for role in roles:
        with st.expander(role['role_name']):
            st.write(role['job_description'])

def view_past_evaluations_page():
    st.markdown("<h2 class='section-title'>View Past Evaluations</h2>", unsafe_allow_html=True)

    saved_roles = get_saved_roles()
    role_names = [role['role_name'] for role in saved_roles]
    selected_role = st.selectbox("Select a job role", [""] + role_names)

    if selected_role:
        role = next((role for role in saved_roles if role['role_name'] == selected_role), None)
        if role:
            results = get_evaluation_results(role['id'])

            if results:
                df = pd.DataFrame(results)
                df = df[['resume_file_name', 'match_score', 'recommendation']]
                df.columns = ['Resume', 'Match Score', 'Recommendation']
                df = df.sort_values('Match Score', ascending=False)

                st.dataframe(df)
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