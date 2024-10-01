import logging
import io
import pypdf
from docx import Document
import re
import spacy
from collections import Counter
from typing import List, Dict, Any
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
import streamlit as st

logger = logging.getLogger(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_md")

def custom_notification(message, type="info", duration=5):
    colors = {
        "info": "#4e73df",
        "success": "#1cc88a",
        "warning": "#f6c23e",
        "error": "#e74a3b"
    }
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    
    if type not in colors:
        type = "info"
    
    notification_key = f"notification_{hash(message)}_{type}"
    
    notification_placeholder = st.empty()
    notification_placeholder.markdown(f"""
    <div style="
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border: 1px solid {colors[type]};
        background-color: {colors[type]}22;
        color: {colors[type]};
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: space-between;">
        <span>{icons[type]} {message}</span>
        <button onclick="this.parentElement.style.display='none';" style="
            background: none;
            border: none;
            color: {colors[type]};
            cursor: pointer;
            font-size: 1.2em;
            font-weight: bold;">
            ×
        </button>
    </div>
    """, unsafe_allow_html=True)
    
    if duration > 0:
        import threading
        threading.Timer(duration, notification_placeholder.empty).start()
        
def extract_text_from_file(file) -> str:
    if file is None:
        raise ValueError("File object is None.")
    
    file_content = file.read()
    file.seek(0)  # Reset file pointer to the beginning
    
    print(f"File content size: {len(file_content)} bytes")
    
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

def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        print(f"PDF content size: {len(file_content)} bytes")
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_content))
        print(f"Number of pages in PDF: {len(pdf_reader.pages)}")
        if len(pdf_reader.pages) == 0:
            raise ValueError("PDF file has no pages.")
        text = "".join([page.extract_text() for page in pdf_reader.pages])
        if not text.strip():
            raise ValueError("Extracted text is empty.")
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_docx(file_content: bytes) -> str:
    try:
        doc = Document(io.BytesIO(file_content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        raise ValueError(f"Failed to extract text from DOCX: {str(e)}")

def generate_job_requirements(job_description: str) -> Dict[str, Any]:
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

def generate_recommendation(match_score: int) -> str:
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

def extract_score(rating):
    match = re.search(r'\((\d+)', rating)
    return int(match.group(1)) if match else 0

def generate_pdf_report(results: List[Dict[str, Any]], run_id: str, job_title: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER))

    elements.append(Paragraph(f"Evaluation Report for {job_title}", styles['Title']))
    elements.append(Spacer(1, 12))

    # Summary table
    summary_data = [["Rank", "Resume", "Match Score", "Recommendation"]]
    for i, result in enumerate(sorted(results, key=lambda x: x['match_score'], reverse=True), 1):
        summary_data.append([
            str(i),
            result['file_name'],
            f"{result['match_score']}%",
            generate_recommendation(result['match_score'])
        ])

    summary_table = Table(summary_data, colWidths=[40, 200, 70, 180])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 12))

    for result in sorted(results, key=lambda x: x['match_score'], reverse=True):
        elements.append(Paragraph(f"Detailed Analysis: {result['file_name']}", styles['Heading1']))
        elements.append(Spacer(1, 6))

        # Candidate Information
        info_data = [
            ["Match Score", f"{result['match_score']}%"],
            ["Recommendation", generate_recommendation(result['match_score'])],
            ["Fit Summary", generate_fit_summary(result['match_score'], job_title)]
        ]
        info_table = Table(info_data, colWidths=[100, 400])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Summary", styles['Heading2']))
        elements.append(Paragraph(result.get('summary', 'N/A'), styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Experience Relevance", styles['Heading2']))
        experience = result.get('experience_relevance', {})
        for relevance in ['High', 'Medium', 'Low']:
            elements.append(Paragraph(f"{relevance} Relevance Experience", styles['Heading3']))
            data = []
            for job, details in experience.items():
                if isinstance(details, dict):
                    for project, rating in details.items():
                        score = extract_score(rating)
                        if (relevance == 'High' and score >= 8) or \
                           (relevance == 'Medium' and 6 <= score < 8) or \
                           (relevance == 'Low' and score < 6):
                            data.append([f"{job} - {project}", f"{score}/10"])
                else:
                    score = extract_score(details)
                    if (relevance == 'High' and score >= 8) or \
                       (relevance == 'Medium' and 6 <= score < 8) or \
                       (relevance == 'Low' and score < 6):
                        data.append([job, f"{score}/10"])
            
            if data:
                t = Table(data, colWidths=[400, 50])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.beige),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(t)
            elements.append(Spacer(1, 6))

        # Key Strengths and Areas for Improvement
        elements.append(Paragraph("Key Strengths and Areas for Improvement", styles['Heading2']))
        strengths_improvements = [["Key Strengths", "Areas for Improvement"]]
        strengths = result.get('key_strengths', [])
        improvements = result.get('areas_for_improvement', [])
        max_len = max(len(strengths), len(improvements))
        for i in range(max_len):
            strengths_improvements.append([
                strengths[i] if i < len(strengths) else "",
                improvements[i] if i < len(improvements) else ""
            ])
        si_table = Table(strengths_improvements, colWidths=[250, 250])
        si_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(si_table)
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Skills Gap", styles['Heading2']))
        for skill in result.get('skills_gap', []):
            elements.append(Paragraph(f"• {skill}", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Recruiter Questions", styles['Heading2']))
        for i, question in enumerate(result.get('recruiter_questions', []), 1):
            elements.append(Paragraph(f"{i}. {question['question']}", styles['Normal']))
            elements.append(Paragraph(f"Purpose: {question['purpose']}", ParagraphStyle('Italic', parent=styles['Normal'], fontName='Helvetica-Oblique')))
            elements.append(Spacer(1, 6))

        elements.append(PageBreak())

    doc.build(elements)
    return buffer.getvalue()

def generate_resume_summary(result: Dict[str, Any], job_title: str) -> str:
    return f"""
    <div class="resume-summary">
        <h2>Detailed Analysis: {result['file_name']}</h2>
        <p><strong>Match Score:</strong> {result['match_score']}%</p>
        <p><strong>Recommendation:</strong> {generate_recommendation(result['match_score'])}</p>
        <p><strong>Fit Summary:</strong> {generate_fit_summary(result['match_score'], job_title)}</p>
        <p><strong>Summary:</strong> {result.get('summary', 'N/A')}</p>
        <h3>Experience Relevance:</h3>
        <ul>
            {"".join(generate_experience_relevance(result.get('experience_relevance', {})))}
        </ul>
        <h3>Key Strengths:</h3>
        <ul>{"".join(f"<li>{strength}</li>" for strength in result.get('key_strengths', []))}</ul>
        <h3>Areas for Improvement:</h3>
        <ul>{"".join(f"<li>{area}</li>" for area in result.get('areas_for_improvement', []))}</ul>
        <h3>Skills Gap:</h3>
        <ul>{"".join(f"<li>{skill}</li>" for skill in result.get('skills_gap', []))}</ul>
        <h3>Recruiter Questions:</h3>
        <ol>{"".join(generate_recruiter_questions(result.get('recruiter_questions', [])))}</ol>
    </div>
    """

def generate_experience_relevance(experience_relevance: Dict[str, Any]) -> str:
    return "".join(
        f"<li>{job}:<ul>{''.join(f'<li>{project}: {relevance}</li>' for project, relevance in details.items()) if isinstance(details, dict) else f'<li>{details}</li>'}</ul></li>"
        for job, details in experience_relevance.items()
    )

def generate_recruiter_questions(questions: List[Dict[str, str]]) -> str:
    return "".join(
        f"<li>{question['question']}<br><small>Purpose: {question['purpose']}</small></li>"
        if isinstance(question, dict)
        else f"<li>{question}</li>"
        for question in questions
    )
    
