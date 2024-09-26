import logging
import io
import pypdf
from docx import Document
import re
import spacy
from collections import Counter
from typing import List, Dict, Any
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY


logger = logging.getLogger(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_md")

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
    if match_score >= 80:
        return "Highly recommend for interview"
    elif 65 <= match_score < 80:
        return "Recommend for interview"
    elif 50 <= match_score < 65:
        return "Consider for interview with reservations"
    else:
        return "Not recommended for interview at this time"

def generate_fit_summary(match_score: int, job_title: str) -> str:
    if match_score >= 80:
        return f"The candidate is an excellent fit for the {job_title} role, meeting or exceeding most job requirements."
    elif 65 <= match_score < 80:
        return f"The candidate is a good fit for the {job_title} role, meeting many of the job requirements with some minor gaps."
    elif 50 <= match_score < 65:
        return f"The candidate shows potential for the {job_title} role but has some gaps that would require further assessment."
    else:
        return f"The candidate is not a strong fit for the {job_title} role, with considerable gaps in required skills and experience."
    
def generate_pdf_report(results: List[Dict[str, Any]], run_id: str, job_title: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

    elements.append(Paragraph(f"Evaluation Report for {job_title}", styles['Title']))
    elements.append(Spacer(1, 12))

    # Create the table for the summary
    data = [["Rank", "Resume", "Match Score", "Recommendation"]]
    for i, result in enumerate(sorted(results, key=lambda x: x['match_score'], reverse=True), 1):
        data.append([
            str(i),
            result['file_name'],
            f"{result['match_score']}%",
            generate_recommendation(result['match_score'])
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

    for result in sorted(results, key=lambda x: x['match_score'], reverse=True):
        elements.append(Paragraph(f"Detailed Analysis: {result['file_name']}", styles['Heading2']))
        elements.append(Paragraph(f"Match Score: {result['match_score']}%", styles['Normal']))
        elements.append(Paragraph(f"Recommendation: {generate_recommendation(result['match_score'])}", styles['Normal']))
        elements.append(Paragraph(f"Fit Summary: {generate_fit_summary(result['match_score'], job_title)}", styles['Normal']))
        elements.append(Paragraph(f"Summary: {result.get('summary', 'N/A')}", styles['Normal']))
        
        elements.append(Paragraph("Experience Relevance:", styles['Heading3']))
        for job, details in result.get('experience_relevance', {}).items():
            elements.append(Paragraph(f"- {job}:", styles['Normal']))
            if isinstance(details, dict):
                for project, relevance in details.items():
                    elements.append(Paragraph(f"  * {project}: {relevance}", styles['Normal']))
            else:
                elements.append(Paragraph(f"  * {details}", styles['Normal']))
        
        elements.append(Paragraph("Key Strengths:", styles['Heading3']))
        for strength in result.get('key_strengths', []):
            elements.append(Paragraph(f"- {strength}", styles['Normal']))
        
        elements.append(Paragraph("Areas for Improvement:", styles['Heading3']))
        for area in result.get('areas_for_improvement', []):
            elements.append(Paragraph(f"- {area}", styles['Normal']))
        
        elements.append(Paragraph("Skills Gap:", styles['Heading3']))
        for skill in result.get('skills_gap', []):
            elements.append(Paragraph(f"- {skill}", styles['Normal']))
        
        elements.append(Paragraph("Recruiter Questions:", styles['Heading3']))
        for i, question in enumerate(result.get('recruiter_questions', []), 1):
            if isinstance(question, dict):
                elements.append(Paragraph(f"{i}. {question['question']}", styles['Normal']))
                elements.append(Paragraph(f"   Purpose: {question['purpose']}", styles['Normal']))
            else:
                elements.append(Paragraph(f"{i}. {question}", styles['Normal']))
        
        elements.append(Spacer(1, 12))

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
    
def is_valid_email(email: str) -> bool:
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None