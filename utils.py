import logging
import io
import PyPDF2
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
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
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

def generate_pdf_report(evaluation_results: List[Dict[str, Any]], run_id: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    elements = []

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

    elements.append(Paragraph(f"Evaluation Report (Run ID: {run_id})", styles['Heading1']))
    elements.append(Spacer(1, 12))

    # Summary table
    data = [['Rank', 'Candidate', 'Match Score (%)', 'Recommendation']]
    for i, result in enumerate(sorted(evaluation_results, key=lambda x: x.get('match_score', 0), reverse=True), 1):
        data.append([
            i,
            result.get('file_name', 'Unknown'),
            result.get('match_score', 0),
            result.get('recommendation', 'N/A')
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

    for result in evaluation_results:
        elements.append(Paragraph(f"{result.get('file_name', 'Unknown')}", styles['Heading2']))
        elements.append(Paragraph(f"Match Score: {result.get('match_score', 0)}%", styles['Normal']))
        elements.append(Paragraph(f"Recommendation: {result.get('recommendation', 'N/A')}", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Brief Summary:", styles['Heading3']))
        elements.append(Paragraph(result.get('brief_summary', 'No summary available.'), styles['Justify']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Fit Summary:", styles['Heading3']))
        elements.append(Paragraph(result.get('fit_summary', 'No fit summary available.'), styles['Justify']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Experience and Project Relevance:", styles['Heading3']))
        experience_and_project_relevance = result.get('experience_and_project_relevance', {})
        if isinstance(experience_and_project_relevance, dict):
            for key, value in experience_and_project_relevance.items():
                elements.append(Paragraph(f"{key}: {value}", styles['Justify']))
        elif isinstance(experience_and_project_relevance, list):
            for item in experience_and_project_relevance:
                elements.append(Paragraph(f"- {item}", styles['Justify']))
        elif isinstance(experience_and_project_relevance, int):
            elements.append(Paragraph(f"Relevance Score: {experience_and_project_relevance}", styles['Justify']))
        else:
            elements.append(Paragraph(str(experience_and_project_relevance), styles['Justify']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Skills Gap:", styles['Heading3']))
        for skill in result.get('skills_gap', []):
            elements.append(Paragraph(f"- {skill}", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Key Strengths:", styles['Heading3']))
        for strength in result.get('key_strengths', []):
            elements.append(Paragraph(f"- {strength}", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Areas for Improvement:", styles['Heading3']))
        for area in result.get('areas_for_improvement', []):
            elements.append(Paragraph(f"- {area}", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Recommended Interview Questions:", styles['Heading3']))
        for question in result.get('recruiter_questions', []):
            elements.append(Paragraph(f"- {question}", styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("", styles['Normal']))  # Add a blank line between candidates

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

def is_valid_email(email: str) -> bool:
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None