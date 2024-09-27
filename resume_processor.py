import logging
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, List, Optional
import json
import spacy
import yaml
import os
import random
from llama_service import llama_service
from utils import extract_text_from_file, generate_job_requirements
from score_calculator import score_calculator

class ResumeProcessor:
    def __init__(self):
        self.job_roles = self.load_job_roles()
        self.logger = logging.getLogger(__name__)

    def load_job_roles(self) -> Dict[str, Any]:
        try:
            with open('job_roles.yaml', 'r') as file:
                return yaml.safe_load(file)['job_roles']
        except Exception as e:
            self.logger.error(f"Error loading job roles: {str(e)}")
            return {}

    def load_job_requirements(self, job_title: str) -> Dict[str, Any]:
        role = self.job_roles.get(job_title, {})
        return {
            'required_skills': role.get('required_skills', []),
            'preferred_skills': role.get('specific_knowledge', []),
            'soft_skills': role.get('soft_skills', []),
            'education_level': role.get('education_level', ''),
            'years_of_experience': role.get('years_of_experience', 0),
            'industry_experience': role.get('industry_experience', [])
        }

    def process_resume(self, resume_text: str, job_description: str, job_title: str) -> Dict[str, Any]:
        try:
            self.logger.info(f"Starting resume processing for job title: {job_title}")

            # Load and merge job requirements
            job_requirements = self.load_job_requirements(job_title)
            if not job_requirements:
                self.logger.warning(f"No job requirements found for title: {job_title}. Using generated requirements.")
                job_requirements = generate_job_requirements(job_description)
            self.logger.debug(f"Job requirements: {job_requirements}")

            description_requirements = generate_job_requirements(job_description)
            self.logger.debug(f"Generated description requirements: {description_requirements}")

            merged_requirements = self._merge_requirements(job_requirements, description_requirements)
            self.logger.debug(f"Merged requirements: {merged_requirements}")

            # Perform LLM analysis
            self.logger.info("Calling LLM service for resume analysis")
            llm_analysis = self._get_llama_analysis(resume_text, job_description, job_title)

            if llm_analysis is None:
                return self._generate_error_result("Unknown", "LLAMA analysis failed")

            # Process analysis results
            self.logger.info("Processing analysis results")
            result = self._process_analysis(llm_analysis, "Unknown", job_title, merged_requirements)
            self.logger.debug(f"Processed analysis result: {result}")

            # Extract experience
            result['experience'] = self._extract_experience(resume_text)
            self.logger.debug(f"Extracted experience: {result['experience']}")

            # Calculate score using the dynamic approach
            score_result = score_calculator.calculate_score(llm_analysis, merged_requirements)
            result.update(score_result)
            self.logger.debug(f"Calculated score: {score_result}")

            # Generate additional fields
            result['recommendation'] = self._generate_recommendation(result['match_score'])
            result['fit_summary'] = self._generate_fit_summary(result['match_score'], job_title)
            result['recruiter_questions'] = self._generate_recruiter_questions(llm_analysis, job_title, job_description)

            self.logger.info(f"Processed resume. Match score: {result['match_score']}, Recommendation: {result['recommendation']}")
            return result

        except Exception as e:
            self.logger.error(f"Error in process_resume: {str(e)}", exc_info=True)
            return self._generate_error_result("Unknown", str(e))

    def _get_llama_analysis(self, resume_text: str, job_description: str, job_title: str) -> Optional[Dict[str, Any]]:
        try:
            analysis = llama_service.analyze_resume(resume_text, job_description, job_title)
            self.logger.debug(f"Received analysis from LLAMA: {analysis}")
            return analysis
        except Exception as e:
            self.logger.error(f"Error in LLAMA analysis: {str(e)}", exc_info=True)
            return None

    def _get_falcon_analysis(self, resume_text: str, job_description: str, job_title: str) -> Optional[Dict[str, Any]]:
        # Placeholder for Falcon analysis
        # Replace this with actual Falcon service call when implemented
        self.logger.warning("Falcon analysis not implemented yet")
        return None

    def _combine_analyses(self, llama_analysis: Optional[Dict[str, Any]], falcon_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        combined = {}
        for key in set(llama_analysis.keys() if llama_analysis else []) | set(falcon_analysis.keys() if falcon_analysis else []):
            llama_value = llama_analysis.get(key) if llama_analysis else None
            falcon_value = falcon_analysis.get(key) if falcon_analysis else None

            if llama_value is not None and falcon_value is not None:
                if isinstance(llama_value, (int, float)) and isinstance(falcon_value, (int, float)):
                    combined[key] = (llama_value + falcon_value) / 2
                elif isinstance(llama_value, list) and isinstance(falcon_value, list):
                    combined[key] = list(set(llama_value + falcon_value))
                else:
                    combined[key] = f"LLAMA: {llama_value} | Falcon: {falcon_value}"
            else:
                combined[key] = llama_value if llama_value is not None else falcon_value

        return combined

    def _extract_experience(self, resume_text: str) -> str:
        experience_section = re.search(r'PROFESSIONAL EXPERIENCE.*?(?=EDUCATION|ACADEMIC PROJECTS|$)', resume_text, re.DOTALL | re.IGNORECASE)
        return experience_section.group(0) if experience_section else ''

    def _process_analysis(self, llm_analysis: Dict[str, Any], file_name: str, job_title: str, job_requirements: Dict[str, Any]) -> Dict[str, Any]:
        processed = {
            'file_name': file_name,
            'brief_summary': llm_analysis.get('Brief Summary', 'No summary available'),
            'key_strengths': self._format_list(llm_analysis.get('Key Strengths', [])),
            'areas_for_improvement': self._format_list(llm_analysis.get('Areas for Improvement', [])),
            'skills_gap': self._format_list(llm_analysis.get('Missing Critical Skills', [])),
            'experience_relevance': llm_analysis.get('Experience and Project Relevance', 'Not assessed'),
        }
        return processed

    def _merge_requirements(self, job_requirements: Dict[str, Any], description_requirements: Dict[str, Any]) -> Dict[str, Any]:
        merged = job_requirements.copy()
        for key, value in description_requirements.items():
            if key in merged:
                if isinstance(merged[key], list) and isinstance(value, list):
                    merged[key] = list(set(merged[key] + value))
                elif isinstance(merged[key], (int, float)) and isinstance(value, (int, float)):
                    merged[key] = max(merged[key], value)
                else:
                    merged[key] = value
            else:
                merged[key] = value
        return merged
    
    def _format_list(self, items: List[Any]) -> List[str]:
        if not items:
            return ["No items available"]
        
        formatted_items = []
        for item in items:
            if isinstance(item, str):
                formatted_item = item.strip()
                if formatted_item:
                    formatted_items.append(formatted_item)
            elif isinstance(item, dict):
                formatted_items.append(json.dumps(item))
            else:
                str_item = str(item).strip()
                if str_item:
                    formatted_items.append(str_item)
        
        return formatted_items if formatted_items else ["No items available"]

    def _generate_recommendation(self, match_score: int) -> str:
        if match_score >= 85:
            return "Strongly recommend for interview"
        elif 70 <= match_score < 85:
            return "Recommend for interview"
        elif 55 <= match_score < 70:
            return "Consider for interview with reservations"
        else:
            return "Not recommended for interview at this time"

    def _generate_fit_summary(self, match_score: int, job_title: str) -> str:
        if match_score >= 85:
            return f"The candidate is an excellent fit for the {job_title} role, meeting or exceeding most job requirements."
        elif 70 <= match_score < 85:
            return f"The candidate is a good fit for the {job_title} role, meeting many of the job requirements with some minor gaps."
        elif 55 <= match_score < 70:
            return f"The candidate shows potential for the {job_title} role but has some gaps that would require further assessment."
        else:
            return f"The candidate is not a strong fit for the {job_title} role, with considerable gaps in required skills and experience."

    def _generate_error_result(self, file_name: str, error_message: str) -> Dict[str, Any]:
        return {
            'file_name': file_name,
            'match_score': 0,
            'brief_summary': f"Error occurred during analysis: {error_message}",
            'recommendation': 'Unable to provide a recommendation due to an error',
            'experience_and_project_relevance': 'Unable to assess due to an error',
            'skills_gap': ['Unable to assess due to an error'],
            'key_strengths': ['Unable to assess due to an error'],
            'areas_for_improvement': ['Unable to assess due to an error'],
            'recruiter_questions': ['Unable to provide questions due to an error'],
            'fit_summary': 'Unable to generate fit summary due to an error',
        }

    def _generate_recruiter_questions(self, analysis: Dict[str, Any], job_title: str, job_description: str) -> List[str]:
        # Implement your recruiter questions generation logic here
        # This is a placeholder implementation
        return analysis.get('Recruiter Questions', ["No questions available"])

# Create an instance of ResumeProcessor
resume_processor = ResumeProcessor()