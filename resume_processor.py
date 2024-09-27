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

class ResumeProcessor:
    def __init__(self):
        self.job_roles = self.load_job_roles()

    def load_job_roles(self) -> Dict[str, Any]:
        try:
            with open('job_roles.yaml', 'r') as file:
                return yaml.safe_load(file)['job_roles']
        except Exception as e:
            self.logger.error(f"Error loading job roles: {str(e)}")
            return {}

    def get_job_requirements(self, job_title: str) -> Dict[str, Any]:
        role = self.job_roles.get(job_title, {})
        return {
            'required_skills': role.get('required_skills', []),
            'preferred_skills': role.get('specific_knowledge', []),
            'soft_skills': role.get('soft_skills', []),
            'education_level': role.get('education_level', ''),
            'years_of_experience': role.get('years_of_experience', 0),
            'industry_experience': role.get('industry_experience', [])
        }

    def load_job_requirements(self, job_title: str) -> Dict[str, Any]:
        if job_title in self.job_roles:
            return self.job_roles[job_title]
        else:
            self.logger.warning(f"No predefined requirements found for job title: {job_title}")
            return {}

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

            merged_requirements = {**description_requirements, **job_requirements}
            self.logger.debug(f"Merged requirements: {merged_requirements}")

            # Perform LLM analyses
            self.logger.info("Calling LLM services for resume analysis")
            llama_analysis = self._get_llama_analysis(resume_text, job_description, job_title)
            falcon_analysis = self._get_falcon_analysis(resume_text, job_description, job_title)

            if llama_analysis is None and falcon_analysis is None:
                return self._generate_error_result("Unknown", "Both LLAMA and Falcon analyses failed")

            combined_analysis = self._combine_analyses(llama_analysis, falcon_analysis)
            self.logger.debug(f"Combined analysis: {combined_analysis}")

            # Process analysis results
            self.logger.info("Processing analysis results")
            result = self._process_analysis(combined_analysis, "Unknown", job_title, merged_requirements)
            self.logger.debug(f"Processed analysis result: {result}")

            # Extract experience
            result['experience'] = self._extract_experience(resume_text)
            self.logger.debug(f"Extracted experience: {result['experience']}")

            # Calculate score using the dynamic approach
            score_result = self.score_calculator.calculate_score(combined_analysis, merged_requirements)
            result.update(score_result)
            self.logger.debug(f"Calculated score: {score_result}")

            # Generate additional fields
            result['recommendation'] = self._generate_recommendation(result['match_score'])
            result['fit_summary'] = self._generate_fit_summary(result['match_score'], job_title)
            result['recruiter_questions'] = self._generate_recruiter_questions(combined_analysis, job_title, job_description)

            self.logger.info(f"Processed resume. Match score: {result['match_score']}, Recommendation: {result['recommendation']}")
            return result

        except Exception as e:
            self.logger.error(f"Error in process_resume: {str(e)}", exc_info=True)
            return self._generate_error_result("Unknown", str(e))

    def _get_llama_analysis(self, resume_text: str, job_description: str, job_title: str) -> Optional[Dict[str, Any]]:
        try:
            analysis = self.llama_service.analyze_resume(resume_text, job_description, job_title)
            self.logger.debug(f"Received analysis from LLAMA: {analysis}")
            return analysis
        except Exception as e:
            self.logger.error(f"Error in LLAMA analysis: {str(e)}", exc_info=True)
            return None

    def _get_falcon_analysis(self, resume_text: str, job_description: str, job_title: str) -> Optional[Dict[str, Any]]:
        try:
            analysis = self.falcon_service.analyze_resume(resume_text, job_description, job_title)
            self.logger.debug(f"Received analysis from Falcon: {analysis}")
            return analysis
        except Exception as e:
            self.logger.error(f"Error in Falcon analysis: {str(e)}", exc_info=True)
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

    def _process_analysis(self, raw_analysis: Dict[str, Any], file_name: str, job_title: str, job_requirements: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.logger.debug(f"Raw analysis: {raw_analysis}")

            processed = {
                'file_name': file_name,
                'brief_summary': raw_analysis.get('Brief Summary', 'No summary available'),
                'match_score': raw_analysis.get('Match Score', 0),
                'key_strengths': self._format_list(raw_analysis.get('Key Strengths', [])),
                'areas_for_improvement': self._format_list(raw_analysis.get('Areas for Improvement', [])),
                'skills_gap': self._format_list(raw_analysis.get('Skills Gap', [])),
                'experience_relevance': raw_analysis.get('Experience and Project Relevance', 'Not assessed'),
            }

            self.logger.debug(f"Processed analysis: {processed}")

            return processed

        except Exception as e:
            self.logger.error(f"Unexpected error processing analysis for {file_name}: {str(e)}", exc_info=True)
            self.logger.error(f"Raw analysis content: {raw_analysis}")
            return self._generate_error_result(file_name, f"Unexpected Error: {str(e)}")

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
        if match_score >= 80:
            return "Highly recommend for interview"
        elif 65 <= match_score < 80:
            return "Recommend for interview"
        elif 50 <= match_score < 65:
            return "Consider for interview with reservations"
        else:
            return "Not recommended for interview at this time"

    def _generate_fit_summary(self, match_score: int, job_title: str) -> str:
        if match_score >= 75:
            return f"The candidate is an excellent fit for the {job_title} role, meeting or exceeding most job requirements."
        elif 60 <= match_score < 75:
            return f"The candidate is a good fit for the {job_title} role, meeting many of the job requirements with some minor gaps."
        elif 40 <= match_score < 60:
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

    def _calculate_weighted_score(self, result: Dict[str, Any], requirements: Dict[str, Any], job_description: str) -> int:
        # Implement your weighted score calculation logic here
        # This is a placeholder implementation
        return min(max(result.get('match_score', 0), 0), 100)

    def _generate_recruiter_questions(self, analysis: Dict[str, Any], job_title: str, job_description: str) -> List[str]:
        # Implement your recruiter questions generation logic here
        # This is a placeholder implementation
        return analysis.get('Recruiter Questions', ["No questions available"])

# Create an instance of ResumeProcessor
resume_processor = ResumeProcessor()