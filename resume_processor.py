import logging
import json
from typing import Dict, Any, List
from llama_service import llama_service
from utils import extract_text_from_file, generate_job_requirements
from config_settings import Config
import yaml
import spacy
import re
import datetime

logger = logging.getLogger(__name__)

class ResumeProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.job_roles = self.load_job_roles()

    def load_job_roles(self):
        with open('job_roles.yaml', 'r') as file:
            return yaml.safe_load(file)['job_roles']

    def load_job_requirements(self, job_title: str) -> Dict[str, Any]:
        job_roles = self.load_job_roles()
        return job_roles.get(job_title, {})

    def process_resume(self, resume_text: str, job_description: str, job_title: str) -> Dict[str, Any]:
        try:
            logger.info(f"Starting resume processing for job title: {job_title}")
        
            job_requirements = self.load_job_requirements(job_title)
            logger.debug(f"Loaded job requirements: {job_requirements}")
        
            description_requirements = generate_job_requirements(job_description)
            logger.debug(f"Generated description requirements: {description_requirements}")
        
            merged_requirements = {**description_requirements, **job_requirements}
            logger.debug(f"Merged requirements: {merged_requirements}")

            logger.info("Calling llama_service for resume analysis")
            analysis = llama_service.analyze_resume(resume_text, job_description, job_title)
            logger.debug(f"Received analysis from llama_service: {analysis}")
    
            logger.info("Processing analysis results")
            result = self._process_analysis(analysis, "Unknown", job_title, merged_requirements)
            logger.debug(f"Processed analysis result: {result}")

            # Extract experience information
            result['experience'] = self._extract_experience(resume_text)
            logger.debug(f"Extracted experience: {result['experience']}")

            # Ensure we have a valid match score
            if result['match_score'] == 0:
                logger.info("Calculating weighted score")
                weighted_score = self._calculate_weighted_score(result, merged_requirements)
                result['match_score'] = weighted_score
                logger.debug(f"Calculated weighted score: {weighted_score}")

            result['recommendation'] = self._generate_recommendation(result['match_score'])
            result['fit_summary'] = self._generate_fit_summary(result['match_score'], job_title)

            logger.info(f"Processed resume. Match score: {result['match_score']}, Recommendation: {result['recommendation']}")
            return result
        except Exception as e:
            logger.error(f"Error in process_resume: {str(e)}", exc_info=True)
            return self._generate_error_result("Unknown", str(e))

    def _extract_experience(self, resume_text: str) -> str:
        # Simple extraction of experience section
        experience_section = re.search(r'EXPERIENCE.*?(?=EDUCATION|SKILLS|$)', resume_text, re.DOTALL | re.IGNORECASE)
        return experience_section.group(0) if experience_section else ''

    def _process_analysis(self, raw_analysis: Dict[str, Any], file_name: str, job_title: str, job_requirements: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Check if the raw_analysis is already in the correct format
            if isinstance(raw_analysis, dict) and 'analysis' in raw_analysis:
                analysis_str = raw_analysis['analysis']
            else:
                analysis_str = str(raw_analysis)

            # Try to extract JSON content
            try:
                analysis = json.loads(self._extract_json_content(analysis_str))
            except json.JSONDecodeError:
                # If JSON parsing fails, use the raw string as the analysis
                analysis = {'Brief Summary': analysis_str}

        except Exception as e:
            logger.error(f"Unexpected error processing analysis for {file_name}: {str(e)}")
            logger.error(f"Raw analysis content: {raw_analysis}")
            return self._generate_error_result(file_name, f"Unexpected Error: {str(e)}")

        processed = {
            'file_name': file_name,
            'match_score': self._normalize_score(int(analysis.get('Match Score', 0))),
            'brief_summary': analysis.get('Brief Summary', 'No summary available'),
            'experience_and_project_relevance': analysis.get('Experience and Project Relevance', 'No relevance information available'),
            'skills_gap': self._format_list(analysis.get('Skills Gap', [])),
            'key_strengths': self._format_list(analysis.get('Key Strengths', [])),
            'areas_for_improvement': self._format_list(analysis.get('Areas for Improvement', [])),
            'recruiter_questions': self._format_list(analysis.get('Recruiter Questions', [])),
            'education': analysis.get('Education', ''),
            'experience_years': int(analysis.get('Total Years of Experience', 0)),
            'skills': analysis.get('Skills', []),
            'knowledge': analysis.get('Knowledge Areas', []),
            'soft_skills': analysis.get('Soft Skills', []),
        }

        # If we don't have a match score from the analysis, calculate one
        if processed['match_score'] == 0:
            processed['match_score'] = self._calculate_weighted_score(processed, job_requirements)

        # Generate fallback content only if necessary
        if not processed['key_strengths'] or processed['key_strengths'] == ['No items available']:
            processed['key_strengths'] = self._generate_key_strengths(analysis, job_requirements)

        if not processed['areas_for_improvement'] or processed['areas_for_improvement'] == ['No items available']:
            processed['areas_for_improvement'] = self._generate_areas_for_improvement(analysis, job_requirements)

        if not processed['recruiter_questions'] or processed['recruiter_questions'] == ['No items available']:
            processed['recruiter_questions'] = self._generate_recruiter_questions(job_requirements)

        logger.debug(f"Processed analysis for {file_name}: {processed}")
        return processed

    def _generate_key_strengths(self, analysis: Dict[str, Any], job_requirements: Dict[str, Any]) -> List[str]:
        strengths = []
        candidate_skills = set(skill.lower() for skill in analysis.get('Skills', []))
        required_skills = set(skill.lower() for skill in job_requirements.get('required_skills', []))
        matching_skills = candidate_skills.intersection(required_skills)
        
        for skill in list(matching_skills)[:3]:  # List top 3 matching skills
            strengths.append(f"Proficiency in {skill.capitalize()}")
        
        if len(strengths) < 3:
            strengths.append("Candidate has relevant experience in the field")
        
        return strengths if strengths else ["No key strengths identified"]

    def _generate_areas_for_improvement(self, analysis: Dict[str, Any], job_requirements: Dict[str, Any]) -> List[str]:
        improvements = []
        candidate_skills = set(skill.lower() for skill in analysis.get('Skills', []))
        required_skills = set(skill.lower() for skill in job_requirements.get('required_skills', []))
        missing_skills = required_skills - candidate_skills
        
        for skill in list(missing_skills)[:3]:  # List top 3 missing skills
            improvements.append(f"Develop proficiency in {skill.capitalize()}")
        
        if len(improvements) < 3:
            improvements.append("Gain more experience in industry-specific projects")
        
        return improvements if improvements else ["No specific areas for improvement identified"]

    def _generate_recruiter_questions(self, job_requirements: Dict[str, Any]) -> List[str]:
        required_skills = job_requirements.get('required_skills', ['relevant skills'])
        first_skill = required_skills[0] if required_skills else 'relevant skills'
        questions = [
            f"Can you describe your experience with {first_skill}?",
            "How do you stay updated with the latest developments in your field?",
            "Can you give an example of a challenging project you've worked on and how you overcame the difficulties?"
        ]
        return questions

    def _calculate_weighted_score(self, result: Dict[str, Any], requirements: Dict[str, Any]) -> int:
        score = 0
        max_score = 0

        weights = {
            'education': 15,
            'experience': 25,
            'skills': 20,
            'specific_knowledge': 15,
            'soft_skills': 5,
            'us_experience': 10,
            'years_of_experience': 10
        }

        # Check education
        candidate_education = result.get('education', '').lower()
        required_education = requirements.get('education_level', '').lower()
        if required_education and required_education in candidate_education:
            score += weights['education']
        elif 'bachelor' in candidate_education and 'master' in required_education:
            score += weights['education'] * 0.5
        max_score += weights['education']

        # Check experience
        candidate_experience = result.get('experience_years', 0)
        required_experience = requirements.get('years_of_experience', 0)
        if candidate_experience >= required_experience:
            score += weights['experience']
        elif required_experience > 0:
            experience_ratio = min(candidate_experience / required_experience, 1)
            score += weights['experience'] * experience_ratio
        max_score += weights['experience']

        # Check years of experience
        years_of_experience = self._calculate_years_of_experience(result.get('experience', ''))
        if years_of_experience >= required_experience:
            score += weights['years_of_experience']
        else:
            score += weights['years_of_experience'] * (years_of_experience / required_experience)
            max_score += weights['years_of_experience']

        # Check U.S. experience
        us_experience = self._check_us_experience(result.get('experience', ''))
        if us_experience:
            score += weights['us_experience']
        max_score += weights['us_experience']

        # Check skills
        required_skills = set(skill.lower() for skill in requirements.get('required_skills', []))
        candidate_skills = set(skill.lower() for skill in result.get('skills', []))
        if required_skills:
            skill_match_ratio = len(required_skills.intersection(candidate_skills)) / len(required_skills)
            score += weights['skills'] * skill_match_ratio
        max_score += weights['skills']

        # Check specific knowledge
        specific_knowledge = set(skill.lower() for skill in requirements.get('specific_knowledge', []))
        candidate_knowledge = set(skill.lower() for skill in result.get('knowledge', []))
        if specific_knowledge:
            knowledge_match_ratio = len(specific_knowledge.intersection(candidate_knowledge)) / len(specific_knowledge)
            score += weights['specific_knowledge'] * knowledge_match_ratio
        max_score += weights['specific_knowledge']

        # Check soft skills
        required_soft_skills = set(skill.lower() for skill in requirements.get('soft_skills', []))
        candidate_soft_skills = set(skill.lower() for skill in result.get('soft_skills', []))
        if required_soft_skills:
            soft_skills_match_ratio = len(required_soft_skills.intersection(candidate_soft_skills)) / len(required_soft_skills)
            score += weights['soft_skills'] * soft_skills_match_ratio
        max_score += weights['soft_skills']

        # Calculate final percentage score
        if max_score > 0:
            final_score = int((score / max_score) * 100)
        else:
            final_score = 0

        # Apply additional adjustments
        if not us_experience:
            final_score = max(final_score - 15, 0)  # Increased penalty for no U.S. experience

        # Adjust for junior status
        if years_of_experience < 3:
            final_score = max(final_score - 10, 0)  # Penalty for junior status

        # Adjust for previous role being primarily Data Analyst
        if self._is_primarily_data_analyst(result.get('experience', '')):
            final_score = max(final_score - 5, 0)  # Small penalty for primarily Data Analyst experience

        # Ensure the score doesn't exceed 90
        final_score = min(final_score, 90)

        logger.info(f"Calculated score: {final_score}. Raw score: {score}, Max score: {max_score}")
        return final_score

    def _parse_date(self, date_string: str) -> datetime:
        try:
            return datetime.strptime(date_string, '%B %Y')
        except ValueError:
            try:
                return datetime.strptime(date_string, '%b %Y')
            except ValueError:
                return None

    def _check_us_experience(self, experience: str) -> bool:
        us_keywords = ['united states', 'usa', 'u.s.', 'america']
        return any(keyword in experience.lower() for keyword in us_keywords)

    def _check_us_experience(self, experience: str) -> bool:
        us_keywords = ['united states', 'usa', 'u.s.', 'america']
        return any(keyword in experience.lower() for keyword in us_keywords)

    def _extract_json_content(self, raw_analysis: str) -> str:
        """Extract JSON content from the raw analysis string."""
        json_pattern = re.compile(r'```json(.*?)```', re.DOTALL)
        match = json_pattern.search(raw_analysis)
        if match:
            return match.group(1).strip()
        else:
            # If no JSON block is found, try to parse the entire string as JSON
            try:
                json.loads(raw_analysis)
                return raw_analysis
            except json.JSONDecodeError:
                raise ValueError("No valid JSON content found in the raw analysis string.")

    def _normalize_score(self, score: int) -> int:
        """Normalize the score to be between 30 and 95."""
        return max(30, min(95, score))

    def _format_list(self, items: List[Any]) -> List[str]:
        if not items:
            return ["No items available"]
    
        formatted_items = []
        for item in items:
            if isinstance(item, str):
                formatted_item = item.strip()
                if len(formatted_item) > 1:
                    formatted_items.append(formatted_item)
            elif isinstance(item, dict):
                formatted_items.append(str(item))
            else:
                str_item = str(item).strip()
                if str_item:
                    formatted_items.append(str_item)
    
        return formatted_items if formatted_items else ["No items available"]

    def _generate_recommendation(self, match_score: int) -> str:
        if match_score >= 80:
            return "Highly recommend for interview"
        elif 60 <= match_score < 80:
            return "Recommend for interview"
        elif 40 <= match_score < 60:
            return "Consider for interview with reservations"
        else:
            return "Not recommended for interview at this time"

    def _generate_fit_summary(self, match_score: int, job_title: str) -> str:
        if match_score >= 80:
            return f"The candidate is an excellent fit for the {job_title} role, meeting or exceeding most job requirements."
        elif 60 <= match_score < 80:
            return f"The candidate is a good fit for the {job_title} role, meeting many of the job requirements with some minor gaps."
        elif 40 <= match_score < 60:
            return f"The candidate shows potential for the {job_title} role but has some gaps that would require further assessment."
        else:
            return f"The candidate is not a strong fit for the {job_title} role, with considerable gaps in required skills and experience."

    def _adjust_match_score(self, score: int, relevance: int) -> int:
        # Adjust score based on experience relevance
        if relevance < 50:
            score = score * (relevance / 100)
    
        # Further adjustments
        if score > 90:
            return 80 + (score - 90) * 0.5  # Compress scores above 90
        elif score > 70:
            return 60 + (score - 70) * 1  # Slightly compress scores between 70 and 90
        elif score < 40:
            return max(20, score - 10)  # Ensure very low scores for poor fits
        else:
            return score  # Keep mid-range scores as they are

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

resume_processor = ResumeProcessor()
