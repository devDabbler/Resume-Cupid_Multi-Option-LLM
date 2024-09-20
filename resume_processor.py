import logging
import os
import re
import yaml
import time
import random
import json
import hashlib
import sqlite3
import traceback
from functools import lru_cache
from typing import List, Dict, Any
import tiktoken
import spacy
from utils import get_logger, get_db_connection
from claude_analyzer import ClaudeAPI
from llama_analyzer import LlamaAPI
from gpt4o_mini_analyzer import GPT4oMiniAPI
from config_settings import Config

logger = get_logger(__name__)
nlp = spacy.load("en_core_web_md")

def clean_text(text: str) -> str:
    # Remove extra whitespace and capitalize sentences
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return '. '.join(sentence.capitalize() for sentence in cleaned.split('. '))

class ResumeProcessor:
    def __init__(self, api_key: str, backend: str, job_roles_file: str = 'job_roles.yaml'):
        logger.info(f"Initializing ResumeProcessor with backend: {backend}")
        self.backend = backend
        
        if backend == "claude":
            logger.info("Initializing ClaudeAPI")
            self.analyzer = ClaudeAPI(api_key)
        elif backend == "llama":
            logger.info("Initializing LlamaAPI")
            self.analyzer = LlamaAPI(api_key)
        elif backend == "gpt4o_mini":
            logger.info("Initializing GPT4oMiniAPI")
            self.analyzer = GPT4oMiniAPI(api_key)
        else:
            logger.error(f"Unsupported backend: {backend}")
            raise ValueError(f"Unsupported backend: {backend}")
        
        self.token_limit = 10000
        self.token_bucket = self.token_limit
        self.last_refill = time.time()
        self.encoder = tiktoken.encoding_for_model("gpt-4")
        
        # Load job roles
        try:
            with open(job_roles_file, 'r') as file:
                self.job_roles = yaml.safe_load(file)
            logger.info(f"Job roles loaded from {job_roles_file}")
        except Exception as e:
            logger.error(f"Error loading job roles from {job_roles_file}: {str(e)}")
            self.job_roles = {}
        
        logger.info(f"ResumeProcessor initialized with {backend} backend and job roles")

    def estimate_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def wait_for_tokens(self, tokens_needed: int):
        while True:
            current_time = time.time()
            time_passed = current_time - self.last_refill
            self.token_bucket = min(self.token_limit, self.token_bucket + time_passed * (self.token_limit / 60))
            self.last_refill = current_time

            if self.token_bucket >= tokens_needed:
                self.token_bucket -= tokens_needed
                return
            else:
                sleep_time = (tokens_needed - self.token_bucket) / (self.token_limit / 60)
                time.sleep(sleep_time + random.uniform(0, 1))  # Add jitter

    @lru_cache(maxsize=128)
    def analyze_resume(self, resume_text: str, job_description: str, job_title: str, importance_factors: Dict[str, float] = None) -> Dict[str, Any]:
        logger.debug(f"Starting resume analysis with {self.backend} backend")
        cache_key = self._get_cache_key(resume_text, job_description, job_title)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.debug(f"Retrieved cached result for {self.backend} backend")
            return cached_result

        logger.debug(f"No cached result found. Performing new analysis with {self.backend} backend")

        combined_text = resume_text + " " + job_description + " " + job_title
        estimated_tokens = self.estimate_tokens(combined_text)
        logger.debug(f"Estimated tokens for analysis: {estimated_tokens}")

        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.wait_for_tokens(estimated_tokens)
                logger.debug(f"Calling {self.backend} API for analysis")
                analysis = self.analyzer.analyze_match(resume_text, job_description, {}, job_title)
                logger.debug(f"API analysis result: {analysis}")
        
                years_of_experience = self._extract_years_of_experience(resume_text)
                logger.debug(f"Extracted years of experience: {years_of_experience}")

                result = {
                    'file_name': '',
                    'match_score': analysis.get('match_score', 0),
                    'years_of_experience': years_of_experience,
                    'brief_summary': analysis.get('brief_summary', 'No summary available'),
                    'experience_and_project_relevance': analysis.get('experience_and_project_relevance', 'No analysis available'),
                    'strengths': [],
                    'areas_for_improvement': [],
                    'skills_gap': analysis.get('skills_gap', []),
                    'recommendation': analysis.get('recommendation_for_interview', 'No recommendation available'),
                    'recruiter_questions': analysis.get('recruiter_questions', []),
                    'project_relevance': analysis.get('experience_and_project_relevance', 'No project relevance analysis available')
                }

                logger.debug("Analysis result compiled successfully")
                self._cache_result(cache_key, result)

                return result

            except Exception as e:
                if "Rate limit exceeded" in str(e) and attempt < max_retries - 1:
                    sleep_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limit exceeded. Retrying in {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Error in resume analysis: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return self._generate_error_result(str(e))

        logger.error("Max retries reached. Unable to complete analysis.")
        return self._generate_error_result("Max retries reached")   

    def analyze_with_fallback(self, resume_text: str, job_description: str, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f"Attempting analysis with {self.backend} backend")
            return self.analyzer.analyze_match(resume_text, job_description, candidate_data)
        except Exception as e:
            if "Rate limit exceeded" in str(e):
                logger.warning(f"{self.backend} backend hit rate limit. No fallback available.")
            else:
                logger.error(f"Error with {self.backend} backend: {str(e)}")
            raise Exception(f"Analysis failed with {self.backend} backend. Unable to complete analysis.")

    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        return {
            'file_name': 'Error',
            'match_score': 0,
            'brief_summary': 'An error occurred during analysis',
            'fit_summary': 'Unable to generate fit summary due to an error',
            'recommendation': 'Unable to provide a recommendation due to an error',
            'experience_and_project_relevance': ['Unable to analyze experience and project relevance due to an error'],
            'skills_gap': ['Unable to determine skills gap due to an error'],
            'key_strengths': ['Unable to identify key strengths due to an error'],
            'areas_for_improvement': ['Unable to identify areas for improvement due to an error'],
            'recruiter_questions': ['Unable to generate recruiter questions due to an error'],
            'error': error_message
        }

    def analyze_match(self, resume: str, job_description: str, candidate_data: Dict[str, Any], job_title: str) -> Dict[str, Any]:
        logger.debug(f"Analyzing match for resume length: {len(resume)}, job description length: {len(job_description)}, job title: {job_title}")

        try:
            raw_analysis = self.analyzer.analyze_match(resume, job_description, candidate_data, job_title)
            logger.debug(f"Raw analysis result: {json.dumps(raw_analysis, indent=2)}")
    
            standardized_result = self._standardize_analysis(raw_analysis, resume, job_title)
            logger.debug(f"Standardized analysis result: {json.dumps(standardized_result, indent=2)}")
    
            if not self._meets_minimum_requirements(standardized_result, job_title):
                standardized_result['recommendation'] = "Do not recommend for interview"
    
            return standardized_result
        except Exception as e:
            logger.error(f"Error in analyze_match: {str(e)}", exc_info=True)
            return self._generate_error_result(str(e))

    def _standardize_analysis(self, raw_analysis: Dict[str, Any], resume: str, job_title: str) -> Dict[str, Any]:
        def clean_and_format(value):
            if isinstance(value, str):
                return clean_text(value)
            elif isinstance(value, list):
                return [clean_and_format(item) for item in value]
            elif isinstance(value, dict):
                return {k: clean_and_format(v) for k, v in value.items()}
            return value

        try:
            match_score = int(raw_analysis.get('match_score', 0))
        except (ValueError, TypeError):
            match_score = 0

        standardized = {
            'file_name': raw_analysis.get('file_name', 'Unknown'),
            'match_score': match_score,
            'brief_summary': clean_and_format(raw_analysis.get('brief_summary', 'No summary available')),
            'fit_summary': clean_and_format(self._generate_fit_summary(match_score, job_title)),
            'recommendation': self._generate_recommendation(match_score),
            'experience_and_project_relevance': self._standardize_relevance(raw_analysis.get('experience_and_project_relevance', {})),
            'skills_gap': self._standardize_skills_gap(raw_analysis.get('skills_gap', {})),
            'key_strengths': clean_and_format(raw_analysis.get('key_strengths', [])),
            'areas_for_improvement': clean_and_format(raw_analysis.get('areas_for_improvement', [])),
            'recruiter_questions': clean_and_format(raw_analysis.get('recruiter_questions', []))[:5],  # Limit to 5 questions
            'confidence_score': self._calculate_confidence_score(raw_analysis)
        }

        return standardized

    def _meets_minimum_requirements(self, result: Dict[str, Any], job_title: str) -> bool:
        logger.debug(f"Checking minimum requirements for {job_title}")
        job_role = self.job_roles.get(job_title.lower().replace(' ', '_'))
        if not job_role:
            logger.warning(f"No job role found for {job_title}. Using default minimum requirements.")
            return True  # Default to True if we don't have specific requirements

        min_requirements = job_role.get('minimum_requirements', {})
        min_score = min_requirements.get('min_score', 0)
        critical_skills = min_requirements.get('critical_skills', [])
        min_experience_years = min_requirements.get('min_experience_years', 0)

        if result['match_score'] < min_score:
            logger.info(f"Candidate does not meet minimum score requirement. Score: {result['match_score']}, Required: {min_score}")
            return False

        missing_critical_skills = [skill for skill in critical_skills 
                               if skill.lower() not in [s.lower() for s in result['skills_gap'].get('missing_skills', [])]]
        if missing_critical_skills:
            logger.info(f"Candidate is missing critical skills: {', '.join(missing_critical_skills)}")
            return False

        candidate_experience = result.get('years_of_experience', 0)
        if candidate_experience < min_experience_years:
            logger.info(f"Candidate does not meet minimum years of experience. Years: {candidate_experience}, Required: {min_experience_years}")
            return False

        return True

    def _generate_recommendation(self, match_score: int) -> str:
        if match_score >= 85:
            return "Highly recommend for interview"
        elif 70 <= match_score < 85:
            return "Recommend for interview"
        elif 50 <= match_score < 70:
            return "Consider for interview with reservations"
        else:
            return "Do not recommend for interview"

    def _standardize_relevance(self, relevance_data: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(relevance_data, dict):
            return {
                'overall_relevance': relevance_data.get('relevance', 0),
                'description': relevance_data.get('description', 'No description available'),
                'relevant_experience': relevance_data.get('relevant_experience', 0),
                'project_relevance': relevance_data.get('project_relevance', 0),
                'technical_skills_relevance': relevance_data.get('technical_skills', 0)
            }
            return {'overall_relevance': 0, 'description': 'Unable to assess relevance'}

    def _standardize_skills_gap(self, skills_gap: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(skills_gap, dict):
            if 'skills' in skills_gap:
                return {
                    'missing_skills': skills_gap['skills'],
                    'description': skills_gap.get('description', 'No description available')
                }
            else:
                return {k: v for k, v in skills_gap.items() if k != 'description'}
        return {'missing_skills': [], 'description': 'Unable to assess skills gap'}

    def _calculate_confidence_score(self, raw_analysis: Dict[str, Any]) -> float:
        # This is a simple implementation. You may want to develop a more sophisticated method.
        required_fields = ['match_score', 'brief_summary', 'experience_and_project_relevance', 'skills_gap', 'key_strengths']
        filled_fields = sum(1 for field in required_fields if raw_analysis.get(field))
        return filled_fields / len(required_fields)

    def _generate_fit_summary(self, match_score: int, job_title: str) -> str:
        if match_score < 50:
            return f"The candidate is not a strong fit for the {job_title} role, with considerable gaps in required skills and experience."
        elif 50 <= match_score < 70:
            return f"The candidate shows potential for the {job_title} role but has some gaps that would require further assessment."
        elif 70 <= match_score < 85:
            return f"The candidate is a good fit for the {job_title} role, meeting many of the job requirements with some minor gaps."
        else:
            return f"The candidate is an excellent fit for the {job_title} role, meeting or exceeding most job requirements."

    def _format_strengths_and_improvements(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted = []
        for item in items:
            if isinstance(item, dict) and 'category' in item and 'points' in item:
                formatted.append({
                    'category': item['category'].capitalize(),
                    'points': [clean_text(point) for point in item['points'] if point.strip()]
                })
        return formatted

    def _invalidate_cache(self, cache_key: str):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('DELETE FROM resume_cache WHERE cache_key = ?', (cache_key,))
        conn.commit()
        conn.close()
        logger.debug(f"Invalidated cache for key: {cache_key}")

    def _get_cache_key(self, resume_text: str, job_description: str, job_title: str) -> str:
        combined = f"{self.backend}:{resume_text}{job_description}{job_title}"
        logger.debug(f"Generated cache key for backend: {self.backend}")
        return hashlib.md5(combined.encode()).hexdigest()

    def clear_cache_for_resume(self, resume_text: str, job_description: str):
        cache_key = self._get_cache_key(resume_text, job_description)
        self._invalidate_cache(cache_key)
        
    def clear_cache(self):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('DELETE FROM resume_cache')
        conn.commit()
        conn.close()
        logger.debug("Cleared all entries from resume_cache")

    def _get_cached_result(self, cache_key: str) -> Dict[str, Any]:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT result FROM resume_cache WHERE cache_key = ?', (cache_key,))
        row = cur.fetchone()
        conn.close()
        if row:
            result = json.loads(row[0])
            logger.debug(f"Retrieved cached result: {result}")
            if 'error' in result or result.get('brief_summary') == 'Unable to complete analysis':
                logger.warning(f"Cached result contains an error. Invalidating cache.")
                self._invalidate_cache(cache_key)
                return None
            return result
        return None

    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
        INSERT OR REPLACE INTO resume_cache (cache_key, result, created_at)
        VALUES (?, ?, datetime('now'))
        ''', (cache_key, json.dumps(result)))
        conn.commit()
        conn.close()

    def rank_resumes(self, resumes: List[str], job_description: str, importance_factors: Dict[str, float] = None) -> List[Dict[str, Any]]:
        results = []
        for resume in resumes:
            analysis = self.analyze_resume(resume, job_description, importance_factors)
            results.append(analysis)

        # Sort results by match score in descending order
        ranked_results = sorted(results, key=lambda x: x['match_score'], reverse=True)
        
        return ranked_results

def create_resume_processor(api_key: str, backend: str = "claude") -> ResumeProcessor:
    logger.info(f"Creating ResumeProcessor with backend: {backend}")
    try:
        return ResumeProcessor(api_key, backend)
    except Exception as e:
        logger.error(f"Failed to create ResumeProcessor: {str(e)}")
        raise

def init_resume_cache():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Create resume_cache table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS resume_cache (
        cache_key TEXT PRIMARY KEY,
        result TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Initialized resume cache")

# Initialize the resume cache table when this module is imported
init_resume_cache()