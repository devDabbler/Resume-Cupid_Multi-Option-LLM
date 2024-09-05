import logging
import os
import time
import random
from typing import List, Dict, Any
import tiktoken
import spacy
from utils import preprocess_text, calculate_similarity
from claude_analyzer import ClaudeAPI
from llama_analyzer import LlamaAPI
from gpt4o_mini_analyzer import GPT4oMiniAPI
import json
import hashlib
import sqlite3
import traceback

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_md")

def get_db_connection():
    db_path = os.getenv('SQLITE_DB_PATH')
    
    logger.info(f"Current working directory: {os.getcwd()}")
    
    if not db_path:
        logger.error("SQLITE_DB_PATH is not set in the environment variables")
        raise ValueError("SQLITE_DB_PATH is not set in the environment variables")
    
    logger.info(f"Environment variable SQLITE_DB_PATH: {db_path}")
    
    # Ensure the directory exists
    dir_path = os.path.dirname(db_path)
    logger.info(f"Directory path: {dir_path}")
    
    try:
        if not dir_path:
            logger.error("Database directory path is empty")
            raise ValueError("Invalid database path")
        
        logger.info(f"Attempting to create directory: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directory created successfully: {dir_path}")
        
        # Resolve the path within the container
        resolved_path = os.path.abspath(db_path)
        logger.info(f"Resolved database path: {resolved_path}")
        
        conn = sqlite3.connect(resolved_path)
        conn.row_factory = sqlite3.Row
        logger.info("Database connection successful")
        return conn
    
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

class ResumeProcessor:
    def __init__(self, api_key: str, backend: str):
        self.backend = backend
        if backend == "claude":
            self.analyzer = ClaudeAPI(api_key)
        elif backend == "llama":
            self.analyzer = LlamaAPI(api_key)
        elif backend == "gpt4o_mini":
            self.analyzer = GPT4oMiniAPI(api_key)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        self.token_limit = 10000
        self.token_bucket = self.token_limit
        self.last_refill = time.time()
        self.encoder = tiktoken.encoding_for_model("gpt-4")
        logger.info(f"ResumeProcessor initialized with {backend} backend")

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
                    'summary': analysis.get('brief_summary', 'No summary available'),
                    'analysis': analysis.get('experience_and_project_relevance', 'No analysis available'),
                    'strengths': [],
                    'areas_for_improvement': [],
                    'skills_gap': analysis.get('skills_gap', []),
                    'recommendation_for_interview': analysis.get('recommendation_for_interview', 'No recommendation available'),
                    'interview_questions': analysis.get('recruiter_questions', []),
                    'project_relevance': analysis.get('experience_and_project_relevance', 'No project relevance analysis available')
                }

                # Ensure 'recommendation' field is always set
                result['recommendation'] = result['recommendation_for_interview']

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
            'file_name': '',
            'error': error_message,
            'match_score': 0,
            'years_of_experience': 0,
            'summary': 'Error occurred during analysis',
            'analysis': 'Unable to complete analysis due to an error',
            'strengths': [],
            'areas_for_improvement': [],
            'skills_gap': [],
            'recommendation': 'Unable to provide a recommendation due to an error',
            'interview_questions': [],
            'project_relevance': 'Unable to analyze project relevance due to an error'
        }

    def analyze_match(self, resume: str, job_description: str, candidate_data: Dict[str, Any], job_title: str) -> Dict[str, Any]:
        logger.debug(f"analyze_match called with args: {resume[:20]}..., {job_description[:20]}..., {candidate_data}, {job_title}")
        return self.analyzer.analyze_match(resume, job_description, candidate_data, job_title)

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

    def _extract_years_of_experience(self, resume_text: str) -> int:
        import re
        experience_matches = re.findall(r'(\d+)\s*(?:years?|yrs?)', resume_text, re.IGNORECASE)
        if experience_matches:
            return max(map(int, experience_matches))
        return 0

    def _get_cached_result(self, cache_key: str) -> Dict[str, Any]:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT result FROM resume_cache WHERE cache_key = ?', (cache_key,))
        row = cur.fetchone()
        conn.close()
        if row:
            result = json.loads(row[0])
            logger.debug(f"Retrieved cached result: {result}")
            if 'error' in result or result.get('summary') == 'Unable to complete analysis':
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
    logger.debug(f"Creating ResumeProcessor with backend: {backend}")
    return ResumeProcessor(api_key, backend)

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

# Initialize the resume cache table when this module is imported
init_resume_cache()