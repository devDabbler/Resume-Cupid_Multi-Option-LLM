import logging
import os
from typing import List, Dict, Any
import numpy as np
import spacy
from utils import preprocess_text, calculate_similarity
from claude_analyzer import ClaudeAPI
from llama_analyzer import LlamaAPI
from gpt4o_mini_analyzer import GPT4oMiniAPI
import json
import hashlib
import sqlite3

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_md")

DB_PATH = os.getenv('SQLITE_DB_PATH')

import os
import sqlite3
import logging

logger = logging.getLogger(__name__)

def get_db_connection():
    db_path = os.getenv('SQLITE_DB_PATH')
    if not db_path:
        raise ValueError("SQLITE_DB_PATH environment variable is not set")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
        raise

class ResumeProcessor:
    def __init__(self, api_key: str, backend: str = "claude"):
        self.nlp = spacy.load("en_core_web_md")
        if backend == "claude":
            self.api = ClaudeAPI(api_key)
        elif backend == "llama":
            self.api = LlamaAPI(api_key)
        elif backend == "gpt-4o-mini":
            self.api = GPT4oMiniAPI(api_key)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        self.job_descriptions: Dict[str, Dict[str, Any]] = {}
        self.cache = {}

    def _get_cache_key(self, resume_text: str, job_description: str) -> str:
        combined = resume_text + job_description
        return hashlib.md5(combined.encode()).hexdigest()

    def analyze_resume(self, resume_text: str, job_description: str, importance_factors: Dict[str, float] = None) -> Dict[str, Any]:
        cache_key = self._get_cache_key(resume_text, job_description)
        
        # Check if result is in memory cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Check if result is in database cache
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            self.cache[cache_key] = cached_result
            return cached_result

        # If not in cache, perform analysis
        analysis = self.api.analyze_match(resume_text, job_description)
        
        # Extract years of experience
        years_of_experience = self._extract_years_of_experience(resume_text)

        # Combine all results
        result = {
            'file_name': '',  # This will be set in main_app.py
            'match_score': analysis.get('match_score', 0),
            'years_of_experience': years_of_experience,
            'summary': analysis.get('summary', 'No summary available'),
            'analysis': analysis.get('analysis', 'No analysis available'),
            'strengths': analysis.get('strengths', []),
            'areas_for_improvement': analysis.get('areas_for_improvement', []),
            'skills_gap': analysis.get('skills_gap', []),
            'recommendation': analysis.get('recommendation', 'No recommendation available'),
            'interview_questions': analysis.get('interview_questions', []),
            'project_relevance': analysis.get('project_relevance', 'No project relevance analysis available')
        }

        # Cache the result
        self._cache_result(cache_key, result)
        self.cache[cache_key] = result

        return result

    def _extract_years_of_experience(self, resume_text: str) -> int:
        # Simple regex to find years of experience
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
        return json.loads(row[0]) if row else None

    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
        INSERT OR REPLACE INTO resume_cache (cache_key, result, created_at)
        VALUES (?, ?, datetime('now'))
        ''', (cache_key, json.dumps(result)))
        conn.commit()
        conn.close()

    def rank_resumes(self, resumes: List[str], job_id: str, importance_factors: Dict[str, float] = None) -> List[Dict[str, Any]]:
        if job_id not in self.job_descriptions:
            raise ValueError(f"No job description found for job_id: {job_id}")

        results = []
        for resume in resumes:
            analysis = self.analyze_resume(resume, self.job_descriptions[job_id]['text'], importance_factors)
            results.append(analysis)

        # Sort results by match score in descending order
        ranked_results = sorted(results, key=lambda x: x['match_score'], reverse=True)
        
        return ranked_results

def create_resume_processor(api_key: str, backend: str = "claude") -> ResumeProcessor:
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