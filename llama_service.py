import time
import logging
from functools import wraps
from typing import Dict, Any
from groq import Groq
from config_settings import Config
import os
import json

logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries=3, backoff_in_seconds=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Max retries reached. Last error: {str(e)}")
                        raise
                    wait_time = backoff_in_seconds * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed. Retrying in {wait_time} seconds. Error: {str(e)}")
                    time.sleep(wait_time)
        return wrapper
    return decorator

class LlamaService:
    def __init__(self):
        print("Environment variables:", os.environ)
        self.api_key = os.getenv('LLAMA_API_KEY')
        print("API Key:", self.api_key)
        if not self.api_key:
            raise ValueError("API key is not set in the environment variables")

        # Do not log the API key for security reasons
        logger.info("LlamaService initialized")

        self.client = Groq(api_key=self.api_key)
        logger.info("Groq client initialized")

    @retry_with_backoff(max_retries=3, backoff_in_seconds=2)
    def analyze(self, prompt: str) -> Dict[str, Any]:
        try:
            sanitized_prompt = self._sanitize_input(prompt)
            logger.debug(f"Sending request to Groq API with prompt: {sanitized_prompt[:500]}...")

            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI assistant specialized in analyzing resumes and job descriptions. "
                            "Provide detailed and accurate analyses, ensuring all fields are populated with relevant information. "
                            "Use the full range of scores from 0 to 100, and be critical yet fair in your evaluations."
                        ),
                    },
                    {
                        "role": "user",
                        "content": sanitized_prompt,
                    }
                ],
                model="llama-3.1-8b-instant",
                max_tokens=1000,
                temperature=0.7,
            )

            logger.debug(f"Received response from Groq API")

            if not completion.choices:
                raise ValueError("No choices returned from Groq API")

            result = completion.choices[0].message.content.strip()
            logger.debug(f"Raw response from Groq: {result}")

            return {"analysis": result}

        except Exception as e:
            logger.error(f"Error during Groq API analysis: {str(e)}", exc_info=True)
            raise

    @retry_with_backoff(max_retries=3, backoff_in_seconds=2)
    def analyze_resume(self, resume_text: str, job_description: str, job_title: str) -> Dict[str, Any]:
        prompt = self._build_analyze_resume_prompt(resume_text, job_description, job_title)

        try:
            sanitized_prompt = self._sanitize_input(prompt)
            logger.debug(f"Sending request to Groq API with prompt length: {len(sanitized_prompt)} characters")

            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                        "You are an AI assistant specialized in analyzing resumes and job descriptions. "
                        "Provide detailed and accurate analyses, ensuring all fields are populated with relevant information. "
                        "Use the full range of scores from 0 to 100, and be critical yet fair in your evaluations."
                        ),
                    },
                    {
                        "role": "user",
                        "content": sanitized_prompt,
                    }
                ],
                model="llama-3.1-8b-instant",
                max_tokens=2000,  # Increased from 1000 to 2000
                temperature=0.5,
            )

            logger.debug(f"Received response from Groq API")

            if not completion.choices:
                raise ValueError("No choices returned from Groq API")

            result = completion.choices[0].message.content.strip()
            logger.debug(f"Raw response from Groq: {result}")

            # Attempt to parse the result as JSON
            try:
                parsed_result = json.loads(result)
            except json.JSONDecodeError:
                logger.warning("Failed to parse result as JSON. Using raw text.")
                parsed_result = {"analysis": result}

            return parsed_result

        except Exception as e:
            logger.error(f"Error during Groq API analysis: {str(e)}", exc_info=True)
            raise

    def _build_analyze_resume_prompt(self, resume_text: str, job_description: str, job_title: str) -> str:
        prompt = (
            f"Analyze the following resume against the provided job description for a {job_title} role.\n\n"
            f"Resume:\n{resume_text}\n\n"
            f"Job Description:\n{job_description}\n\n"
            "Provide a detailed analysis covering:\n"
            "1. Match Score (0-100): Assess how well the candidate's skills and experience match the job requirements.\n"
            "2. Brief Summary: Provide a 2-3 sentence overview of the candidate's fit for the role.\n"
            "3. Experience and Project Relevance: Analyze how the candidate's past experiences and projects align with the job requirements.\n"
            "4. Skills Gap: Identify important skills or qualifications mentioned in the job description that the candidate lacks.\n"
            "5. Key Strengths: List 3-5 specific strengths of the candidate relevant to this role.\n"
            "6. Areas for Improvement: Suggest 2-3 areas where the candidate could improve to better fit the role.\n"
            "7. Recruiter Questions: Suggest 3-5 specific questions for the recruiter to ask the candidate.\n\n"
            "Provide your analysis in a structured JSON format."
        )
        return prompt

    def _sanitize_input(self, text: str) -> str:
        """Sanitize input text to prevent injection attacks and remove any unwanted characters."""
        sanitized_text = text.replace('\r', ' ').replace('\n', ' ').strip()
        return sanitized_text
    
llama_service = LlamaService()
