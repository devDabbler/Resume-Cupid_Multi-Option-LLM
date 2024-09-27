import time
import logging
import functools
from typing import Dict, Any
import os
import json
import re
from groq import Groq

logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries=3, backoff_in_seconds=1):
    def retry_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    time.sleep(backoff_in_seconds * retries)
                    if retries == max_retries:
                        raise e
        return wrapper
    return retry_decorator

class LlamaService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv('LLAMA_API_KEY')
        if not self.api_key:
            raise ValueError("API key is not set in the environment variables")
        self.logger.info("LlamaService initialized")
        self.client = Groq(api_key=self.api_key)
        self.logger.info("Groq client initialized")

    @retry_with_backoff(max_retries=3, backoff_in_seconds=2)
    def analyze_resume(self, resume_text: str, job_description: str, job_title: str) -> Dict[str, Any]:
        prompt = self._build_analyze_resume_prompt(resume_text, job_description, job_title)

        try:
            sanitized_prompt = self._sanitize_input(prompt)
            self.logger.debug(f"Sending request to Groq API with prompt length: {len(sanitized_prompt)} characters")

            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI assistant specialized in analyzing resumes and job descriptions. "
                            "Provide detailed and accurate analyses, ensuring all fields are populated with relevant information. "
                            "Use the full range of scores from 0 to 100, and be critical yet fair in your evaluations. "
                            "Always return your response in valid JSON format."
                        ),
                    },
                    {
                        "role": "user",
                        "content": sanitized_prompt,
                    }
                ],
                model="llama-3.1-8b-instant",
                max_tokens=4000,
                temperature=0.5,
            )

            self.logger.debug(f"Received response from Groq API")

            if not completion.choices:
                raise ValueError("No choices returned from Groq API")

            result = completion.choices[0].message.content.strip()
            self.logger.debug(f"Raw response from Groq: {result}")

            # Attempt to parse the result as JSON
            try:
                parsed_result = self._parse_json_safely(result)
                self.logger.debug("Successfully parsed JSON response.")
                return parsed_result
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parsing error: {e}")
                self.logger.debug(f"Assistant's raw response: {result}")
                # Fall back to manual extraction
                return self._extract_information_manually(result)

        except Exception as e:
            self.logger.error(f"Error during Groq API analysis: {str(e)}", exc_info=True)
            raise

    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        # Try to find JSON object in the text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If still fails, try to fix common JSON errors
                fixed_json_str = self._fix_json_string(json_str)
                return json.loads(fixed_json_str)
        raise ValueError("No valid JSON found in the text")

    def _fix_json_string(self, json_str: str) -> str:
        # Replace single quotes with double quotes
        json_str = json_str.replace("'", '"')
        # Add quotes to unquoted keys
        json_str = re.sub(r'(\w+)(?=\s*:)', r'"\1"', json_str)
        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        return json_str

    def _extract_information_manually(self, text: str) -> Dict[str, Any]:
        result = {}
    
        # Extract Match Score
        match_score = re.search(r'"Match Score":\s*(\d+)', text)
        result['Match Score'] = int(match_score.group(1)) if match_score else 0
    
        # Extract Brief Summary
        brief_summary = re.search(r'"Brief Summary":\s*"([^"]*)"', text)
        result['Brief Summary'] = brief_summary.group(1) if brief_summary else "No summary available"
    
        # Extract Experience and Project Relevance
        experience_relevance = re.search(r'"Experience and Project Relevance":\s*\{([^}]*)\}', text)
        if experience_relevance:
            entries = re.findall(r'"([^"]+)":\s*"([^"]+)"', experience_relevance.group(1))
            result['Experience and Project Relevance'] = {k.strip(): v.strip() for k, v in entries}
        else:
            result['Experience and Project Relevance'] = {}
    
        # Extract Skills Gap
        skills_gap = re.search(r'"Skills Gap":\s*\[([^\]]*)\]', text)
        result['Skills Gap'] = [s.strip(' "').strip() for s in skills_gap.group(1).split(',')] if skills_gap else []
    
        # Extract Key Strengths
        key_strengths = re.search(r'"Key Strengths":\s*\[([^\]]*)\]', text)
        result['Key Strengths'] = [s.strip(' "').strip() for s in key_strengths.group(1).split(',')] if key_strengths else []
    
        # Extract Areas for Improvement
        areas_for_improvement = re.search(r'"Areas for Improvement":\s*\[([^\]]*)\]', text)
        result['Areas for Improvement'] = [s.strip(' "').strip() for s in areas_for_improvement.group(1).split(',')] if areas_for_improvement else []
    
        # Extract Recruiter Questions
        recruiter_questions = re.search(r'"Recruiter Questions":\s*\[([^\]]*)\]', text)
        result['Recruiter Questions'] = [s.strip(' "').strip() for s in recruiter_questions.group(1).split(',')] if recruiter_questions else []
    
        self.logger.debug(f"Manually extracted information: {result}")
        return result
    
    def _build_analyze_resume_prompt(self, resume_text: str, job_description: str, job_title: str) -> str:
        prompt = f"""
        Analyze the candidate's resume for the {job_title} position based on the provided job description. 
        Focus on identifying specific skills and experiences relevant to the {job_title} role.
        Pay special attention to the candidate's experience with key skills and technologies mentioned in the job description.
        
        Be critical in your assessment and provide a realistic evaluation of the candidate's fit for this specific {job_title} role.
        
        Return the analysis in JSON format with the following structure:

        {{
            "Raw Match Score": int,  # An integer between 0 and 100, based solely on general skill matching
            "Brief Summary": "string",
            "Experience and Project Relevance": {{
                "Job Title": {{
                    "Project Name": "Relevance description (score/10)"
                }}
            }},
            "Skills Assessment": {{
                "Technical Skills": ["skill1", "skill2"],
                "Soft Skills": ["skill1", "skill2"],
                "Domain Knowledge": ["skill1", "skill2"]
            }},
            "Missing Critical Skills": ["skill1", "skill2"],
            "Key Strengths": ["strength1", "strength2"],
            "Areas for Improvement": ["area1", "area2"],
            "Recruiter Questions": [
                {{
                    "question": "Detailed question text",
                    "purpose": "Brief explanation of what this question aims to uncover"
                }}
            ]
        }}

        Ensure your response is in valid JSON format. Do not include any explanation or additional text outside the JSON structure.
        For the "Experience and Project Relevance" section, always include a score out of 10 in parentheses at the end of each description, e.g., "Relevant project (7/10)".
        In the "Skills Assessment" section, be thorough and list all relevant skills found in the resume, even if they are not explicitly mentioned in the job description.
        For "Recruiter Questions", provide 3-5 detailed questions that will help uncover the candidate's true abilities, especially for skills or experiences that are not clear from the resume. Focus on questions that will help determine if the candidate can bridge any identified skills gaps.

        Candidate Resume:
        {resume_text}

        Job Description:
        {job_description}
        """
        return prompt

    def _sanitize_input(self, text: str) -> str:
        sanitized_text = text.replace('\r', ' ').replace('\n', ' ').strip()
        return sanitized_text   

llama_service = LlamaService()
