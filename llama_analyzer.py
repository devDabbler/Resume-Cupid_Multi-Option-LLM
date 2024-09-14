import requests
import logging
import json
import re
import ast
from typing import Dict, Any, List
from logger import get_logger
from groq import Groq

# Initialize logging
logger = get_logger(__name__)

class LlamaAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = Groq(api_key=api_key)
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"Initialized LlamaAPI with API key: {api_key[:5]}...")

    def analyze(self, prompt: str) -> Dict[str, Any]:
        try:
            logger.debug("Sending request to Llama API")
            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant specialized in analyzing resumes and job descriptions for Machine Learning Operations Engineering roles. Always provide your responses in JSON format. Use the full range of scores from 0 to 100, and be very critical in your evaluations, especially regarding model governance, risk management, and compliance in financial services.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.1-8b-instant",
                max_tokens=4000,
                temperature=0.7,
            )
            logger.debug(f"Received response from Llama: {completion}")

            result = completion.choices[0].message.content
            logger.debug(f"Raw response from Llama: {result}")

            parsed_content = self._parse_json_response(result)
            processed_content = self._process_parsed_content(parsed_content)

            logger.info(f"Analysis completed successfully for prompt: {prompt[:50]}...")
            return processed_content

        except Exception as e:
            logger.error(f"Llama API request failed: {str(e)}")
            return self._generate_error_response(str(e))

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            # Try to parse the JSON response
            return json.loads(response)
        except json.JSONDecodeError as json_error:
            logger.error(f"JSON parsing error: {str(json_error)}")
            logger.debug(f"Problematic JSON: {response}")
            
            # Attempt to fix common JSON errors
            try:
                # Remove any text before the first '{' and after the last '}'
                cleaned_json = re.search(r'\{.*\}', response, re.DOTALL).group()
                return json.loads(cleaned_json)
            except (AttributeError, json.JSONDecodeError):
                logger.error("Failed to parse JSON even after cleaning")
                return {}

    def _process_parsed_content(self, parsed_content: Dict[str, Any]) -> Dict[str, Any]:
        processed_content = {}
        
        # Flatten nested structures and ensure all values are strings
        for key, value in parsed_content.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    processed_content[f"{key}_{sub_key}"] = self._ensure_string(sub_value)
            elif isinstance(value, list):
                processed_content[key] = [self._ensure_string(item) for item in value]
            else:
                processed_content[key] = self._ensure_string(value)
        
        # Ensure required fields are present
        required_fields = [
            'brief_summary', 'match_score', 'recommendation_for_interview',
            'experience_and_project_relevance', 'skills_gap', 'recruiter_questions'
        ]
        
        for field in required_fields:
            if field not in processed_content:
                processed_content[field] = self._generate_fallback_content(field)
        
        # Ensure match_score is an integer
        try:
            processed_content['match_score'] = int(float(processed_content.get('match_score', 0)))
        except (ValueError, TypeError):
            logger.error(f"Invalid match_score value: {processed_content.get('match_score')}")
            processed_content['match_score'] = 0
        
        # Generate recommendation based on match_score
        processed_content['recommendation'] = self._get_recommendation(processed_content['match_score'])
        
        return processed_content

    def _ensure_string(self, value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)

    def _generate_fallback_content(self, field: str) -> str:
        return f"No {field.replace('_', ' ')} available"

    def _get_recommendation(self, match_score: int) -> str:
        if match_score < 20:
            return "Do not recommend for interview"
        elif 20 <= match_score < 40:
            return "Recommend for interview with significant reservations"
        elif 40 <= match_score < 60:
            return "Recommend for interview with minor reservations"
        elif 60 <= match_score < 80:
            return "Recommend for interview"
        else:
            return "Highly recommend for interview"

    def analyze_match(self, resume: str, job_description: str, candidate_data: Dict[str, Any], job_title: str) -> Dict[str, Any]:
        prompt = f"""
        Analyze the fit between the following resume and job description for the role of {job_title}. 
        Provide a detailed evaluation covering these key areas:

        1. Brief Summary: Provide a concise overview of the candidate's fit for the role in 2-3 sentences.
        2. Match Score: Assign a percentage (0-100%) indicating how well the candidate matches the job requirements.
        3. Recommendation for Interview: Based on the analysis, provide a clear recommendation.
        4. Experience and Project Relevance: Analyze how the candidate's past experiences and projects align with the job requirements.
        5. Skills Gap: Identify any important skills or qualifications mentioned in the job description that the candidate lacks.
        6. Recruiter Questions: Suggest 3-5 specific questions for the recruiter to ask the candidate based on their resume and the job requirements.

        Resume:
        {resume}

        Job Description:
        {job_description}

        Provide your analysis in a structured JSON format.
        """

        return self.analyze(prompt)

    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        logger.warning(f"Generating error response: {error_message}")
        return {
            "error": error_message,
            "brief_summary": "Unable to complete analysis due to an error.",
            "match_score": 0,
            "recommendation_for_interview": "Unable to provide a recommendation due to an error.",
            "experience_and_project_relevance": "Error occurred during analysis. Unable to assess experience and project relevance.",
            "skills_gap": "Error in analysis. Manual skill gap assessment needed.",
            "recruiter_questions": "Due to an error in our system, we couldn't generate specific questions. Please review the resume and job description to formulate relevant questions."
        }

    def clear_cache(self):
        # Llama doesn't have a built-in cache, so this method can be empty
        pass