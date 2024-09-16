import requests
import logging
import json
import re
from typing import Dict, Any
from logger import get_logger
from groq import Groq
import os

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
            logger.debug(f"Sending request to Llama API with prompt: {prompt[:100]}...")
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
            logger.debug(f"Parsed content: {parsed_content}")

            processed_content = self._process_parsed_content(parsed_content)
            logger.debug(f"Processed content: {processed_content}")

            logger.info(f"Analysis completed successfully for prompt: {prompt[:50]}...")
            return processed_content

        except Exception as e:
            logger.error(f"Llama API request failed: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(response)
            logger.debug(f"Successfully parsed JSON response: {parsed}")
            return parsed
        except json.JSONDecodeError as json_error:
            logger.error(f"JSON parsing error: {str(json_error)}")
            logger.debug(f"Problematic JSON: {response}")
            
            try:
                cleaned_json = re.search(r'\{.*\}', response, re.DOTALL).group()
                parsed = json.loads(cleaned_json)
                logger.debug(f"Successfully parsed cleaned JSON: {parsed}")
                return parsed
            except (AttributeError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse JSON even after cleaning: {str(e)}")
                return {}

    def _process_parsed_content(self, parsed_content: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"Processing parsed content: {parsed_content}")
        processed_content = {}
    
        for key, value in parsed_content.items():
            processed_key = key.lower().replace(' ', '_')
            processed_content[processed_key] = value
            logger.debug(f"Processed key-value pair: {processed_key} = {value}")
    
        required_fields = [
            'brief_summary', 'match_score', 'recommendation_for_interview',
            'experience_and_project_relevance', 'skills_gap', 'recruiter_questions'
        ]
    
        for field in required_fields:
            if field not in processed_content:
                processed_content[field] = self._generate_fallback_content(field)
                logger.warning(f"Generated fallback content for missing field: {field}")
    
        try:
            if isinstance(processed_content['match_score'], dict):
                # Calculate a weighted average if match_score is a dictionary
                total_score = sum(processed_content['match_score'].values())
                total_weight = len(processed_content['match_score'])
                processed_content['match_score'] = int(total_score / total_weight)
            elif isinstance(processed_content['match_score'], str):
                # Try to extract a number from the string
                match = re.search(r'\d+', processed_content['match_score'])
                if match:
                    processed_content['match_score'] = int(match.group())
                else:
                    processed_content['match_score'] = 0
            processed_content['match_score'] = int(float(processed_content['match_score']))
            # Adjust the score to be more stringent
            processed_content['match_score'] = max(0, min(100, int(processed_content['match_score'] * 0.8)))
            logger.debug(f"Processed match_score: {processed_content['match_score']}")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid match_score value: {processed_content.get('match_score')}")
            processed_content['match_score'] = 0
    
        processed_content['recommendation'] = self._get_recommendation(processed_content['match_score'])
        logger.debug(f"Generated recommendation: {processed_content['recommendation']}")
    
        return processed_content

    def _generate_fallback_content(self, field: str) -> str:
        fallback = f"No {field.replace('_', ' ')} available"
        logger.debug(f"Generated fallback content for {field}: {fallback}")
        return fallback

    def _get_recommendation(self, match_score: int) -> str:
        if match_score < 30:
            return "Do not recommend for interview"
        elif 30 <= match_score < 50:
            return "Recommend for interview with significant reservations"
        elif 50 <= match_score < 70:
            return "Recommend for interview with minor reservations"
        elif 70 <= match_score < 85:
            return "Recommend for interview"
        else:
            return "Highly recommend for interview"

    def analyze_match(self, resume: str, job_description: str, candidate_data: Dict[str, Any], job_title: str) -> Dict[str, Any]:
        logger.debug(f"Analyzing match for candidate: {candidate_data.get('file_name', 'Unknown')}")
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

        Provide your analysis in a structured JSON format with the following keys:
        "brief_summary", "match_score", "recommendation_for_interview", "experience_and_project_relevance", "skills_gap", "recruiter_questions"
        """

        result = self.analyze(prompt)
        result['file_name'] = candidate_data.get('file_name', 'Unknown')
        logger.debug(f"Analysis result for {result['file_name']}: {result}")
        return result

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
        pass

def initialize_llm():
    """
    Initializes and returns the LlamaAPI instance.
    
    Returns:
    - LlamaAPI: An instance of the LlamaAPI class.
    """
    llama_api_key = os.getenv("LLAMA_API_KEY")
    if not llama_api_key:
        raise ValueError("LLAMA_API_KEY is not set in the environment variables.")
    return LlamaAPI(api_key=llama_api_key)