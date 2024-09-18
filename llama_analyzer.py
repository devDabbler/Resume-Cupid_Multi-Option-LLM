import json
import re
from typing import Dict, Any, List, Optional
from functools import lru_cache
from groq import Groq
import os
from logger import get_logger

logger = get_logger(__name__)

class LlamaAPI:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        logger.info(f"Initialized LlamaAPI with API key: {api_key[:5]}...")

    @lru_cache(maxsize=100)
    def analyze(self, prompt: str) -> Dict[str, Any]:
        try:
            logger.debug(f"Sending request to Llama API with prompt: {prompt[:100]}...")
            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant specialized in analyzing resumes and job descriptions for Machine Learning Operations Engineering roles. Always provide your responses in JSON format. Use the full range of scores from 0 to 100, and be very critical in your evaluations.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.1-8b-instant",
                max_tokens=4000,
                temperature=0.5,
            )
            result = completion.choices[0].message.content
            logger.debug(f"Raw response from Llama: {result}")

            parsed_content = self._parse_json_response(result)
            processed_content = self._process_parsed_content(parsed_content)

            logger.info(f"Analysis completed successfully for prompt: {prompt[:50]}...")
            return processed_content
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            return self._generate_error_response(str(e))

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"JSON parsing error. Attempting to clean and parse.")
            try:
                cleaned_json = re.search(r'\{.*\}', response, re.DOTALL)
                if cleaned_json:
                    return json.loads(cleaned_json.group())
            except Exception as e:
                logger.error(f"Failed to parse JSON even after cleaning: {str(e)}")
            return {}

    def _process_parsed_content(self, parsed_content: Dict[str, Any]) -> Dict[str, Any]:
        processed_content = {
            'brief_summary': parsed_content.get('brief_summary', "No summary available"),
            'match_score': self._process_match_score(parsed_content.get('match_score', 0)),
            'recommendation_for_interview': parsed_content.get('recommendation_for_interview', "Unable to provide a recommendation"),
            'experience_and_project_relevance': parsed_content.get('experience_and_project_relevance', "No relevance information available"),
            'skills_gap': parsed_content.get('skills_gap', []),
            'key_strengths': parsed_content.get('key_strengths', []),
            'areas_for_improvement': parsed_content.get('areas_for_improvement', []),
            'recruiter_questions': parsed_content.get('recruiter_questions', [])
        }

        processed_content['recommendation'] = self._get_recommendation(processed_content['match_score'])
        processed_content['brief_summary'] = self._generate_brief_summary(processed_content['match_score'])

        return processed_content

    def _process_match_score(self, match_score: Any) -> int:
        if isinstance(match_score, dict):
            return int(sum(match_score.values()) / len(match_score))
        elif isinstance(match_score, str):
            match = re.search(r'\d+', match_score)
            return int(match.group()) if match else 0
        elif isinstance(match_score, (int, float)):
            return int(match_score)
        return 0

    def _get_recommendation(self, match_score: int) -> str:
        if match_score < 30:
            return "Do not recommend for interview"
        elif match_score < 50:
            return "Recommend for interview with significant reservations"
        elif match_score < 70:
            return "Recommend for interview with minor reservations"
        elif match_score < 85:
            return "Recommend for interview"
        else:
            return "Highly recommend for interview"

    def _generate_brief_summary(self, match_score: int) -> str:
        if match_score < 50:
            return f"The candidate is not a strong fit for the role. With a match score of {match_score}%, there are significant gaps in required skills and experience."
        elif match_score < 65:
            return f"The candidate shows some potential, but with a match score of {match_score}%, there are considerable gaps in meeting the requirements. Further evaluation is needed."
        else:
            return f"The candidate is a good fit for the role. With a match score of {match_score}%, they demonstrate strong alignment with the required skills and experience."

    def analyze_match(self, resume: str, job_description: str, candidate_data: Dict[str, Any], job_title: str) -> Dict[str, Any]:
        if not resume or not job_description:
            return self._generate_error_response("Empty resume or job description provided")

        prompt = self._generate_analysis_prompt(resume, job_description, job_title)
        result = self.analyze(prompt)
        result['file_name'] = candidate_data.get('file_name', 'Unknown')
        return result

    def _generate_analysis_prompt(self, resume: str, job_description: str, job_title: str) -> str:
        return f"""
        Analyze the following resume against the provided job description for a {job_title} role. Provide a detailed evaluation covering:

        1. Match Score (0-100): Assess how well the candidate's skills and experience match the job requirements.
        2. Brief Summary: Provide a 2-3 sentence overview of the candidate's fit for the role.
        3. Experience and Project Relevance: Analyze how the candidate's past experiences and projects align with the job requirements.
        4. Skills Gap: Identify any important skills or qualifications mentioned in the job description that the candidate lacks.
        5. Key Strengths: List 3-5 specific strengths of the candidate relevant to this role.
        6. Areas for Improvement: Suggest 2-3 areas where the candidate could improve to better fit the role.
        7. Recruiter Questions: Suggest 3-5 specific questions for the recruiter to ask the candidate based on their resume and the job requirements.

        Resume:
        {resume}

        Job Description:
        {job_description}

        Provide your analysis in a structured JSON format with the following keys:
        "match_score", "brief_summary", "experience_and_project_relevance", "skills_gap", "key_strengths", "areas_for_improvement", "recruiter_questions"
        """

    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        logger.warning(f"Generating error response: {error_message}")
        return {
            "error": error_message,
            "brief_summary": "Unable to complete analysis due to an error.",
            "match_score": 0,
            "recommendation_for_interview": "Unable to provide a recommendation due to an error.",
            "experience_and_project_relevance": "Error occurred during analysis. Unable to assess experience and project relevance.",
            "skills_gap": "Error in analysis. Manual skill gap assessment needed.",
            "recruiter_questions": ["Due to an error, we couldn't generate specific questions. Please review the resume and job description to formulate relevant questions."]
        }

def initialize_llm() -> LlamaAPI:
    llama_api_key = os.getenv("LLAMA_API_KEY")
    if not llama_api_key:
        raise ValueError("LLAMA_API_KEY is not set in the environment variables.")
    return LlamaAPI(api_key=llama_api_key)

# Main execution
if __name__ == "__main__":
    llm = initialize_llm()
    # Add any test or example usage here