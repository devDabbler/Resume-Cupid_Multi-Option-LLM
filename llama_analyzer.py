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
            logger.debug(f"Sending request to Llama API with prompt: {prompt[:500]}...")
            logger.debug(f"API Key: {self.client.api_key[:5]}...")  # Log first 5 characters of API key
        
            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant specialized in analyzing resumes and job descriptions. Provide detailed and accurate analyses, ensuring all fields are populated with relevant information. Use the full range of scores from 0 to 100, and be critical yet fair in your evaluations.",
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
        
            logger.debug(f"Received response from Llama API: {completion}")
        
            if not completion.choices:
                raise ValueError("No choices returned from Llama API")
        
            result = completion.choices[0].message.content
            logger.debug(f"Raw response from Llama: {result}")

            parsed_content = self._parse_json_response(result)
            logger.debug(f"Parsed content: {json.dumps(parsed_content, indent=2)}")

            processed_content = self._process_parsed_content(parsed_content)
            logger.debug(f"Processed content: {json.dumps(processed_content, indent=2)}")

            return processed_content
        except Exception as e:
            logger.error(f"Error during Llama API analysis: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"JSON parsing error. Attempting to clean and parse.")
            try:
                # Remove any text before the first '{' and after the last '}'
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
        logger.debug(f"Entering analyze_match method in LlamaAPI")
        if not resume or not job_description:
            logger.error("Empty resume or job description provided")
            return self._generate_error_response("Empty resume or job description provided")

        prompt = f"""
        Analyze the following resume against the provided job description for a {job_title} role. 
        Pay special attention to the candidate's experience and skills related to AI and machine learning, as these are critical for this role.

        Provide a detailed analysis covering:
        1. Match Score (0-100): Assess how well the candidate's skills and experience match the job requirements. Be very critical in your scoring.
        2. Brief Summary: Provide a 2-3 sentence overview of the candidate's fit for the role.
        3. Experience and Project Relevance: Analyze how the candidate's past experiences and projects align with the job requirements. Provide numerical scores for overall relevance, relevant experience, project relevance, and technical skills relevance.
        4. Skills Gap: Identify any important skills or qualifications mentioned in the job description that the candidate lacks. This is crucial for determining the candidate's suitability.
        5. Key Strengths: List 3-5 specific strengths of the candidate relevant to this role.
        6. Areas for Improvement: Suggest 2-3 areas where the candidate could improve to better fit the role.
        7. Recruiter Questions: Suggest 3-5 specific questions for the recruiter to ask the candidate based on their resume and the job requirements.

        Resume:
        {resume}

        Job Description:
        {job_description}

        Provide your analysis in a structured JSON format.
        """

        try:
            logger.debug("Calling self.analyze method")
            result = self.analyze(prompt)
            logger.debug(f"Result from self.analyze: {json.dumps(result, indent=2)}")
            result['file_name'] = candidate_data.get('file_name', 'Unknown')
            return result
        except Exception as e:
            logger.error(f"Error in analyze_match: {str(e)}", exc_info=True)
            return self._generate_error_response(f"Error in analyze_match: {str(e)}")

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