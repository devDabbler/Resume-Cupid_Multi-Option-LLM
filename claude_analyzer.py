import requests
import logging
import json
import re
import simplejson
import ast
import ujson
from typing import Dict, Any

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClaudeAPI:
    def __init__(self, api_key: str, api_url: str = "https://api.anthropic.com/v1/messages"):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        logger.info("ClaudeAPI initialized")

    def analyze(self, prompt: str) -> Dict[str, Any]:
        try:
            logger.debug("Sending request to Claude API")
            payload = {
                "model": "claude-3-5-sonnet-20240620",
                "max_tokens": 2000,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()

            result = response.json()
            logger.debug(f"Received response from Claude: {result}")

            # Extract the content from the Claude API response
            if 'content' in result and result['content']:
                content = result['content'][0]['text']
            else:
                error_message = "No content found in Claude API response"
                logger.error(error_message)
                return self._generate_error_response(error_message)
                logger.debug(f"Raw content from Claude: {content[:100]}...")  # Log first 100 characters

                # Print the full raw content
                print("Full raw content:")
                print(content)

                # Define a function to clean the JSON string
                def clean_json_string(s):
                    # Remove any invalid characters
                    s = re.sub(r'[^\x00-\x7F]+', '', s)
                    # Remove any control characters
                    s = re.sub(r'[\x00-\x1F\x7F]', '', s)
                    # Remove specific problematic characters
                    s = s.replace('\u2028', '').replace('\u2029', '')
                    return s

                # Clean the JSON string
                cleaned_result = clean_json_string(content)
                print("Cleaned JSON:")
                print(cleaned_result)

                # Try parsing with different methods
            try:
                parsed_content = json.loads(cleaned_result)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response with json: {str(e)}")
                try:
                    parsed_content = ast.literal_eval(cleaned_result)
                except (ValueError, SyntaxError) as e:
                    logger.error(f"Error parsing response with ast.literal_eval: {str(e)}")
                    logger.error(f"Problematic JSON: {result}")
                    # Print the problematic part
                    error_line = e.lineno if hasattr(e, 'lineno') else 0
                    error_col = e.colno if hasattr(e, 'colno') else 0
                    problematic_part = result.split('\n')[error_line-1:error_line+1]
                    logger.error(f"Problematic part: {problematic_part}")
                    return self._generate_error_response(f"Error parsing JSON response: {str(e)}")

                logger.debug(f"Parsed content: {parsed_content}")
                
                # Ensure all required fields are present and populated
                required_fields = [
                    'brief_summary', 'match_score', 'recommendation_for_interview',
                    'experience_and_project_relevance', 'skills_gap', 'recruiter_questions'
                ]
                for field in required_fields:
                    if field not in parsed_content or not parsed_content[field]:
                        parsed_content[field] = self._generate_fallback_content(field, parsed_content)
                        logger.warning(f"Field '{field}' was missing or empty in API response, generated fallback content")
                
                # Ensure match_score is an integer
                try:
                    parsed_content['match_score'] = int(parsed_content['match_score'])
                except (ValueError, TypeError):
                    logger.error(f"Invalid match_score value: {parsed_content['match_score']}")
                    parsed_content['match_score'] = 0

                # Add recommendation based on match_score
                match_score = parsed_content['match_score']
                if match_score < 50:
                    recommendation = "Do not recommend for interview"
                elif 50 <= match_score < 65:
                    recommendation = "Recommend for interview with significant reservations (soft match)"
                elif 65 <= match_score < 80:
                    recommendation = "Recommend for interview with reservations"
                elif 80 <= match_score < 90:
                    recommendation = "Recommend for interview"
                else:
                    recommendation = "Highly recommend for interview"
                
                parsed_content['recommendation'] = recommendation
                logger.info(f"Analysis completed successfully for prompt: {prompt[:50]}...")
                return parsed_content
            else:
                logger.error("No content found in Claude API response")
                raise ValueError("No content found in Claude API response")
    
        except requests.exceptions.RequestException as e:
            logger.error(f"Claude API request failed: {str(e)}")
            return self._generate_error_response(str(e))
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing Claude API response: {str(e)}")
            return self._generate_error_response(str(e))
        except Exception as e:
            logger.error(f"Unexpected error in Claude analysis: {str(e)}")
            return self._generate_error_response(str(e))

    def analyze_match(self, resume: str, job_description: str, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Starting resume analysis")
        prompt = f"""
        Analyze the fit between the following resume and job description with extreme accuracy. 
        Focus specifically on how well the candidate's skills and experience match the job requirements.
    
        Provide a detailed analysis for each of the following areas:

        1. Brief Summary: Provide a concise overview of the candidate's fit for the role in 2-3 sentences. This is mandatory and must always be included.
        2. Match Score: Provide a percentage between 0 and 100, where 0% means the candidate has none of the required skills or experience, and 100% means the candidate perfectly matches all job requirements. Be very critical and realistic in this scoring.
        3. Recommendation for Interview: Based on the match score, provide a recommendation (e.g., "Highly recommend", "Recommend", "Recommend with reservations", "Do not recommend").
        4. Experience and Project Relevance: Provide a comprehensive analysis of the candidate's work experience and relevant projects, specifically relating them to the job requirements. This section is crucial and must always contain detailed information.
        5. Skills Gap: List all important skills or qualifications mentioned in the job description that the candidate lacks. Be exhaustive in this analysis.
        6. Recruiter Questions: Suggest 3-5 specific questions for the recruiter to ask the candidate based on their resume and the job requirements. These questions are mandatory and must always be included.

        Format your response as JSON with the following structure:
        {{
        "brief_summary": "Your brief summary here",
        "match_score": The percentage match (0-100),
        "recommendation_for_interview": "Your recommendation here",
        "experience_and_project_relevance": "Your detailed analysis here",
        "skills_gap": ["Skill 1", "Skill 2", ...],
        "recruiter_questions": ["Question 1?", "Question 2?", ...]
        }}

        Ensure that all fields are populated with relevant, detailed information.

        Candidate Data:
        {json.dumps(candidate_data)}

        Job Description:
        {job_description}

        Resume:
        {resume}

        JSON Response:
        """

        return self.analyze(prompt)

    def _generate_fallback_content(self, field: str, parsed_content: Dict[str, Any]) -> str:
        if field == 'brief_summary':
            return f"Based on the match score of {parsed_content.get('match_score', 'N/A')}, the candidate appears to be a {parsed_content.get('recommendation', 'potential')} fit for the role. Further analysis is needed to determine specific strengths and weaknesses."
        elif field == 'experience_and_project_relevance':
            return "Unable to provide a detailed analysis of experience and project relevance. A manual review of the resume is recommended to assess the candidate's fit for the role."
        elif field == 'skills_gap':
            return ["Unable to determine specific skills gap. A manual comparison of the resume against the job requirements is recommended."]
        elif field == 'recruiter_questions':
            return ["What specific experience do you have that relates to this role?", "Can you describe any relevant projects you've worked on?", "How do you stay updated with the latest developments in your field?"]
        else:
            return f"No {field.replace('_', ' ')} available"

    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        logger.warning(f"Generating error response: {error_message}")
        return {
            "error": error_message,
            "brief_summary": "Unable to complete analysis due to an error.",
            "match_score": 0,
            "recommendation_for_interview": "Unable to provide a recommendation due to an error.",
            "experience_and_project_relevance": "Unable to assess experience and project relevance due to an error.",
            "skills_gap": ["Unable to determine skills gap due to an error."],
            "recruiter_questions": ["Unable to generate recruiter questions due to an error."]
        }
