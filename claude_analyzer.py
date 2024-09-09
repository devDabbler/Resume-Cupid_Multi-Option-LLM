import requests
import logging
import json
import re
import ast
from typing import Dict, Any, List
from utils import get_logger

logger = get_logger(__name__)

class ClaudeClient:
    def __init__(self, api_key: str, api_url: str = "https://api.anthropic.com/v1/messages"):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        logger.debug(f"ClaudeAPI initialized with headers: {self.headers}")

    def create_completion(self, messages: list, model: str = "claude-3-5-sonnet-20240620", max_tokens: int = 2000):
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens
        }
        try:
            logger.debug(f"Sending request to {self.api_url}")
            logger.debug(f"Headers: {self.headers}")
            logger.debug(f"Data: {data}")
            response = requests.post(self.api_url, headers=self.headers, json=data)
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response content: {response.text[:100]}...")  # First 100 chars
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error in create_completion: {str(e)}")
            raise

class ClaudeAPI:
    def __init__(self, api_key: str):
        logger.debug(f"Initializing ClaudeAPI with API key: {api_key[:5]}...")  # Log first 5 characters of API key
        try:
            self.client = ClaudeClient(api_key)
            logger.info("ClaudeAPI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ClaudeAPI: {str(e)}")
            raise
        
    def analyze(self, prompt: str) -> Dict[str, Any]:
        try:
            logger.debug("Sending request to Claude API")
            messages = [{"role": "user", "content": prompt}]
            completion = self.client.create_completion(messages)
            logger.debug(f"Received response from Claude: {completion}")

            result = completion['content'][0]['text']
            logger.debug(f"Raw response from Claude: {result}")

            # Clean and parse JSON
            cleaned_result = self._clean_json_string(result)
            parsed_content = self._parse_json(cleaned_result)

            if parsed_content is None:
                raise ValueError("Failed to parse JSON response")

            return self._process_parsed_content(parsed_content)

        except Exception as e:
            logger.error(f"Claude API request failed: {str(e)}")
            return self._generate_error_response(str(e))

    def _clean_json_string(self, s: str) -> str:
        # Remove any text before the first '{' and after the last '}'
        start = s.find('{')
        end = s.rfind('}')
        if start != -1 and end != -1:
            s = s[start:end+1]
        # Remove any non-ASCII characters
        s = ''.join(char for char in s if ord(char) < 128)
        # Remove any newlines or extra whitespace
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    def _parse_json(self, cleaned_result: str) -> Dict[str, Any]:
        logger.debug(f"Attempting to parse JSON: {cleaned_result}")
        try:
            return json.loads(cleaned_result)
        except json.JSONDecodeError as json_err:
            logger.error(f"JSON parsing error: {str(json_err)}")
            try:
                # Try to fix common JSON errors
                fixed_json = re.sub(r"(?<!\\)\\(?![\"\\\/bfnrt]|u[0-9a-fA-F]{4})", r"\\\\", cleaned_result)
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                logger.error("Failed to fix and parse JSON")
                try:
                    return ast.literal_eval(cleaned_result)
                except (ValueError, SyntaxError) as ast_err:
                    logger.error(f"AST parsing error: {str(ast_err)}")
                    return None

    def _process_parsed_content(self, parsed_content: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = [
            'brief_summary', 'match_score', 'recommendation_for_interview',
            'experience_and_project_relevance', 'skills_gap', 'recruiter_questions'
        ]
        for field in required_fields:
            if field not in parsed_content or not parsed_content[field]:
                parsed_content[field] = self._generate_fallback_content(field, parsed_content)
                logger.warning(f"Field '{field}' was missing or empty, generated fallback content")

        try:
            parsed_content['match_score'] = int(parsed_content['match_score'])
        except (ValueError, TypeError):
            logger.error(f"Invalid match_score value: {parsed_content.get('match_score')}")
            parsed_content['match_score'] = 0

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
        return parsed_content

    def _generate_fallback_content(self, field: str, parsed_content: Dict[str, Any]) -> Any:
        if field == 'brief_summary':
            return "Unable to generate a brief summary due to an error in the analysis."
        elif field == 'match_score':
            return 0
        elif field == 'recommendation_for_interview':
            return "Unable to provide a recommendation due to an error in the analysis."
        elif field == 'experience_and_project_relevance':
            return "Unable to assess experience and project relevance due to an error in the analysis."
        elif field == 'skills_gap':
            return ["Unable to determine skills gap due to an error in the analysis."]
        elif field == 'recruiter_questions':
            return ["Unable to generate recruiter questions due to an error in the analysis."]
        else:
            return f"No {field.replace('_', ' ')} available due to an error in the analysis."

    def analyze_match(self, resume: str, job_description: str, candidate_data: Dict[str, Any], job_title: str) -> Dict[str, Any]:
        try:
            logger.info("Starting resume analysis")
            prompt = f"""
            Analyze the fit between the following resume and job description with extreme accuracy. 
            Focus specifically on how well the candidate's skills and experience match the job requirements for the role of {job_title}.
    
            Provide a detailed analysis for each of the following areas:

            1. Brief Summary: Provide a concise overview of the candidate's fit for the role of {job_title} in 2-3 sentences. This is mandatory and must always be included.
            2. Match Score: Provide a percentage between 0 and 100, where 0% means the candidate has none of the required skills or experience, and 100% means the candidate perfectly matches all job requirements for {job_title}. Be very critical and realistic in this scoring.
            3. Recommendation for Interview: Based on the match score and the candidate's fit for {job_title}, provide a recommendation (e.g., "Highly recommend", "Recommend", "Recommend with reservations", "Do not recommend").
            4. Experience and Project Relevance: Provide a comprehensive analysis of the candidate's work experience and relevant projects, specifically relating them to the job requirements for {job_title}. This section is crucial and must always contain detailed information.
            5. Skills Gap: List all important skills or qualifications mentioned in the job description for {job_title} that the candidate lacks. Be exhaustive in this analysis.
            6. Recruiter Questions: Suggest 3-5 specific questions for the recruiter to ask the candidate based on their resume and the job requirements for {job_title}. These questions are mandatory and must always be included.

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

            Job Title: {job_title}

            Candidate Data:
            {json.dumps(candidate_data)}

            Job Description:
            {job_description}

            Resume:
            {resume}

            JSON Response:
            """
            result = self.analyze(prompt)
        
            logger.debug(f"Arguments for analyze_match: resume={resume[:20]}..., job_description={job_description[:20]}..., candidate_data={candidate_data}, job_title={job_title}")
            logger.debug(f"Number of arguments: {len([resume, job_description, candidate_data, job_title])}")
            logger.debug(f"Analysis result: {result}")

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
            "experience_and_project_relevance": "Unable to assess experience and project relevance due to an error.",
            "skills_gap": ["Unable to determine skills gap due to an error."],
            "recruiter_questions": ["Unable to generate recruiter questions due to an error."]
        }

class ClaudeClient:
    def __init__(self, api_key: str, api_url: str = "https://api.anthropic.com/v1/messages"):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

    def create_completion(self, messages: list, model: str = "claude-3-5-sonnet-20240620", max_tokens: int = 2000):
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens
        }
        response = requests.post(self.api_url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()

class ClaudeAPI:
    def __init__(self, api_key: str):
        self.client = ClaudeClient(api_key)
        logger.info("ClaudeAPI initialized")

    def analyze(self, prompt: str) -> Dict[str, Any]:
        try:
            logger.debug("Sending request to Claude API")
            messages = [{"role": "user", "content": prompt}]
            completion = self.client.create_completion(messages)
            logger.debug(f"Received response from Claude: {completion}")

            result = completion['content'][0]['text']
            logger.debug(f"Raw response from Claude: {result}")

            # Clean and parse JSON
            cleaned_result = self._clean_json_string(result)
            parsed_content = self._parse_json(cleaned_result)

            if parsed_content is None:
                raise ValueError("Failed to parse JSON response")

            return self._process_parsed_content(parsed_content)

        except Exception as e:
            logger.error(f"Claude API request failed: {str(e)}")
            return self._generate_error_response(str(e))

    def _clean_json_string(self, s: str) -> str:
        # Remove any text before the first '{' and after the last '}'
        start = s.find('{')
        end = s.rfind('}')
        if start != -1 and end != -1:
            s = s[start:end+1]
        # Remove any non-ASCII characters
        s = ''.join(char for char in s if ord(char) < 128)
        # Remove any newlines or extra whitespace
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    def _parse_json(self, cleaned_result: str) -> Dict[str, Any]:
        logger.debug(f"Attempting to parse JSON: {cleaned_result}")
        try:
            return json.loads(cleaned_result)
        except json.JSONDecodeError as json_err:
            logger.error(f"JSON parsing error: {str(json_err)}")
            try:
                # Try to fix common JSON errors
                fixed_json = re.sub(r"(?<!\\)\\(?![\"\\\/bfnrt]|u[0-9a-fA-F]{4})", r"\\\\", cleaned_result)
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                logger.error("Failed to fix and parse JSON")
                try:
                    return ast.literal_eval(cleaned_result)
                except (ValueError, SyntaxError) as ast_err:
                    logger.error(f"AST parsing error: {str(ast_err)}")
                    return None

    def _process_parsed_content(self, parsed_content: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = [
            'brief_summary', 'match_score', 'recommendation_for_interview',
            'experience_and_project_relevance', 'skills_gap', 'recruiter_questions'
        ]
        for field in required_fields:
            if field not in parsed_content or not parsed_content[field]:
                parsed_content[field] = self._generate_fallback_content(field, parsed_content)
                logger.warning(f"Field '{field}' was missing or empty, generated fallback content")

        try:
            parsed_content['match_score'] = int(parsed_content['match_score'])
        except (ValueError, TypeError):
            logger.error(f"Invalid match_score value: {parsed_content.get('match_score')}")
            parsed_content['match_score'] = 0

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
        return parsed_content

    def _generate_fallback_content(self, field: str, parsed_content: Dict[str, Any]) -> Any:
        if field == 'brief_summary':
            return "Unable to generate a brief summary due to an error in the analysis."
        elif field == 'match_score':
            return 0
        elif field == 'recommendation_for_interview':
            return "Unable to provide a recommendation due to an error in the analysis."
        elif field == 'experience_and_project_relevance':
            return "Unable to assess experience and project relevance due to an error in the analysis."
        elif field == 'skills_gap':
            return ["Unable to determine skills gap due to an error in the analysis."]
        elif field == 'recruiter_questions':
            return ["Unable to generate recruiter questions due to an error in the analysis."]
        else:
            return f"No {field.replace('_', ' ')} available due to an error in the analysis."

    def analyze_match(self, resume: str, job_description: str, candidate_data: Dict[str, Any], job_title: str) -> Dict[str, Any]:
        try:
            logger.info("Starting resume analysis")
            prompt = f"""
            Analyze the fit between the following resume and job description with extreme accuracy. 
            Focus specifically on how well the candidate's skills and experience match the job requirements for the role of {job_title}.
    
            Provide a detailed analysis for each of the following areas:

            1. Brief Summary: Provide a concise overview of the candidate's fit for the role of {job_title} in 2-3 sentences. This is mandatory and must always be included.
            2. Match Score: Provide a percentage between 0 and 100, where 0% means the candidate has none of the required skills or experience, and 100% means the candidate perfectly matches all job requirements for {job_title}. Be very critical and realistic in this scoring.
            3. Recommendation for Interview: Based on the match score and the candidate's fit for {job_title}, provide a recommendation (e.g., "Highly recommend", "Recommend", "Recommend with reservations", "Do not recommend").
            4. Experience and Project Relevance: Provide a comprehensive analysis of the candidate's work experience and relevant projects, specifically relating them to the job requirements for {job_title}. This section is crucial and must always contain detailed information.
            5. Skills Gap: List all important skills or qualifications mentioned in the job description for {job_title} that the candidate lacks. Be exhaustive in this analysis.
            6. Recruiter Questions: Suggest 3-5 specific questions for the recruiter to ask the candidate based on their resume and the job requirements for {job_title}. These questions are mandatory and must always be included.

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

            Job Title: {job_title}

            Candidate Data:
            {json.dumps(candidate_data)}

            Job Description:
            {job_description}

            Resume:
            {resume}

            JSON Response:
            """
            result = self.analyze(prompt)
        
            logger.debug(f"Arguments for analyze_match: resume={resume[:20]}..., job_description={job_description[:20]}..., candidate_data={candidate_data}, job_title={job_title}")
            logger.debug(f"Number of arguments: {len([resume, job_description, candidate_data, job_title])}")
            logger.debug(f"Analysis result: {result}")

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
            "experience_and_project_relevance": "Unable to assess experience and project relevance due to an error.",
            "skills_gap": ["Unable to determine skills gap due to an error."],
            "recruiter_questions": ["Unable to generate recruiter questions due to an error."]
        }