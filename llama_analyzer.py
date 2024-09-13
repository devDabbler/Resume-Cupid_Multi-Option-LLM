import groq
from groq import Groq
import requests
import json
import re
import ast
from typing import Dict, Any, List
from logger import get_logger

# Initialize logging
logger = get_logger(__name__)

class LlamaAPI:
    def __init__(self, api_key: str):
        self.client = groq.Groq(api_key=api_key)
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
            logger.debug(f"Raw response from Llama: {result}")  # Log the entire response for debugging

            try:
                # Try to parse the JSON response
                parsed_content = json.loads(result)
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON parsing error: {str(json_error)}")
                logger.debug(f"Problematic JSON: {result}")
                
                # Attempt to fix common JSON errors
                try:
                    # Remove any text before the first '{' and after the last '}'
                    cleaned_json = re.search(r'\{.*\}', result, re.DOTALL).group()
                    parsed_content = json.loads(cleaned_json)
                except (AttributeError, json.JSONDecodeError):
                    logger.error("Failed to parse JSON even after cleaning")
                    return self._generate_error_response("Invalid JSON response from API")

            # Process the parsed content
            processed_content = self._process_parsed_content(parsed_content)

            logger.info(f"Analysis completed successfully for prompt: {prompt[:50]}...")
            return processed_content

        except Exception as e:
            logger.error(f"Llama API request failed: {str(e)}")
            return self._generate_error_response(str(e))

    def analyze_match(self, resume: str, job_description: str, candidate_data: Dict[str, Any], job_title: str) -> Dict[str, Any]:
        try:
            logger.info("Starting resume analysis")
            prompt = f"""
            You are a highly skilled AI recruiter specializing in Machine Learning Operations Engineering roles. Your task is to analyze the fit between the following resume and job description with extreme accuracy and specificity. 
            Focus on how well the candidate's skills and experience match the job requirements for the role of {job_title}, particularly in areas of model governance, risk management, and compliance in financial services.

            Provide a detailed analysis for each of the following areas:

            1. Brief Summary: Provide a concise overview of the candidate's fit for the role of {job_title} in 2-3 sentences.
            2. Match Score: Provide a percentage between 0 and 100, where:
               - 0-20%: The candidate has almost none of the required skills or experience in model governance, risk management, or compliance
               - 21-40%: The candidate has some relevant skills but lacks significant experience in model governance or financial services
               - 41-60%: The candidate has several relevant skills and some experience, but significant gaps remain in model governance or risk management
               - 61-80%: The candidate is a good fit with minor gaps in specific areas of model governance or compliance
               - 81-100%: The candidate is an excellent fit with strong experience in model governance, risk management, and compliance in financial services
               Be very critical and realistic in this scoring. Do not hesitate to give low scores for candidates who lack specific experience in model governance and risk management in financial services.
            3. Recommendation for Interview: Based on the match score and the candidate's fit for {job_title}, provide a recommendation.
            4. Experience and Project Relevance: Provide a comprehensive analysis of the candidate's work experience and relevant projects, specifically relating them to model governance, risk management, and compliance in financial services.
            5. Skills Gap: List all important skills or qualifications mentioned in the job description for {job_title} that the candidate lacks, particularly focusing on model governance, risk management, and compliance skills.
            6. Recruiter Questions: Suggest 3-5 specific questions for the recruiter to ask the candidate based on their resume and the job requirements for {job_title}, focusing on model governance and risk management experience.

            Ensure that all fields are populated with relevant, detailed information. Do not include any text outside of the JSON structure.

            Job Title: {job_title}

            Candidate Data:
            {json.dumps(candidate_data)}

            Job Description:
            {job_description}

            Resume:
            {resume}

            JSON Response:
            """
            logger.debug(f"Generated prompt: {prompt[:500]}...")  # Log first 500 characters of the prompt
            result = self.analyze(prompt)
        
            logger.debug(f"Analysis result: {result}")
        
            return result

        except Exception as e:
            logger.error(f"Error in analyze_match: {str(e)}", exc_info=True)
            return self._generate_error_response(f"Error in analyze_match: {str(e)}")

    def _process_parsed_content(self, parsed_content: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = [
            'Brief Summary', 'Match Score', 'Recommendation for Interview',
            'Experience and Project Relevance', 'Skills Gap', 'Recruiter Questions'
        ]
        for field in required_fields:
            if field not in parsed_content or not parsed_content[field]:
                parsed_content[field.lower()] = self._generate_fallback_content(field.lower(), parsed_content)
                logger.warning(f"Field '{field}' was missing or empty in API response, generated fallback content")
            else:
                # Clean the content
                parsed_content[field.lower()] = self._clean_content(parsed_content[field])
    
        try:
            parsed_content['match_score'] = int(float(parsed_content.get('Match Score', 0)))
        except (ValueError, TypeError):
            logger.error(f"Invalid match_score value: {parsed_content.get('Match Score')}")
            parsed_content['match_score'] = 0

        match_score = parsed_content['match_score']
        if match_score == 0:
            recommendation = "Do not recommend for interview (not suitable for the role)"
        elif match_score < 30:
            recommendation = "Do not recommend for interview (significant skill gaps)"
        elif 30 <= match_score < 50:
            recommendation = "Recommend for interview with significant reservations"
        elif 50 <= match_score < 70:
            recommendation = "Recommend for interview with minor reservations"
        elif 70 <= match_score < 85:
            recommendation = "Recommend for interview"
        else:
            recommendation = "Highly recommend for interview"

        parsed_content['recommendation'] = recommendation
        return parsed_content

    def _clean_content(self, content):
        if isinstance(content, str):
            # Remove any ANSI escape sequences (terminal color codes)
            content = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', content)
            # Remove any other unwanted characters or formatting
            content = re.sub(r'[^\w\s.,!?:;()\[\]{}\-]', '', content)
        elif isinstance(content, dict):
            return {k: self._clean_content(v) for k, v in content.items()}
        elif isinstance(content, list):
            return [self._clean_content(item) for item in content]
        return content

    def _generate_fallback_content(self, field: str, parsed_content: Dict[str, Any]) -> Any:
        if field == 'brief_summary':
            return "Unable to generate a brief summary due to incomplete analysis. A manual review is recommended."
        elif field == 'match_score':
            return 0
        elif field == 'recommendation_for_interview':
            return "Unable to provide a recommendation due to incomplete analysis."
        elif field == 'experience_and_project_relevance':
            return "Unable to assess experience and project relevance. A thorough manual review of the resume is necessary."
        elif field == 'skills_gap':
            return ["Unable to determine specific skills gap. A manual comparison against job requirements is needed."]
        elif field == 'recruiter_questions':
            return [
                "Can you elaborate on your experience with model governance in financial services?",
                "What specific risk management frameworks have you implemented in your previous roles?",
                "How do you ensure compliance with regulatory requirements in ML model development and deployment?",
                "Can you provide examples of projects where you've improved model monitoring and documentation processes?"
            ]
        else:
            return f"No {field.replace('_', ' ')} available due to incomplete analysis."

    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        logger.warning(f"Generating error response: {error_message}")
        return {
            "error": error_message,
            "brief_summary": "Unable to complete analysis due to an error. Manual review required.",
            "match_score": 0,
            "recommendation_for_interview": "Unable to provide a recommendation due to an error. Please review the resume manually.",
            "experience_and_project_relevance": "Error occurred during analysis. Unable to assess experience and project relevance.",
            "skills_gap": ["Error in analysis. Manual skill gap assessment needed."],
            "recruiter_questions": [
                "Due to an error in our system, we couldn't generate specific questions. Please review the resume and job description to formulate relevant questions."
            ]
        }