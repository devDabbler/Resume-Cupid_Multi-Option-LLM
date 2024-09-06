import os
from openai import OpenAI
import logging
import json
from typing import Dict, Any

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPT4oMiniAPI:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = "gpt-4o-mini"
        logger.info(f"Initialized GPT4oMiniAPI with API key: {api_key[:5]}... and model: {self.model_name}")

    def analyze(self, prompt: str) -> Dict[str, Any]:
        try:
            logger.debug(f"Sending request to {self.model_name} API")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an AI assistant specialized in analyzing resumes and job descriptions. Always provide your responses in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,  # Increased to allow for more detailed responses
                n=1,
                temperature=0.7,
            )
            
            logger.debug(f"{self.model_name} API response: {response}")
            
            result = response.choices[0].message.content
            logger.debug(f"Raw response from {self.model_name}: {result[:100]}...")  # Log first 100 characters

            # Parse the JSON response
            parsed_content = json.loads(result)
            logger.debug(f"Successfully parsed content: {parsed_content}")

            # Validate parsed content structure
            self._validate_parsed_content(parsed_content)

            return parsed_content

        except Exception as e:
            logger.error(f"{self.model_name} API request failed: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))

    def _validate_parsed_content(self, parsed_content: Dict[str, Any]):
        required_fields = ['brief_summary', 'match_score', 'recommendation_for_interview', 
                           'experience_and_project_relevance', 'skills_gap', 'recruiter_questions']
        for field in required_fields:
            if field not in parsed_content:
                logger.warning(f"Required field '{field}' missing from parsed content. Using fallback content.")
                parsed_content[field] = self._generate_fallback_content(field, parsed_content)

        # Ensure match_score is an integer
        try:
            parsed_content['match_score'] = int(parsed_content['match_score'])
        except (ValueError, TypeError):
            logger.error(f"Invalid match_score value: {parsed_content['match_score']}")
            parsed_content['match_score'] = 0

    def analyze_match(self, resume: str, job_description: str, candidate_data: Dict[str, Any], job_title: str) -> Dict[str, Any]:
        try:
            logger.info("Starting resume analysis")
            prompt = f"""
            Analyze the fit between the following resume and job description with extreme accuracy. 
            Focus specifically on how well the candidate's skills and experience match the job requirements for the role of {job_title}.

            Provide a detailed analysis for each of the following areas:

            1. Brief Summary: Provide a concise overview of the candidate's fit for the role of {job_title} in 2-3 sentences.
            2. Match Score: Provide a percentage between 0 and 100, where 0% means the candidate has none of the required skills or experience, and 100% means the candidate perfectly matches all job requirements for {job_title}. Be very critical and realistic in this scoring.
            3. Recommendation for Interview: Based on the match score and the candidate's fit for {job_title}, provide a recommendation (e.g., "Highly recommend", "Recommend", "Recommend with reservations", "Do not recommend").
            4. Experience and Project Relevance: Provide a comprehensive analysis of the candidate's work experience and relevant projects, specifically relating them to the job requirements for {job_title}.
            5. Skills Gap: List all important skills or qualifications mentioned in the job description for {job_title} that the candidate lacks.
            6. Recruiter Questions: Suggest 3-5 specific questions for the recruiter to ask the candidate based on their resume and the job requirements for {job_title}.

            Format your response as JSON with the following structure:
            {{
            "brief_summary": "Your brief summary here",
            "match_score": The percentage match (0-100),
            "recommendation_for_interview": "Your recommendation here",
            "experience_and_project_relevance": "Your detailed analysis here",
            "skills_gap": ["Skill 1", "Skill 2", ...],
            "recruiter_questions": ["Question 1?", "Question 2?", ...]
            }}

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
            result = self.analyze(prompt)
    
            logger.debug(f"Arguments for analyze_match: resume={resume[:20]}..., job_description={job_description[:20]}..., candidate_data={candidate_data}, job_title={job_title}")
            logger.debug(f"Number of arguments: {len([resume, job_description, candidate_data, job_title])}")

            return result

        except Exception as e:
            logger.error(f"Error in analyze_match: {str(e)}", exc_info=True)
            return self._generate_error_response(f"Error in analyze_match: {str(e)}")

    def _generate_fallback_content(self, field: str, parsed_content: Dict[str, Any]) -> Any:
        if field == 'brief_summary':
            return "Unable to generate a brief summary due to an error in the analysis."
        elif field == 'match_score':
            return 0
        elif field == 'recommendation_for_interview':
            return "Unable to provide a recommendation due to incomplete analysis."
        elif field == 'experience_and_project_relevance':
            return "Unable to provide a detailed analysis of experience and project relevance. A manual review is recommended."
        elif field == 'skills_gap':
            return ["Unable to determine specific skills gap. A manual comparison is recommended."]
        elif field == 'recruiter_questions':
            return ["What specific experience do you have that relates to this role?", 
                    "Can you describe any relevant projects you've worked on?", 
                    "How do you stay updated with the latest developments in your field?"]
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