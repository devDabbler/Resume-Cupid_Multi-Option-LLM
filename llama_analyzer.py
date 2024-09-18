import requests
import logging
import json
import re
from typing import Dict, Any, List
from logger import get_logger
from groq import Groq
import os
from utils import extract_text_from_file, generate_brief_summary, generate_fit_summary, format_nested_structure, get_recommendation, generate_generic_questions, format_recruiter_questions

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

    def generate_recruiter_questions_with_llm(self, resume_text, job_description):
        prompt = f"""
        You are an expert recruiter. Based on the following resume and job description, generate four insightful and role-specific interview questions.
        
        Resume:
        {resume_text}

        Job Description:
        {job_description}

        Ensure the questions explore the candidate's fit for the role, technical skills, soft skills, and any gaps in experience.
        """
        
        recruiter_questions = self.analyze(prompt)
        return recruiter_questions.get("choices", ["No questions generated"])

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
                temperature=0.5,
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
            logger.error(f"Error during analysis: {str(e)}")
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
            match_score = processed_content.get('match_score')
            if isinstance(match_score, dict):
                total_score = sum(match_score.values())
                total_weight = len(match_score)
                processed_content['match_score'] = int(total_score / total_weight)
            elif isinstance(match_score, str):
                match = re.search(r'\d+', match_score)
                if match:
                    processed_content['match_score'] = int(match.group())
                else:
                    processed_content['match_score'] = 0
            elif match_score is None or match_score == 0:
                exp_proj_relevance = processed_content.get('experience_and_project_relevance', {})
                if isinstance(exp_proj_relevance, dict):
                    relevance = exp_proj_relevance.get('overall_relevance', 0)
                else:
                    match = re.search(r'\d+', str(exp_proj_relevance))
                    relevance = int(match.group()) if match else 0

                skills_gap = processed_content.get('skills_gap', {})
                if isinstance(skills_gap, dict):
                    gap = skills_gap.get('overall_gap', 100)
                else:
                    match = re.search(r'\d+', str(skills_gap))
                    gap = int(match.group()) if match else 100

                processed_content['match_score'] = int((relevance * 0.7 + (100 - gap) * 0.3))

            processed_content['match_score'] = int(float(processed_content['match_score']))
            processed_content['match_score'] = max(0, min(100, processed_content['match_score']))
            logger.debug(f"Processed match_score: {processed_content['match_score']}")

        except (ValueError, TypeError) as e:
            logger.error(f"Invalid match_score value: {processed_content.get('match_score')}")
            processed_content['match_score'] = 0

        processed_content['recommendation'] = self._get_recommendation(processed_content['match_score'])
        logger.debug(f"Generated recommendation: {processed_content['recommendation']}")

        match_score = processed_content['match_score']
        if match_score < 50:
            processed_content['brief_summary'] = (
            f"The candidate is not a strong fit for the Data Scientist role at Fractal Analytics. "
            f"With a match score of {match_score}%, there are significant gaps in required skills and experience for this position."
            )
        elif 50 <= match_score < 65:
            processed_content['brief_summary'] = (
                f"The candidate shows some potential for the Data Scientist role at Fractal Analytics, "
                f"but with a match score of {match_score}%, there are considerable gaps in meeting the requirements. "
                f"Further evaluation is needed."
            )
        else:
            processed_content['brief_summary'] = (
                f"The candidate is a good fit for the Data Scientist role at Fractal Analytics. "
                f"With a match score of {match_score}%, they demonstrate strong alignment with the required skills and experience for this position."
            )

        logger.debug(f"Final processed content: {processed_content}")
        return processed_content

    def _generate_fallback_content(self, field: str) -> Any:
        fallback_content = {
            'brief_summary': "No summary available",
            'match_score': 0,
            'recommendation_for_interview': "Unable to provide a recommendation",
            'experience_and_project_relevance': "No relevance information available",
            'skills_gap': "Unable to determine skills gap",
            'recruiter_questions': ["No recruiter questions generated"]
        }
        return fallback_content.get(field, f"No {field.replace('_', ' ')} available")

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
    
        if not resume or not job_description:
            logger.error("Empty resume or job description provided")
            return self._generate_error_response("Empty input provided")
    
        try:
            prompt = f"""
            Analyze the following resume against the provided job description for a Data Scientist role. Provide a detailed evaluation covering:

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
            
            logger.debug(f"Prompt length: {len(prompt)}")
        
            result = self.analyze(prompt)
            result['file_name'] = candidate_data.get('file_name', 'Unknown')
            logger.debug(f"Analysis result for {result['file_name']}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error during analysis for {candidate_data.get('file_name', 'Unknown')}: {str(e)}", exc_info=True)
            return self._generate_error_response(str(e))

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
    llama_api_key = os.getenv("LLAMA_API_KEY")
    if not llama_api_key:
        raise ValueError("LLAMA_API_KEY is not set in the environment variables.")
    return LlamaAPI(api_key=llama_api_key)

def generate_error_result(file_name: str, error_message: str) -> Dict[str, Any]:
    return {
        'file_name': file_name,
        'brief_summary': f"Error occurred during analysis: {error_message}",
        'fit_summary': "Unable to generate fit summary due to an error",
        'match_score': 0,
        'experience_and_project_relevance': "Unable to assess due to an error",
        'skills_gap': "Unable to determine skills gap due to an error",
        'key_strengths': [],
        'areas_for_improvement': [],
        'recruiter_questions': ["Unable to generate recruiter questions due to an error"],
        'recommendation': "Unable to provide a recommendation due to an error"
    }

def get_strengths_and_improvements(resume_text: str, job_description: str, llm: LlamaAPI) -> Dict[str, List[Dict[str, Any]]]:
    prompt = f"""
    Analyze the following resume and job description, then provide a structured summary of the candidate's key strengths and areas for improvement. Focus on the most impactful points relevant to the job.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Provide a JSON object with the following structure:
    {{
        "strengths": [
            {{ "category": "Technical Skills", "points": ["...", "..."] }},
            {{ "category": "Experience", "points": ["...", "..."] }},
            {{ "category": "Soft Skills", "points": ["...", "..."] }}
        ],
        "improvements": [
            {{ "category": "Skills Gap", "points": ["...", "..."] }},
            {{ "category": "Experience", "points": ["...", "..."] }},
            {{ "category": "Industry Knowledge", "points": ["...", "..."] }}
        ]
    }}

    Each category should have 2-3 concise points (1-2 sentences each).
    """
    
    try:
        response = llm.analyze(prompt)
        strengths_and_improvements = json.loads(response.get('brief_summary', '{}'))
    except:
        # Fallback data if analysis fails
        strengths_and_improvements = {
            'strengths': [
                {'category': 'Technical Skills', 'points': ['Candidate possesses relevant technical skills for the role']},
                {'category': 'Experience', 'points': ['Candidate has experience in related fields']},
                {'category': 'Soft Skills', 'points': ['Candidate likely has essential soft skills for the position']}
            ],
            'improvements': [
                {'category': 'Skills Gap', 'points': ['Consider assessing any potential skill gaps during the interview']},
                {'category': 'Experience', 'points': ['Explore depth of experience in specific areas during the interview']},
                {'category': 'Industry Knowledge', 'points': ['Evaluate industry-specific knowledge in the interview process']}
            ]
        }
    
    return strengths_and_improvements

# You may need to import these functions from utils.py or define them here if they're not already imported
def generate_brief_summary(score: int, job_title: str) -> str:
    if score < 30:
        return f"The candidate is not a strong fit for the {job_title} role. With a match score of {score}%, there are significant gaps in required skills and experience for this position."
    elif 30 <= score < 50:
        return f"The candidate shows limited potential for the {job_title} role. With a match score of {score}%, there are considerable gaps in meeting the requirements. Further evaluation is needed."
    elif 50 <= score < 65:
        return f"The candidate shows some potential for the {job_title} role, but with a match score of {score}%, there are gaps in meeting the requirements. Further evaluation is recommended."
    elif 65 <= score < 80:
        return f"The candidate is a good fit for the {job_title} role. With a match score of {score}%, they demonstrate alignment with many of the required skills and experience for this position."
    else:
        return f"The candidate is an excellent fit for the {job_title} role. With a match score of {score}%, they demonstrate strong alignment with the required skills and experience for this position."

def generate_fit_summary(result: Dict[str, Any]) -> str:
    score = result['match_score']
    if score < 50:
        return "The candidate is not a strong fit, with considerable gaps in required skills and experience."
    elif 50 <= score < 65:
        return "The candidate shows potential but has significant gaps that would require further assessment."
    elif 65 <= score < 80:
        return "The candidate is a good fit, meeting many of the job requirements with some minor gaps."
    else:
        return "The candidate is an excellent fit, meeting or exceeding most job requirements."

def format_nested_structure(data: Any) -> str:
    if isinstance(data, dict):
        return "\n".join([f"{k}: {format_nested_structure(v)}" for k, v in data.items()])
    elif isinstance(data, list):
        return "\n".join([f"- {format_nested_structure(item)}" for item in data])
    else:
        return str(data)

def format_recruiter_questions(questions: List[str]) -> List[str]:
    return [str(q) for q in questions[:5]]  # Limit to 5 questions and ensure they're strings

# Main execution
if __name__ == "__main__":
    llm = initialize_llm()
    # You can add any test or example usage here