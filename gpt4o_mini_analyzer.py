from openai import OpenAI
import logging
import json
import re
from typing import Dict, Any
import os

class GPT4oMiniAPI:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        logging.debug(f"Initialized GPT4oMiniAPI with API key: {api_key[:5]}...{api_key[-5:]}")

    def analyze(self, prompt: str) -> Dict[str, Any]:
        try:
            logging.debug("Sending request to GPT-4o-mini API")
            response = self.client.chat.completions.create(
                model="gpt-4",  # Change this to the correct GPT-4o model when available
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes resumes and job descriptions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                n=1,
                temperature=0.7,
            )
            
            logging.debug(f"GPT-4o-mini API response: {response}")
            
            parsed_response = self._parse_json_response(response.choices[0].message.content)
            
            # Add recommendation based on match_score
            match_score = parsed_response.get('match_score', 0)
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
            
            parsed_response['recommendation'] = recommendation
            return parsed_response

        except Exception as e:
            logging.error(f"OpenAI API request failed: {str(e)}")
            return self._generate_error_response(str(e))

    def analyze_match(self, resume: str, job_description: str) -> Dict[str, Any]:
        prompt = f"""
        Analyze the fit between the following resume and job description with extreme accuracy. 
        Focus specifically on how well the candidate's skills and experience match the job requirements.
        
        Provide a match score as a percentage between 0 and 100, where:
        0% means the candidate has none of the required skills or experience
        100% means the candidate perfectly matches all job requirements
        
        Be very critical and realistic in this scoring. A typical software developer without ML experience should score very low for an ML job.
        
        Format your response as JSON with the following structure:
        {{
            "summary": "A brief 1-2 sentence summary of the candidate's fit for the role",
            "match_score": The percentage match (0-100),
            "analysis": "Detailed analysis of the fit, focusing on job-specific requirements",
            "strengths": ["Strength 1", "Strength 2", ...],
            "areas_for_improvement": ["Area 1", "Area 2", ...],
            "skills_gap": ["Missing Skill 1", "Missing Skill 2", ...],
            "interview_questions": ["Question 1", "Question 2", ...],
            "project_relevance": "Analysis of how personal projects relate to the specific job requirements"
        }}

        Job Description:
        {job_description}

        Resume:
        {resume}

        JSON Response:
        """

        return self.analyze(prompt)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            # First, try to parse the entire response as JSON
            try:
                parsed_json = json.loads(response)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the response
                json_match = re.search(r'\{(?:[^{}]|(?R))*\}', response)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_json = json.loads(json_str)
                else:
                    raise ValueError("No valid JSON found in the response")

            # Ensure all expected keys exist
            keys = [
                'summary', 'match_score', 'analysis', 'strengths', 
                'areas_for_improvement', 'skills_gap', 
                'interview_questions', 'project_relevance'
            ]
            for key in keys:
                if key not in parsed_json:
                    parsed_json[key] = None  # Assign None or appropriate default

            # Ensure match_score is an integer
            if 'match_score' in parsed_json:
                try:
                    parsed_json['match_score'] = int(parsed_json['match_score'])
                except (ValueError, TypeError):
                    parsed_json['match_score'] = 0

            return parsed_json

        except Exception as e:
            logging.error(f"Error parsing GPT-4o-mini API response: {str(e)}")
            logging.error(f"Raw response: {response}")
            return self._generate_error_response(f"Error parsing API response: {str(e)}")

    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        return {
            "error": error_message,
            "summary": "Unable to complete analysis",
            "match_score": 0,
            "recommendation": "Unable to provide a recommendation due to an error.",
            "analysis": "Unable to complete analysis",
            "strengths": [],
            "areas_for_improvement": [],
            "skills_gap": [],
            "interview_questions": [],
            "project_relevance": ""
        }