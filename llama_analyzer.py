import groq
import os
import json
import logging
from typing import Dict, Any

class LlamaAPI:
    def __init__(self, api_key: str):
        self.client = groq.Groq(api_key=api_key)

    def analyze(self, prompt: str) -> Dict[str, Any]:
        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.1-8b-instant",
                max_tokens=1024,
            )

            result = completion.choices[0].message.content
            logging.debug(f"Received response from Llama: {result}")
            return self._parse_json_response(result)

        except Exception as e:
            logging.error(f"Llama API request failed: {str(e)}")
            return self._generate_error_response(str(e))

    def analyze_match(self, resume: str, job_description: str) -> Dict[str, Any]:
        prompt = f"""
        Analyze the fit between the following resume and job description with extreme accuracy. 
        Focus specifically on how well the candidate's skills and experience match the job requirements.
    
        Provide a match score as a percentage between 0 and 100, where:
        0% means the candidate has none of the required skills or experience
        100% means the candidate perfectly matches all job requirements
    
        Be very critical and realistic in this scoring. A typical software developer without ML experience should score very low for an ML job.
    
        Based on the match score, provide one of these recommendations:
        - If match_score < 50%: "Do not recommend for interview"
        - If 50% <= match_score < 65%: "Recommend for interview with significant reservations (soft match)"
        - If 65% <= match_score < 80%: "Recommend for interview with reservations"
        - If 80% <= match_score < 90%: "Recommend for interview"
        - If match_score >= 90%: "Highly recommend for interview"
    
        Format your response as JSON with the following structure:
        {{
            "summary": "A brief 1-2 sentence summary of the candidate's fit for the role",
            "match_score": The percentage match (0-100),
            "recommendation": "The interview recommendation based on the criteria above",
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
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                parsed_json = json.loads(json_str)
        
                # Ensure all expected keys exist
                keys = [
                    'summary', 'match_score', 'recommendation', 'analysis', 
                    'strengths', 'areas_for_improvement', 'skills_gap', 
                    'interview_questions', 'project_relevance'
                ]
                for key in keys:
                    if key not in parsed_json:
                        parsed_json[key] = None  # Assign None or appropriate default
        
                return parsed_json
            else:
                raise ValueError("No JSON object found in the response")
        except json.JSONDecodeError as json_error:
            logging.error(f"Error parsing Llama API response: {str(json_error)}")
            logging.error(f"Raw response: {response}")
            return self._generate_error_response(f"Error parsing API response: {str(json_error)}")
        except Exception as e:
            logging.error(f"Unexpected error in parsing Llama API response: {str(e)}")
            logging.error(f"Raw response: {response}")
        return self._generate_error_response(f"Unexpected error in parsing API response: {str(e)}")

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