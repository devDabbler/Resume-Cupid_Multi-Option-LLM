import requests
import logging
import json
from typing import Dict, Any

class ClaudeAPI:
    def __init__(self, api_key: str, api_url: str = "https://api.anthropic.com/v1/messages"):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

    def analyze(self, prompt: str) -> Dict[str, Any]:
        try:
            logging.debug("Sending request to Claude API")
            payload = {
                "model": "claude-3-5-sonnet-20240620",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()

            result = response.json()
            logging.debug(f"Received response from Claude: {result}")

            # Extract the content from the Claude API response
            if 'content' in result:
                content = result['content'][0]['text']
                # Parse the JSON content
                parsed_content = json.loads(content)
                
                # Add recommendation based on match_score
                match_score = parsed_content.get('match_score', 0)
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
            else:
                raise ValueError("No content found in Claude API response")
    
        except requests.exceptions.RequestException as e:
            logging.error(f"Claude API request failed: {str(e)}")
            return self._generate_error_response(str(e))
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error parsing Claude API response: {str(e)}")
            return self._generate_error_response(str(e))
        except Exception as e:
            logging.error(f"Unexpected error in Claude analysis: {str(e)}")
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