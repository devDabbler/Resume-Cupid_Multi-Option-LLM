import logging
from typing import Dict, Any
from llama_service import LlamaService
import concurrent.futures
import time
from score_calculator import score_calculator

logger = logging.getLogger(__name__)

class LLMOrchestrator:
    def __init__(self):
        self.services = {
            'llama': LlamaService(),
        }

    def analyze_resume(self, resume_text: str, job_description: str, job_title: str) -> Dict[str, Any]:
        results = {}
        for name, service in self.services.items():
            try:
                logger.info(f"Starting analysis with {name} service")
                results[name] = service.analyze_resume(resume_text, job_description, job_title)
                logger.info(f"Completed analysis with {name} service")
            except Exception as e:
                logger.error(f"Error in {name} service: {str(e)}")
                results[name] = {"error": str(e)}
        
        return self._aggregate_results(results, job_title)

    def _analyze_with_service(self, name: str, service: Any, resume_text: str, job_description: str, job_title: str) -> Dict[str, Any]:
        logger.info(f"Starting analysis with {name} service")
        result = service.analyze_resume(resume_text, job_description, job_title)
        logger.info(f"Completed analysis with {name} service")
        return result

    def _aggregate_results(self, results: Dict[str, Dict[str, Any]], job_title: str) -> Dict[str, Any]:
        aggregated_result = {
            "match_score": 0,
            "summary": "",
            "key_strengths": [],
            "areas_for_improvement": [],
            "skills_gap": [],
            "experience_relevance": {},
            "recruiter_questions": []
        }

        if 'llama' in results and 'error' not in results['llama']:
            llama_result = results['llama']
            aggregated_result["match_score"] = score_calculator.calculate_score(llama_result)  # Updated line
            aggregated_result["summary"] = llama_result.get("Brief Summary", "No summary available")
            aggregated_result["key_strengths"] = llama_result.get("Key Strengths", [])
            aggregated_result["areas_for_improvement"] = llama_result.get("Areas for Improvement", [])
            aggregated_result["skills_gap"] = llama_result.get("Missing Critical Skills", [])
            aggregated_result["experience_relevance"] = llama_result.get("Experience and Project Relevance", {})
            aggregated_result["recruiter_questions"] = llama_result.get("Recruiter Questions", [])
            aggregated_result["recommendation"] = self._generate_recommendation(aggregated_result["match_score"])
            aggregated_result["fit_summary"] = self._generate_fit_summary(aggregated_result["match_score"], job_title)

        logger.info(f"Aggregated result: {aggregated_result}")
        return aggregated_result

    def _generate_recommendation(self, match_score: int) -> str:
        if match_score >= 90:
            return "Strongly recommend for immediate interview"
        elif 80 <= match_score < 90:
            return "Highly recommend for interview"
        elif 70 <= match_score < 80:
            return "Recommend for interview"
        elif 60 <= match_score < 70:
            return "Consider for interview with reservations"
        elif 50 <= match_score < 60:
            return "Potentially consider for interview, but significant gaps exist"
        else:
            return "Do not recommend for interview at this time"

    def _generate_fit_summary(self, match_score: int, job_title: str) -> str:
        if match_score >= 90:
            return f"The candidate is an exceptional fit for the {job_title} role, exceeding most job requirements and demonstrating outstanding qualifications."
        elif 80 <= match_score < 90:
            return f"The candidate is an excellent fit for the {job_title} role, meeting or exceeding most job requirements with minor areas for improvement."
        elif 70 <= match_score < 80:
            return f"The candidate is a good fit for the {job_title} role, meeting many of the job requirements with some areas for development."
        elif 60 <= match_score < 70:
            return f"The candidate shows potential for the {job_title} role but has notable gaps that would require further assessment and development."
        elif 50 <= match_score < 60:
            return f"The candidate has some relevant skills for the {job_title} role, but significant gaps exist that may hinder their immediate success."
        else:
            return f"The candidate is not a strong fit for the {job_title} role, with considerable gaps in required skills and experience."

llm_orchestrator = LLMOrchestrator()