import logging
from typing import Dict, Any
from llama_service import LlamaService
import concurrent.futures
import time
from score_calculator import score_calculator
from utils import generate_recommendation, generate_fit_summary

logger = logging.getLogger(__name__)

class LLMOrchestrator:
    def __init__(self):
        self.services = {
            'llama': LlamaService(),
        }

    def analyze_resume(self, resume_text: str, job_description: str, job_title: str) -> Dict[str, Any]:
        print(f"Debug: LLMOrchestrator received job title: '{job_title}'")
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
        print(f"Debug: _aggregate_results received job title: '{job_title}'")
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
            aggregated_result["match_score"] = score_calculator.calculate_score(llama_result, job_title)  # Updated line
            aggregated_result["summary"] = llama_result.get("Brief Summary", "No summary available")
            aggregated_result["key_strengths"] = llama_result.get("Key Strengths", [])
            aggregated_result["areas_for_improvement"] = llama_result.get("Areas for Improvement", [])
            aggregated_result["skills_gap"] = llama_result.get("Missing Critical Skills", [])
            aggregated_result["experience_relevance"] = llama_result.get("Experience and Project Relevance", {})
            aggregated_result["recruiter_questions"] = llama_result.get("Recruiter Questions", [])
            aggregated_result["recommendation"] = generate_recommendation(aggregated_result["match_score"])
            aggregated_result["fit_summary"] = generate_fit_summary(aggregated_result["match_score"], job_title)

        logger.info(f"Aggregated result: {aggregated_result}")
        return aggregated_result

llm_orchestrator = LLMOrchestrator()