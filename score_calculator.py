from typing import Dict, Any, List, Set
import re
import spacy

class ScoreCalculator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.core_skills = {
            "machine learning": 10,
            "deep learning": 9,
            "statistical analysis": 9,
            "data visualization": 7,
            "python": 10,
            "mathematics": 10,
            "linear algebra": 9,
            "calculus": 9,
            "probability": 9,
            "statistics": 9,
            "data analytics": 8,
            "predictive modeling": 9,
            "nlp": 8,
            "time series analysis": 8,
            "feature engineering": 8,
            "data pipeline": 7,
            "mlops": 8,
        }
        self.preferred_skills = {
            "scalable ml": 9,
            "mapreduce": 8,
            "spark": 8,
            "generative ai": 9,
            "large language models": 9,
            "llm": 9,
            "embedding models": 8,
            "prompt engineering": 8,
            "fine-tuning": 8,
            "reinforcement learning": 8,
            "tensorflow": 7,
            "pytorch": 7,
            "langchain": 7,
        }
        self.programming_skills = {
            "python": 10,
            "r": 7,
            "sql": 8,
            "javascript": 7,
            "java": 6,
            "c++": 6,
        }
        self.tools_and_frameworks = {
            "tensorflow": 8,
            "pytorch": 8,
            "scikit-learn": 8,
            "pandas": 8,
            "numpy": 8,
            "langchain": 7,
            "hugging face": 7,
            "docker": 6,
            "kubernetes": 6,
            "aws": 7,
            "azure": 7,
            "gcp": 7,
        }

    def calculate_score(self, llm_result: Dict[str, Any], job_requirements: Dict[str, Any]) -> Dict[str, Any]:
        skills_score = self._calculate_skills_score(llm_result.get("Skills Assessment", {}), job_requirements)
        experience_score = self._calculate_experience_score(llm_result.get("Experience and Project Relevance", {}), job_requirements)
        education_score = self._calculate_education_score(llm_result.get("Education", ""), job_requirements)
        
        # Adjust weights to prioritize experience more heavily
        weighted_score = (skills_score * 0.3) + (experience_score * 0.6) + (education_score * 0.1)
        
        missing_skills_penalty = self._calculate_missing_skills_penalty(llm_result.get("Missing Critical Skills", []), job_requirements)
        
        # Reduce the impact of missing skills penalty
        final_score = max(0, weighted_score - missing_skills_penalty * 0.2)
        
        return {
            'match_score': min(100, int(final_score)),
            'skills_score': skills_score,
            'experience_score': experience_score,
            'education_score': education_score,
            'missing_skills_penalty': missing_skills_penalty
        }

    def _calculate_skills_score(self, skills_assessment: Dict[str, List[str]], job_requirements: Dict[str, Any]) -> float:
        required_skills = set(job_requirements.get('required_skills', []))
        preferred_skills = set(job_requirements.get('preferred_skills', []))
        
        total_score = 0
        total_weight = 0
        
        for category, skills in skills_assessment.items():
            for skill in skills:
                skill_lower = skill.lower()
                if self._skill_match(skill_lower, required_skills):
                    total_score += self._get_skill_score(skill_lower) * 1.3
                    total_weight += 13
                elif self._skill_match(skill_lower, preferred_skills):
                    total_score += self._get_skill_score(skill_lower) * 1.2
                    total_weight += 12
                else:
                    total_score += self._get_skill_score(skill_lower)
                    total_weight += 10
        
        return (total_score / total_weight) * 100 if total_weight > 0 else 0

    def _get_skill_score(self, skill: str) -> int:
        if skill in self.core_skills:
            return self.core_skills[skill]
        elif skill in self.preferred_skills:
            return self.preferred_skills[skill]
        elif skill in self.programming_skills:
            return self.programming_skills[skill]
        elif skill in self.tools_and_frameworks:
            return self.tools_and_frameworks[skill]
        else:
            return 3  # Base score for unrecognized skills

    def _skill_match(self, skill: str, skill_set: Set[str]) -> bool:
        if skill in skill_set:
            return True
        return any(self.nlp(skill).similarity(self.nlp(s)) > 0.8 for s in skill_set)
    
    def _calculate_experience_score(self, experience_relevance: Dict[str, Any], job_requirements: Dict[str, Any]) -> float:
        required_years = job_requirements.get('years_of_experience', 0)
        total_score = 0
        total_weight = 0
        
        for project, relevance in experience_relevance.items():
            if isinstance(relevance, dict):
                for sub_project, sub_relevance in relevance.items():
                    score = self._extract_score(sub_relevance)
                    total_score += score
                    total_weight += 10
            else:
                score = self._extract_score(relevance)
                total_score += score
                total_weight += 10
        
        experience_score = (total_score / total_weight) * 100 if total_weight > 0 else 0
        
        years_of_experience = self._extract_years_of_experience(experience_relevance)
        years_score = min(years_of_experience / required_years, 2) * 100 if required_years > 0 else 100
        
        industry_match = self._calculate_industry_match(experience_relevance, job_requirements.get('industry_experience', []))
        
        return (experience_score * 0.4 + years_score * 0.4 + industry_match * 0.2)

    def _calculate_education_score(self, education: str, job_requirements: Dict[str, Any]) -> float:
        required_education = job_requirements.get('education_level', '').lower()
        candidate_education = education.lower()

        education_levels = ['high school', 'associate', 'bachelor', 'master', 'phd']
        required_index = next((i for i, level in enumerate(education_levels) if level in required_education), -1)
        candidate_index = next((i for i, level in enumerate(education_levels) if level in candidate_education), -1)

        if candidate_index >= required_index:
            return 100
        else:
            return (candidate_index / required_index) * 100 if required_index > 0 else 80

    def _extract_score(self, relevance: Any) -> int:
        if isinstance(relevance, dict):
            relevance = str(relevance)
        
        match = re.search(r'(\d+)(?:/10|\))', relevance)
        if match:
            return int(match.group(1))
        else:
            relevance_lower = relevance.lower()
            if 'highly relevant' in relevance_lower:
                return 9
            elif 'moderately relevant' in relevance_lower:
                return 7
            elif 'slightly relevant' in relevance_lower:
                return 5
            else:
                return 3

    def _apply_score_transformation(self, score: float) -> float:
        if score >= 85:
            return score * 1.06
        elif score >= 70:
            return score * 1.04
        elif score >= 55:
            return score * 0.98
        else:
            return score * 0.95

    def _calculate_missing_skills_penalty(self, missing_skills: List[str], job_requirements: Dict[str, Any]) -> float:
        required_skills = set(job_requirements.get('required_skills', []))
        preferred_skills = set(job_requirements.get('preferred_skills', []))
        
        penalty = 0
        for skill in missing_skills:
            skill_lower = skill.lower()
            if self._skill_match(skill_lower, required_skills):
                penalty += self._get_skill_score(skill_lower) * 0.2
            elif self._skill_match(skill_lower, preferred_skills):
                penalty += self._get_skill_score(skill_lower) * 0.1
        return penalty

    def _extract_years_of_experience(self, experience_relevance: Dict[str, Any]) -> int:
        total_years = 0
        for job, details in experience_relevance.items():
            years = self._extract_years_from_text(job)
            if years > 0:
                total_years += years
            elif isinstance(details, dict):
                for project, description in details.items():
                    years = self._extract_years_from_text(description)
                    if years > 0:
                        total_years += years
            elif isinstance(details, str):
                years = self._extract_years_from_text(details)
                if years > 0:
                    total_years += years
        return total_years

    def _extract_years_from_text(self, text: str) -> int:
        matches = re.findall(r'(\d+)\s*(?:year|yr)s?', text, re.IGNORECASE)
        return sum(map(int, matches))

    def _calculate_industry_match(self, experience_relevance: Dict[str, Any], required_industries: List[str]) -> float:
        candidate_industries = set()
        for job, details in experience_relevance.items():
            candidate_industries.update(self._extract_industries(job))
            if isinstance(details, dict):
                for project, description in details.items():
                    candidate_industries.update(self._extract_industries(description))
            elif isinstance(details, str):
                candidate_industries.update(self._extract_industries(details))

        required_industries_set = set(required_industries)
        matched_industries = candidate_industries.intersection(required_industries_set)
        return len(matched_industries) / len(required_industries_set) * 100 if required_industries_set else 100

    def _extract_industries(self, text: str) -> Set[str]:
        doc = self.nlp(text.lower())
        return set(ent.text for ent in doc.ents if ent.label_ == "ORG")

score_calculator = ScoreCalculator()