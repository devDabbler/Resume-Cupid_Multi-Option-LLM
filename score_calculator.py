from typing import Dict, Any, List
import re

class ScoreCalculator:
    def __init__(self):
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

    def calculate_score(self, llm_result: Dict[str, Any]) -> int:
        skills_score = self._calculate_skills_score(llm_result.get("Skills Assessment", {}))
        experience_score = self._calculate_experience_score(llm_result.get("Experience and Project Relevance", {}))
        education_score = self._calculate_education_score(llm_result.get("Education", ""))
        
        # Adjust weights to prioritize relevant experience and core skills
        weighted_score = (skills_score * 0.4) + (experience_score * 0.5) + (education_score * 0.1)
        
        # Apply penalties for missing key skills, but with reduced impact
        missing_skills_penalty = self._calculate_missing_skills_penalty(llm_result.get("Missing Critical Skills", []))
        
        # Apply a non-linear transformation to better differentiate scores
        transformed_score = self._apply_score_transformation(weighted_score)
        
        # Apply missing skills penalty with reduced impact
        final_score = max(0, transformed_score - missing_skills_penalty * 0.4)
        
        # Cap the maximum score at 98 if there are any missing critical skills
        if missing_skills_penalty > 0:
            final_score = min(final_score, 98)
        
        return min(100, int(final_score))

    def _calculate_skills_score(self, skills_assessment: Dict[str, List[str]]) -> float:
        total_score = 0
        total_weight = 0
        
        for category, skills in skills_assessment.items():
            for skill in skills:
                skill_lower = skill.lower()
                if skill_lower in self.core_skills:
                    total_score += self.core_skills[skill_lower] * 1.3  # Further increase importance of core skills
                    total_weight += 13
                elif skill_lower in self.preferred_skills:
                    total_score += self.preferred_skills[skill_lower] * 1.2  # Increase importance of preferred skills
                    total_weight += 12
                elif skill_lower in self.programming_skills:
                    total_score += self.programming_skills[skill_lower]
                    total_weight += 8
                elif skill_lower in self.tools_and_frameworks:
                    total_score += self.tools_and_frameworks[skill_lower]
                    total_weight += 8
                else:
                    total_score += 3  # Maintain score for unrecognized skills
                    total_weight += 10
        
        return (total_score / total_weight) * 100 if total_weight > 0 else 0

    def _calculate_experience_score(self, experience_relevance: Dict[str, Any]) -> float:
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
        
        # Apply a multiplier based on the number of relevant projects
        project_count_multiplier = min(len(experience_relevance) / 2, 1.5)  # Increase the impact of multiple relevant projects
        
        return (total_score / total_weight) * 100 * project_count_multiplier if total_weight > 0 else 0

    def _calculate_education_score(self, education: str) -> float:
        if "PhD" in education:
            return 100
        elif "MS" in education or "Master" in education:
            return 95
        elif "BS" in education or "Bachelor" in education:
            return 90
        else:
            return 80  # Maintain base score for candidates without relevant degrees

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
                return 3  # Maintain base score for irrelevant experience

    def _apply_score_transformation(self, score: float) -> float:
        # Apply a non-linear transformation to better differentiate scores
        if score >= 85:
            return score * 1.06  # Further reduce boost for high scores
        elif score >= 70:
            return score * 1.04  # Slightly reduce boost for good scores
        elif score >= 55:
            return score * 0.98  # Maintain slight penalty for mediocre scores
        else:
            return score * 0.95  # Maintain penalty for low scores

    def _calculate_missing_skills_penalty(self, missing_skills: List[str]) -> float:
        penalty = 0
        for skill in missing_skills:
            skill_lower = skill.lower()
            if skill_lower in self.preferred_skills:
                penalty += self.preferred_skills[skill_lower] * 0.4  # Increase penalty for missing preferred skills
            elif skill_lower in self.core_skills:
                penalty += self.core_skills[skill_lower] * 0.3  # Increase penalty for missing core skills
        return penalty

score_calculator = ScoreCalculator()