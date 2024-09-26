from typing import Dict, Any, List
import re
import yaml
import os

class ScoreCalculator:
    def __init__(self):
        self.job_roles = self.load_job_roles()

    def load_job_roles(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(script_dir, 'job_roles.yaml')
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)['job_roles']

    def calculate_score(self, llm_result: Dict[str, Any], job_title: str) -> int:
        print(f"Debug: Received job title: '{job_title}'")
        print(f"Debug: Available job roles: {list(self.job_roles.keys())}")
        
        # Remove "- C3" if present and normalize
        normalized_job_title = job_title.replace("- C3", "").strip().title()
        print(f"Debug: Normalized job title: '{normalized_job_title}'")
        
        if normalized_job_title not in self.job_roles:
            # Try to find a partial match
            partial_matches = [role for role in self.job_roles.keys() if role in normalized_job_title]
            if partial_matches:
                normalized_job_title = partial_matches[0]
                print(f"Debug: Partial match found: '{normalized_job_title}'")
            else:
                raise ValueError(f"Unknown job title: {job_title}. Normalized: {normalized_job_title}")

        job_role = self.job_roles[normalized_job_title]
        skills_score = self._calculate_skills_score(llm_result.get("Skills Assessment", {}), job_role)
        experience_score = self._calculate_experience_score(llm_result.get("Experience and Project Relevance", {}), job_role)
        education_score = self._calculate_education_score(llm_result.get("Education", ""), job_role)
        
        weighted_score = (skills_score * 0.5) + (experience_score * 0.4) + (education_score * 0.1)
        transformed_score = self._apply_score_transformation(weighted_score)
        
        return min(100, int(transformed_score))

    def _calculate_skills_score(self, skills_assessment: Dict[str, List[str]], job_role: Dict[str, Any]) -> float:
        total_score = 0
        total_weight = 0
        required_skills = {skill.lower(): 10 for skill in job_role.get('required_skills', [])}
        specific_knowledge = {skill.lower(): 8 for skill in job_role.get('specific_knowledge', [])}
        soft_skills = {skill.lower(): 5 for skill in job_role.get('soft_skills', [])}

        for category, skills in skills_assessment.items():
            for skill in skills:
                skill_lower = skill.lower()
                if skill_lower in required_skills:
                    total_score += required_skills[skill_lower] * 1.2
                    total_weight += 12
                elif skill_lower in specific_knowledge:
                    total_score += specific_knowledge[skill_lower] * 1.1
                    total_weight += 8.8
                elif skill_lower in soft_skills:
                    total_score += soft_skills[skill_lower]
                    total_weight += 5
                else:
                    total_score += 2
                    total_weight += 10
        
        return (total_score / total_weight) * 100 if total_weight > 0 else 0

    def _calculate_experience_score(self, experience_relevance: Dict[str, Any], job_role: Dict[str, Any]) -> float:
        total_score = 0
        total_weight = 0
        required_experience = job_role.get('years_of_experience', 0)
        
        for project, relevance in experience_relevance.items():
            score = self._extract_score(relevance)
            total_score += score
            total_weight += 10
        
        project_count_multiplier = min(len(experience_relevance) / required_experience, 1.5)
        
        return (total_score / total_weight) * 100 * project_count_multiplier if total_weight > 0 else 0

    def _calculate_education_score(self, education: str, job_role: Dict[str, Any]) -> float:
        required_education = job_role.get('education_level', '').lower()
        if "phd" in education.lower() and "phd" in required_education:
            return 100
        elif ("ms" in education.lower() or "master" in education.lower()) and "ms" in required_education:
            return 90
        elif ("bs" in education.lower() or "bachelor" in education.lower()) and "bs" in required_education:
            return 80
        else:
            return 60

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
                return 4
            else:
                return 2

    def _apply_score_transformation(self, score: float) -> float:
        if score >= 85:
            return score * 1.1
        elif score >= 70:
            return score * 1.05
        elif score >= 55:
            return score * 0.95
        else:
            return score * 0.9

score_calculator = ScoreCalculator()