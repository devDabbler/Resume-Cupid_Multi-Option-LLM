from abc import ABC, abstractmethod
from typing import Dict, Any

class LLMService(ABC):
    @abstractmethod
    def analyze_resume(self, resume_text: str, job_description: str, job_title: str) -> Dict[str, Any]:
        pass