from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel

@dataclass
class LLMResponse:
    """Structured response from an LLM."""
    text: str
    raw_response: Any
    metadata: Dict[str, Any]

class LLMConfig(BaseModel):
    """Base configuration for LLM providers."""
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2

class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for the given text."""
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in the given text."""
        pass 