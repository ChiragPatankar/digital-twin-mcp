from typing import Dict, Type
from .base import BaseLLM, LLMConfig
from .openai_llm import OpenAILLM
# Import other LLM implementations as they are created

class LLMFactory:
    """Factory for creating LLM instances."""
    
    _providers: Dict[str, Type[BaseLLM]] = {
        "openai": OpenAILLM,
        # Add other providers as they are implemented
    }
    
    @classmethod
    def create(cls, config: LLMConfig) -> BaseLLM:
        """Create an LLM instance based on the configuration."""
        provider = config.provider.lower()
        if provider not in cls._providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        return cls._providers[provider](config)
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLM]) -> None:
        """Register a new LLM provider."""
        cls._providers[name.lower()] = provider_class 