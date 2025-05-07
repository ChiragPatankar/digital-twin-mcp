import os
from typing import Dict, List, Any, Optional
import openai
from tiktoken import encoding_for_model
from .base import BaseLLM, LLMConfig, LLMResponse

class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.encoding = encoding_for_model(config.model)
    
    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using OpenAI's API."""
        messages = self._prepare_messages(prompt, context)
        
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            **kwargs
        )
        
        return LLMResponse(
            text=response.choices[0].message.content,
            raw_response=response,
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "usage": response.usage._asdict()
            }
        )
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI's API."""
        response = await self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in the given text."""
        return len(self.encoding.encode(text))
    
    def _prepare_messages(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Prepare messages for the OpenAI API."""
        messages = []
        
        if context:
            # Add system message with context
            system_message = self._format_context(context)
            messages.append({"role": "system", "content": system_message})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context into a system message."""
        context_parts = []
        
        if "personality" in context:
            traits = context["personality"]
            context_parts.append("Personality traits:")
            for trait, value in traits.items():
                context_parts.append(f"- {trait}: {value:.2f}")
        
        if "memories" in context:
            memories = context["memories"]
            context_parts.append("\nRelevant memories:")
            for memory in memories:
                context_parts.append(f"- {memory['content']}")
        
        return "\n".join(context_parts) 