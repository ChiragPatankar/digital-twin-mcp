from typing import Any, Dict, List, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.embeddings.base import Embeddings
from .base import BaseLLM, LLMConfig

class DigitalTwinLLM(LLM):
    """LangChain wrapper for our LLM implementations."""
    
    llm: BaseLLM
    
    def __init__(self, llm: BaseLLM):
        """Initialize with a BaseLLM implementation."""
        super().__init__()
        self.llm = llm
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return f"digital_twin_{self.llm.config.provider}"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the LLM and return the response."""
        response = self.llm.generate(prompt, **kwargs)
        return response.text

class DigitalTwinEmbeddings(Embeddings):
    """LangChain wrapper for our embedding implementations."""
    
    llm: BaseLLM
    
    def __init__(self, llm: BaseLLM):
        """Initialize with a BaseLLM implementation."""
        super().__init__()
        self.llm = llm
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return [self.llm.embed(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query."""
        return self.llm.embed(text) 