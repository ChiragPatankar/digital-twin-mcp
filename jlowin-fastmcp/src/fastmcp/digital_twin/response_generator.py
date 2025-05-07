from typing import Dict, List, Any, Optional
from .llm.base import LLMConfig, LLMResponse
from .llm.factory import LLMFactory
from .llm.langchain_wrapper import DigitalTwinLLM, DigitalTwinEmbeddings
from .prompts.manager import PromptManager
import yaml
import os
import logging

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generates contextual responses based on personality and memories."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the response generator with configuration."""
        self.config = self._load_config(config_path)
        self.llm = LLMFactory.create(LLMConfig(**self.config["llm"]))
        self.langchain_llm = DigitalTwinLLM(self.llm)
        self.embeddings = DigitalTwinEmbeddings(self.llm)
        self.prompt_manager = PromptManager()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    async def generate_response(
        self,
        prompt: str,
        personality_traits: Dict[str, float],
        relevant_memories: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Generate a response based on context and personality."""
        try:
            # Format memories for the prompt
            memories_str = "\n".join(
                f"- {memory['content']}" for memory in relevant_memories
            )
            
            # Get communication style from config
            style = self.config["personality"]["response_style"]
            style_str = (
                f"Formal (level {style['formality_level']:.1f}), "
                f"Enthusiastic (level {style['enthusiasm_level']:.1f}), "
                f"Polite (level {style['politeness_level']:.1f})"
            )
            
            # Format the reply prompt
            formatted_prompt = self.prompt_manager.format_reply(
                name=context.get("name", "Digital Twin"),
                message=prompt,
                context=str(context) if context else "No additional context",
                memories=memories_str,
                traits=personality_traits,
                style=style_str
            )
            
            # Generate response using LLM
            response = await self.llm.generate(
                prompt=formatted_prompt,
                context=context,
                **kwargs
            )
            
            # Adjust response style based on personality
            adjusted_response = self._adjust_response_style(
                response.text,
                personality_traits
            )
            
            return adjusted_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def _adjust_response_style(
        self,
        response: str,
        personality_traits: Dict[str, float]
    ) -> str:
        """Adjust the response style based on personality traits."""
        # Get response style configuration
        style_config = self.config["personality"]["response_style"]
        
        # Adjust formality based on conscientiousness
        if personality_traits.get("conscientiousness", 0.5) > style_config["formality_level"]:
            response = response.replace("gonna", "going to")
            response = response.replace("wanna", "want to")
            response = response.replace("yeah", "yes")
            response = response.replace("nope", "no")
        
        # Adjust enthusiasm based on extraversion
        if personality_traits.get("extraversion", 0.5) > style_config["enthusiasm_level"]:
            response = response.replace(".", "!")
            response = response.replace("!", "!!")
            if not response.endswith("!"):
                response += "!"
        
        # Adjust politeness based on agreeableness
        if personality_traits.get("agreeableness", 0.5) > style_config["politeness_level"]:
            response = "I think " + response.lower()
            if not any(word in response.lower() for word in ["please", "thank", "appreciate"]):
                response += " Thank you for asking!"
        
        return response
    
    async def analyze_interaction(
        self,
        interaction: str,
        current_personality: Dict[str, float],
        current_knowledge: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Analyze an interaction for personality and knowledge updates."""
        try:
            # Format current state
            personality_str = "\n".join(
                f"- {trait}: {value:.2f}" for trait, value in current_personality.items()
            )
            knowledge_str = "\n".join(
                f"- {item['content']}" for item in current_knowledge
            )
            
            # Format the memory update prompt
            formatted_prompt = self.prompt_manager.format_memory_update(
                interaction=interaction,
                current_personality=personality_str,
                current_knowledge=knowledge_str
            )
            
            # Get analysis from LLM
            response = await self.llm.generate(prompt=formatted_prompt)
            
            # Parse the response into sections
            sections = {}
            current_section = None
            current_items = []
            
            for line in response.text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                if line.endswith(":"):
                    if current_section:
                        sections[current_section] = current_items
                    current_section = line[:-1].lower()
                    current_items = []
                elif line.startswith("- "):
                    current_items.append(line[2:])
            
            if current_section:
                sections[current_section] = current_items
            
            return sections
            
        except Exception as e:
            logger.error(f"Error analyzing interaction: {e}")
            raise 