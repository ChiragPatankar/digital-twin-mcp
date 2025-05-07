from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from string import Template
import logging

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages prompt templates for the digital twin."""
    
    def __init__(self, templates_dir: Optional[str] = None):
        """Initialize the prompt manager.
        
        Args:
            templates_dir: Optional path to templates directory.
                          Defaults to the package's prompts directory.
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent
        
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, Template] = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load all template files from the templates directory."""
        for template_file in self.templates_dir.glob("*.txt"):
            try:
                with open(template_file, "r", encoding="utf-8") as f:
                    template_name = template_file.stem
                    self.templates[template_name] = Template(f.read())
                logger.info(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
    
    def get_template(self, name: str) -> Optional[Template]:
        """Get a template by name.
        
        Args:
            name: Name of the template (without .txt extension)
            
        Returns:
            The template if found, None otherwise
        """
        return self.templates.get(name)
    
    def format_prompt(
        self,
        template_name: str,
        **kwargs: Any
    ) -> str:
        """Format a prompt using the specified template.
        
        Args:
            template_name: Name of the template to use
            **kwargs: Variables to substitute in the template
            
        Returns:
            Formatted prompt string
            
        Raises:
            KeyError: If template not found
            ValueError: If template formatting fails
        """
        template = self.get_template(template_name)
        if template is None:
            raise KeyError(f"Template not found: {template_name}")
        
        try:
            return template.substitute(**kwargs)
        except Exception as e:
            raise ValueError(f"Error formatting template {template_name}: {e}")
    
    def format_personality(
        self,
        name: str,
        traits: Dict[str, float],
        style: str
    ) -> str:
        """Format the base personality prompt.
        
        Args:
            name: Name of the digital twin
            traits: Dictionary of personality traits and their values
            style: Communication style description
            
        Returns:
            Formatted personality prompt
        """
        traits_str = "\n".join(f"- {trait}: {value:.2f}" for trait, value in traits.items())
        return self.format_prompt(
            "base_personality",
            name=name,
            traits=traits_str,
            style=style
        )
    
    def format_reply(
        self,
        name: str,
        message: str,
        context: str,
        memories: str,
        traits: Dict[str, float],
        style: str
    ) -> str:
        """Format the reply simulation prompt.
        
        Args:
            name: Name of the digital twin
            message: User's message
            context: Current context
            memories: Relevant memories
            traits: Personality traits
            style: Communication style
            
        Returns:
            Formatted reply prompt
        """
        traits_str = ", ".join(f"{trait}: {value:.2f}" for trait, value in traits.items())
        return self.format_prompt(
            "reply_simulation",
            name=name,
            message=message,
            context=context,
            memories=memories,
            traits=traits_str,
            style=style
        )
    
    def format_memory_update(
        self,
        interaction: str,
        current_personality: str,
        current_knowledge: str
    ) -> str:
        """Format the memory update prompt.
        
        Args:
            interaction: Recent interaction text
            current_personality: Current personality state
            current_knowledge: Current knowledge base
            
        Returns:
            Formatted memory update prompt
        """
        return self.format_prompt(
            "memory_update",
            interaction=interaction,
            current_personality=current_personality,
            current_knowledge=current_knowledge
        ) 