from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from .response_generator import ResponseGenerator
from .memory import Memory
from .personality import Personality

logger = logging.getLogger(__name__)

class DigitalTwinInteraction:
    """Handles interactions with the digital twin."""
    
    def __init__(
        self,
        response_generator: ResponseGenerator,
        memory: Memory,
        personality: Personality,
        config_path: str = "config.yaml"
    ):
        """Initialize the interaction handler.
        
        Args:
            response_generator: Response generator instance
            memory: Memory system instance
            personality: Personality system instance
            config_path: Path to configuration file
        """
        self.response_generator = response_generator
        self.memory = memory
        self.personality = personality
        self.config = self.response_generator.config
        self.interaction_history: List[Dict[str, Any]] = []
    
    async def simulate_response(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Simulate the digital twin's response to user input.
        
        Args:
            user_input: The user's message
            context: Optional additional context
            
        Returns:
            Tuple of (response, metadata)
        """
        try:
            # Get relevant memories
            relevant_memories = await self.memory.get_relevant_memories(
                user_input,
                limit=self.config["memory"]["max_memories"]
            )
            
            # Get current personality state
            personality_traits = self.personality.get_traits()
            
            # Generate response
            response = await self.response_generator.generate_response(
                prompt=user_input,
                personality_traits=personality_traits,
                relevant_memories=relevant_memories,
                context=context
            )
            
            # Record interaction
            interaction = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_input": user_input,
                "response": response,
                "context": context,
                "relevant_memories": relevant_memories
            }
            self.interaction_history.append(interaction)
            
            # Update profile after interaction
            await self.update_profile(interaction)
            
            return response, {
                "relevant_memories": relevant_memories,
                "personality_traits": personality_traits,
                "interaction_id": len(self.interaction_history) - 1
            }
            
        except Exception as e:
            logger.error(f"Error in simulate_response: {e}")
            raise
    
    async def update_profile(self, interaction: Dict[str, Any]) -> None:
        """Update the digital twin's profile based on interaction.
        
        Args:
            interaction: The interaction to analyze
        """
        try:
            # Get current state
            current_personality = self.personality.get_traits()
            current_knowledge = await self.memory.get_recent_memories(
                limit=self.config["memory"]["max_memories"]
            )
            
            # Analyze interaction
            analysis = await self.response_generator.analyze_interaction(
                interaction=f"User: {interaction['user_input']}\nAssistant: {interaction['response']}",
                current_personality=current_personality,
                current_knowledge=current_knowledge
            )
            
            # Update personality
            if "personality_updates" in analysis:
                for update in analysis["personality_updates"]:
                    await self.personality.process_update(update)
            
            # Update memory
            if "knowledge_updates" in analysis:
                for update in analysis["knowledge_updates"]:
                    await self.memory.add_memory(
                        content=update,
                        memory_type="semantic",
                        metadata={"source": "interaction_analysis"}
                    )
            
            if "memory_formation" in analysis:
                for memory in analysis["memory_formation"]:
                    await self.memory.add_memory(
                        content=memory,
                        memory_type="episodic",
                        metadata={
                            "source": "interaction_analysis",
                            "interaction_id": len(self.interaction_history) - 1
                        }
                    )
            
        except Exception as e:
            logger.error(f"Error in update_profile: {e}")
            raise
    
    async def reflect_chain(
        self,
        time_period: Optional[str] = None,
        max_memories: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform longer-term reflection and summarization.
        
        Args:
            time_period: Optional time period to reflect on (e.g., "1d", "1w", "1m")
            max_memories: Optional maximum number of memories to consider
            
        Returns:
            Dictionary containing reflection insights
        """
        try:
            # Get memories for reflection
            memories = await self.memory.get_memories_for_reflection(
                time_period=time_period,
                limit=max_memories or self.config["memory"]["max_memories"]
            )
            
            # Get personality evolution
            personality_history = self.personality.get_evolution_history()
            
            # Format reflection prompt
            reflection_prompt = f"""Reflection Period: {time_period or 'all time'}

Personality Evolution:
{self._format_personality_history(personality_history)}

Key Memories:
{self._format_memories_for_reflection(memories)}

Task: Analyze this period and provide insights about:
1. How has the personality evolved?
2. What patterns emerge in the interactions?
3. What core values and beliefs have been reinforced or changed?
4. What new knowledge has been acquired?
5. How has the communication style developed?

Format your response as:
PERSONALITY_INSIGHTS:
- [List of personality insights]

INTERACTION_PATTERNS:
- [List of observed patterns]

CORE_VALUES:
- [List of core values and changes]

KNOWLEDGE_GROWTH:
- [List of knowledge acquisitions]

COMMUNICATION_EVOLUTION:
- [List of communication style developments]"""

            # Get reflection from LLM
            reflection = await self.response_generator.llm.generate(
                prompt=reflection_prompt
            )
            
            # Parse reflection into sections
            sections = self._parse_reflection_sections(reflection.text)
            
            # Store reflection as a special memory
            await self.memory.add_memory(
                content=reflection.text,
                memory_type="semantic",
                metadata={
                    "type": "reflection",
                    "time_period": time_period,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            return sections
            
        except Exception as e:
            logger.error(f"Error in reflect_chain: {e}")
            raise
    
    def _format_personality_history(
        self,
        history: List[Dict[str, Any]]
    ) -> str:
        """Format personality history for reflection."""
        formatted = []
        for entry in history:
            formatted.append(f"Time: {entry['timestamp']}")
            for trait, value in entry['traits'].items():
                formatted.append(f"- {trait}: {value:.2f}")
            formatted.append("")
        return "\n".join(formatted)
    
    def _format_memories_for_reflection(
        self,
        memories: List[Dict[str, Any]]
    ) -> str:
        """Format memories for reflection."""
        formatted = []
        for memory in memories:
            formatted.append(f"Time: {memory['timestamp']}")
            formatted.append(f"Type: {memory['type']}")
            formatted.append(f"Content: {memory['content']}")
            if memory.get('metadata'):
                formatted.append(f"Metadata: {memory['metadata']}")
            formatted.append("")
        return "\n".join(formatted)
    
    def _parse_reflection_sections(self, reflection: str) -> Dict[str, List[str]]:
        """Parse reflection text into sections."""
        sections = {}
        current_section = None
        current_items = []
        
        for line in reflection.split("\n"):
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
    
    async def route_interaction(
        self,
        user_input: str,
        interaction_type: str = "simulate",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Route user input through appropriate interaction chain.
        
        Args:
            user_input: The user's input
            interaction_type: Type of interaction to perform
            context: Optional additional context
            
        Returns:
            Dictionary containing interaction results
        """
        try:
            if interaction_type == "simulate":
                # Generate response
                response, metadata = await self.simulate_response(
                    user_input=user_input,
                    context=context
                )
                return {
                    "type": "simulate",
                    "response": response,
                    "metadata": metadata
                }
                
            elif interaction_type == "reflect":
                # Perform reflection
                insights = await self.reflect_chain(
                    time_period=context.get("time_period"),
                    max_memories=context.get("max_memories")
                )
                return {
                    "type": "reflect",
                    "insights": insights
                }
                
            elif interaction_type == "update":
                # Update profile
                await self.update_profile({
                    "user_input": user_input,
                    "context": context
                })
                return {
                    "type": "update",
                    "status": "success"
                }
                
            else:
                raise ValueError(f"Unknown interaction type: {interaction_type}")
                
        except Exception as e:
            logger.error(f"Error in route_interaction: {e}")
            raise 