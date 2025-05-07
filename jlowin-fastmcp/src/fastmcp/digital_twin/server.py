from typing import Dict, List, Optional, Any
from fastmcp import FastMCP
from .personality import Personality
from .memory import MemoryManager
from .response_generator import ResponseGenerator
from .sentiment import SentimentAnalyzer
import json
from datetime import datetime

class DigitalTwinServer:
    """A self-evolving digital twin MCP server that simulates user behavior and responses."""
    
    def __init__(self, name: str, initial_traits: Optional[Dict[str, float]] = None):
        self.name = name
        self.personality = Personality()
        self.memory_manager = MemoryManager()
        self.response_generator = ResponseGenerator()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.mcp = FastMCP(f"{name}'s Digital Twin")
        
        # Initialize with default traits if none provided
        if initial_traits is None:
            initial_traits = {
                "openness": 0.5,
                "conscientiousness": 0.5,
                "extraversion": 0.5,
                "agreeableness": 0.5,
                "neuroticism": 0.5
            }
        
        for trait_name, value in initial_traits.items():
            self.personality.add_trait(trait_name, value)
        
        self._setup_mcp_tools()
    
    def _setup_mcp_tools(self):
        """Set up MCP tools for the digital twin."""
        
        @self.mcp.tool()
        def get_personality() -> Dict[str, float]:
            """Get the current personality traits."""
            return {name: trait.value for name, trait in self.personality.traits.items()}
        
        @self.mcp.tool()
        def add_memory(content: str, context: Dict[str, Any], 
                      importance: int = 50, category: str = "general") -> None:
            """Add a new memory to the digital twin."""
            self.memory_manager.add_memory(content, context, importance, category)
        
        @self.mcp.tool()
        def get_recent_memories(limit: int = 10) -> List[Dict[str, Any]]:
            """Get the most recent memories."""
            return self.memory_manager.get_recent_memories(limit)
        
        @self.mcp.tool()
        def search_memories(query: str, limit: int = 5) -> List[Dict[str, Any]]:
            """Search memories based on content."""
            return self.memory_manager.search_memories(query, limit)
        
        @self.mcp.tool()
        def update_personality(trait_updates: Dict[str, float]) -> None:
            """Update personality traits based on interaction."""
            self.personality.evolve(trait_updates)
        
        @self.mcp.tool()
        def analyze_sentiment(text: str) -> Dict[str, Any]:
            """Analyze the sentiment and emotion in text."""
            return self.sentiment_analyzer.analyze(text)
    
    def process_interaction(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Process an interaction with the digital twin.
        This is where the personality and memory systems work together to generate a response.
        """
        # Analyze sentiment and get emotional context
        sentiment_analysis = self.sentiment_analyzer.analyze(prompt)
        emotional_context = self.sentiment_analyzer.get_emotional_context(prompt)
        
        # Merge emotional context with provided context
        full_context = {**context, **emotional_context}
        
        # Store the interaction in memory with sentiment analysis
        self.memory_manager.add_memory(
            content=prompt,
            context={
                **full_context,
                "sentiment": sentiment_analysis
            },
            importance=70,  # High importance for direct interactions
            category="interaction"
        )
        
        # Update personality based on interaction context
        personality_context = {
            trait: full_context.get(trait, 0.5)
            for trait in self.personality.traits.keys()
        }
        self.personality.evolve(personality_context)
        
        # Get relevant memories for context
        relevant_memories = self.memory_manager.search_memories(prompt, limit=3)
        
        # Generate response using personality traits and memories
        response = self.response_generator.generate_response(
            prompt=prompt,
            personality_traits=self.personality.traits,
            relevant_memories=relevant_memories
        )
        
        return response
    
    def run(self, host: str = "localhost", port: int = 8000):
        """Run the digital twin MCP server."""
        self.mcp.run(host=host, port=port) 