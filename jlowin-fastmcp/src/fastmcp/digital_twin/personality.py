from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime

class PersonalityTrait(BaseModel):
    """Represents a single personality trait with its value and evolution history."""
    name: str
    value: float = Field(default=0.5, ge=0.0, le=1.0)
    history: List[Dict[str, float]] = Field(default_factory=list)
    
    def update(self, new_value: float, timestamp: Optional[datetime] = None) -> None:
        """Update the trait value and record its history."""
        self.value = max(0.0, min(1.0, new_value))
        self.history.append({
            "value": self.value,
            "timestamp": timestamp or datetime.now()
        })

class Personality(BaseModel):
    """Manages the digital twin's personality traits and their evolution."""
    traits: Dict[str, PersonalityTrait] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def add_trait(self, name: str, initial_value: float = 0.5) -> None:
        """Add a new personality trait."""
        self.traits[name] = PersonalityTrait(name=name, value=initial_value)
    
    def update_trait(self, name: str, new_value: float) -> None:
        """Update a specific trait's value."""
        if name in self.traits:
            self.traits[name].update(new_value)
            self.last_updated = datetime.now()
    
    def get_trait_vector(self) -> np.ndarray:
        """Get all trait values as a numpy array for ML processing."""
        return np.array([trait.value for trait in self.traits.values()])
    
    def evolve(self, interaction_context: Dict[str, float]) -> None:
        """
        Evolve personality based on interaction context.
        This is a simple implementation that can be enhanced with more sophisticated ML models.
        """
        for trait_name, trait in self.traits.items():
            if trait_name in interaction_context:
                # Simple weighted update
                current_value = trait.value
                context_value = interaction_context[trait_name]
                # Gradually move towards the context value
                new_value = current_value * 0.9 + context_value * 0.1
                self.update_trait(trait_name, new_value) 