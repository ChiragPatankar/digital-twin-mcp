from typing import Dict, Any, Tuple
from transformers import pipeline
import numpy as np

class SentimentAnalyzer:
    """Analyzes sentiment and emotional context in text."""
    
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.emotion_analyzer = pipeline("text-classification", 
                                       model="j-hartmann/emotion-english-distilroberta-base")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze both sentiment and emotion in the text."""
        # Get basic sentiment
        sentiment = self.sentiment_analyzer(text)[0]
        
        # Get detailed emotion
        emotion = self.emotion_analyzer(text)[0]
        
        # Combine results
        return {
            "sentiment": {
                "label": sentiment["label"],
                "score": sentiment["score"]
            },
            "emotion": {
                "label": emotion["label"],
                "score": emotion["score"]
            }
        }
    
    def get_emotional_context(self, text: str) -> Dict[str, float]:
        """Convert sentiment and emotion analysis into personality-relevant context."""
        analysis = self.analyze(text)
        
        # Map sentiment and emotion to personality traits
        context = {
            "neuroticism": 0.5,  # Default value
            "agreeableness": 0.5,
            "extraversion": 0.5
        }
        
        # Adjust neuroticism based on negative sentiment
        if analysis["sentiment"]["label"] == "NEGATIVE":
            context["neuroticism"] = min(1.0, 0.5 + analysis["sentiment"]["score"])
        
        # Adjust agreeableness based on emotion
        if analysis["emotion"]["label"] in ["joy", "love"]:
            context["agreeableness"] = min(1.0, 0.5 + analysis["emotion"]["score"])
        elif analysis["emotion"]["label"] in ["anger", "fear"]:
            context["agreeableness"] = max(0.0, 0.5 - analysis["emotion"]["score"])
        
        # Adjust extraversion based on emotion intensity
        if analysis["emotion"]["score"] > 0.7:
            context["extraversion"] = min(1.0, 0.5 + analysis["emotion"]["score"])
        
        return context 