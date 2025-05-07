from fastmcp.digital_twin.server import DigitalTwinServer
import time

def main():
    # Create a digital twin with custom initial personality traits
    twin = DigitalTwinServer(
        name="Alice",
        initial_traits={
            "openness": 0.8,        # High openness to new experiences
            "conscientiousness": 0.7,  # Fairly organized and responsible
            "extraversion": 0.6,    # Moderately outgoing
            "agreeableness": 0.9,   # Very agreeable and cooperative
            "neuroticism": 0.3      # Emotionally stable
        }
    )
    
    # Example interactions with different emotional contexts
    interactions = [
        {
            "prompt": "What do you think about artificial intelligence?",
            "context": {
                "openness": 0.9,  # High openness for discussing new technologies
                "topic": "technology",
                "sentiment": "positive"
            }
        },
        {
            "prompt": "I'm feeling really anxious about the future.",
            "context": {
                "neuroticism": 0.4,  # Slightly more stable to help
                "topic": "emotions",
                "sentiment": "negative"
            }
        },
        {
            "prompt": "Let's go to a party!",
            "context": {
                "extraversion": 0.8,  # More outgoing for social activities
                "topic": "social",
                "sentiment": "excited"
            }
        }
    ]
    
    # Process each interaction
    for interaction in interactions:
        print(f"\nUser: {interaction['prompt']}")
        response = twin.process_interaction(
            prompt=interaction['prompt'],
            context=interaction['context']
        )
        print(f"Digital Twin: {response}")
        
        # Get current personality state
        personality = twin.get_personality()
        print("\nCurrent Personality State:")
        for trait, value in personality.items():
            print(f"- {trait}: {value:.2f}")
        
        # Small delay between interactions
        time.sleep(1)
    
    # Run the MCP server
    print("\nStarting Digital Twin MCP Server...")
    twin.run()

if __name__ == "__main__":
    main() 