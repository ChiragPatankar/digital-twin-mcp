# Self-Evolving Digital Twin MCP Server

A sophisticated digital twin system that simulates user-specific responses, maintains evolving personality profiles, and manages vectorized memories using advanced language models.

## 🌟 Features

- **Personality Simulation**: Generate contextually appropriate responses based on personality traits and communication style
- **Memory Management**: Store and retrieve vectorized memories with semantic search capabilities
- **Self-Evolution**: Automatically update personality traits and knowledge based on interactions
- **Reflection System**: Perform periodic self-reflection and summarization of experiences
- **Modular Architecture**: Easily extensible system with clear separation of concerns

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- HuggingFace API key (optional, for DialoGPT)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/digital-twin-mcp.git
cd digital-twin-mcp
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
DATABASE_URL=sqlite:///digital_twin.db
```

## 📁 Project Structure

```
digital-twin-mcp/
├── src/
│   └── fastmcp/
│       └── digital_twin/
│           ├── __init__.py
│           ├── personality.py      # Personality system
│           ├── memory.py          # Memory management
│           ├── response_generator.py # Response generation
│           ├── interaction.py     # Interaction handling
│           ├── server.py          # FastAPI server
│           └── prompts/           # Prompt templates
│               ├── base_personality.txt
│               ├── reply_simulation.txt
│               ├── memory_update.txt
│               └── reflection.txt
├── examples/
│   └── digital_twin_example.py    # Usage examples
├── tests/                         # Test suite
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## 💡 Usage

### Basic Example

```python
from fastmcp.digital_twin import DigitalTwinInteraction, ResponseGenerator, Memory, Personality

# Initialize components
response_generator = ResponseGenerator()
memory = Memory()
personality = Personality()

# Create interaction handler
interaction = DigitalTwinInteraction(
    response_generator=response_generator,
    memory=memory,
    personality=personality
)

# Simulate a response
response, metadata = await interaction.simulate_response(
    user_input="Hello, how are you?",
    context={"mood": "casual"}
)

# Perform reflection
insights = await interaction.reflect_chain(time_period="1w")
```

### Advanced Features

1. **Memory Management**:
```python
# Add a memory
await memory.add_memory(
    content="User prefers formal communication",
    memory_type="preference",
    metadata={"importance": 0.8}
)

# Retrieve relevant memories
relevant_memories = await memory.get_relevant_memories(
    query="communication style",
    limit=5
)
```

2. **Personality Evolution**:
```python
# Get current personality
traits = personality.get_traits()

# Update personality
await personality.process_update({
    "trait": "formality",
    "value": 0.8,
    "reason": "User preference"
})
```

3. **Interaction Routing**:
```python
# Route through different chains
result = await interaction.route_interaction(
    user_input="Tell me about yourself",
    interaction_type="simulate",
    context={"depth": "detailed"}
)
```

## 🔧 Configuration

The system can be configured through environment variables or a configuration file. Key settings include:

- `OPENAI_MODEL`: Language model to use (default: "gpt-4")
- `MAX_MEMORIES`: Maximum number of memories to store (default: 1000)
- `PERSONALITY_UPDATE_INTERVAL`: How often to update personality (default: 3600s)
- `MEMORY_CHUNK_SIZE`: Size of memory chunks (default: 1000)



## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- OpenAI for GPT models
- HuggingFace for DialoGPT
- FastAPI for the web framework
- SQLAlchemy for database management 
