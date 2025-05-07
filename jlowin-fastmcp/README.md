# FastMCP Digital Twin

A self-evolving digital twin implementation using the Model Context Protocol (MCP). This project creates a digital twin that simulates user behavior and responses based on evolving personality traits, memory, and preferences.

## Features

- Personality system with evolving traits
- Persistent memory storage with SQLite
- MCP server integration for tool exposure
- Memory search and retrieval
- Personality evolution based on interactions

## Installation

```bash
pip install -e .
```

## Usage

```python
from fastmcp.digital_twin.server import DigitalTwinServer

# Create a digital twin
twin = DigitalTwinServer(
    name="Alice",
    initial_traits={
        "openness": 0.8,
        "conscientiousness": 0.7,
        "extraversion": 0.6,
        "agreeableness": 0.9,
        "neuroticism": 0.3
    }
)

# Process an interaction
response = twin.process_interaction(
    prompt="What do you think about AI?",
    context={
        "openness": 0.9,
        "topic": "technology"
    }
)

# Run the MCP server
twin.run()
```

## Project Structure

```
jlowin-fastmcp/
├── src/
│   └── fastmcp/
│       └── digital_twin/
│           ├── personality.py
│           ├── memory.py
│           └── server.py
├── examples/
│   └── digital_twin_example.py
└── pyproject.toml
```

## Development

1. Clone the repository
2. Install dependencies: `pip install -e .`
3. Run the example: `python examples/digital_twin_example.py`

## License

MIT License 