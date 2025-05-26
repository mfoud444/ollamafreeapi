Here's the enhanced README.md with clear information about the relationship with ollama-python:

```markdown
# OllamaFreeAPI

[![PyPI Version](https://img.shields.io/pypi/v/ollamafreeapi)](https://pypi.org/project/ollamafreeapi/)
[![Python Versions](https://img.shields.io/pypi/pyversions/ollamafreeapi)](https://pypi.org/project/ollamafreeapi/)
[![License](https://img.shields.io/pypi/l/ollamafreeapi)](https://opensource.org/licenses/MIT)
[![GitHub Actions](https://github.com/mfoud444/ollamafreeapi/actions/workflows/publish.yml/badge.svg)](https://github.com/mfoud444/ollamafreeapi/actions)

A lightweight, zero-configuration Python client for interacting with LLMs served via Ollama across distributed servers.

## Relationship with ollama-python

OllamaFreeAPI builds upon the official [ollama-python](https://github.com/ollama/ollama-python) library with these key enhancements:

- **Zero Configuration**: No need to set up endpoints or API keys
- **Pre-configured Models**: Access to ready-to-use models across distributed servers
- **Automatic Failover**: Built-in load balancing across multiple Ollama instances
- **Simplified Interface**: Higher-level abstractions for common use cases

> ℹ️ Under the hood, OllamaFreeAPI uses the official `ollama` Python client to communicate with Ollama servers.

## Features

- 🚀 **Instant Setup** - Works out-of-the-box with no configuration
- 🌐 **Distributed Network** - Automatic routing to available servers
- 🔍 **Model Discovery** - Browse available models with `list_families()` and `list_models()`
- ⚡ **Performance Optimized** - Intelligent server selection based on latency
- 🔄 **Seamless Integration** - Compatible with existing Ollama deployments

## Installation

```bash
pip install ollamafreeapi
```

## Quick Start

```python
from ollamafreeapi import OllamaFreeAPI

# Connect to the distributed Ollama network
client = OllamaFreeAPI()

# Discover available models
print("Available families:", client.list_families())
print("Mistral models:", client.list_models(family='mistral'))

# Have a conversation
response = client.chat(
    model_name="llama3:latest",
    prompt="Explain the difference between Python and JavaScript",
    temperature=0.7
)
print(response)

# Stream a response
for chunk in client.stream_chat("mistral:latest", "Write a short story about AI:"):
    print(chunk, end='', flush=True)
```

## Advanced Features

### Server Information

```python
# Get all servers hosting a model
servers = client.get_model_servers("llama2:13b")
for server in servers:
    print(f"Server: {server['url']}")
    print(f"Location: {server['location']['country']}")
    print(f"Performance: {server['performance']['tokens_per_second']} tokens/s")
```

### Request Generation

```python
# Generate the raw API request
request = client.generate_api_request(
    model_name="deepseek-r1:7b",
    prompt="Explain blockchain technology",
    temperature=0.8,
    top_p=0.95,
    num_predict=256
)
```

## Model Parameters

All API calls support these optional parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 0.7 | Controls randomness (lower = more deterministic) |
| `top_p` | float | 0.9 | Probability threshold for nucleus sampling |
| `num_predict` | int | 128 | Maximum number of tokens to generate |
| `stop` | list[str] | [] | Sequences where the model will stop generating |
| `repeat_penalty` | float | 1.1 | Penalty for repeated content |

## Frequently Asked Questions

**Q: How is this different from ollama-python?**  
A: OllamaFreeAPI provides pre-configured access to a distributed network of Ollama servers with automatic failover, while ollama-python requires manual server configuration.

**Q: Do I need to run my own Ollama server?**  
A: No! OllamaFreeAPI connects to our managed network of servers by default.

**Q: Can I use this with my existing Ollama installation?**  
A: Yes, you can configure it to use your local Ollama instance if preferred.


## License

Distributed under the MIT License. See `LICENSE` for more information.

## Support

For support, feature requests, or to report issues, please [open an issue](https://github.com/mfoud444/ollamafreeapi/issues).
