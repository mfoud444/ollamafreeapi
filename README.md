# OllamaFreeAPI 

[![PyPI Version](https://img.shields.io/pypi/v/ollamafreeapi)](https://pypi.org/project/ollamafreeapi/)
[![Python Versions](https://img.shields.io/pypi/pyversions/ollamafreeapi)](https://pypi.org/project/ollamafreeapi/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Free API](https://img.shields.io/badge/Free%20Forever-✓-success)](https://pypi.org/project/ollamafreeapi/)


# Unlock AI Innovation for Free

**Access the world's best open language models in one place!**  

OllamaFreeAPI provides free access to leading open-source LLMs including:
- 🦙 **LLaMA** (Meta)
- 🌪️ **Mistral** (Mistral AI)
- 🔍 **DeepSeek** (DeepSeek)
- 🦄 **Qwen** (Alibaba Cloud) 

No payments. No credit cards. Just pure AI power at your fingertips.

```bash
pip install ollamafreeapi
```

## 📚 Documentation

- [API Reference](docs/client.md) - Complete API documentation
- [Usage Examples](docs/examples.md) - Practical code examples
- [Model Catalog](docs/models.md) - Available models and their capabilities

## Why Choose OllamaFreeAPI?

| Feature | Others | OllamaFreeAPI |
|---------|--------|---------------|
| Free Access | ❌ Limited trials | ✅ Always free |
| Model Variety | 3-5 models | Verified endpoints only |
| Reliability | Highly variable | Validated active models |
| Ease of Use | Complex setup | Zero-config |
| Community Support | Paid only | Free & active |

## 📊 Project Statistics

Here are some key statistics about the current state of OllamaFreeAPI:

*   **Active Models:** 16 (Ready to use and tested)
*   **Model Families:** 3 (gemma, llama, qwen)
*   **Endpoints:** 6 highly reliable server nodes

## 🚀 Quick Start

### Streaming Example
```python
from ollamafreeapi import OllamaFreeAPI

client = OllamaFreeAPI()

# Stream responses in real-time
for chunk in client.stream_chat('What is quantum computing?', model='llama3.2:3b'):
    print(chunk, end='', flush=True)
```

### Non-Streaming Example
```python
from ollamafreeapi import OllamaFreeAPI

client = OllamaFreeAPI()

# Get instant responses
response = client.chat(
    model="gpt-oss:20b",
    prompt="Explain neural networks like I'm five",
    temperature=0.7
)
print(response)
```

## 🌟 Featured Models

### Popular Foundation Models
- `llama3.2:3b` - Meta's efficient 3.2B parameter model
- `deepseek-r1:latest` - Strong reasoning capabilities built on Qwen
- `gpt-oss:20b` - Powerful Gemma-based 20B completion model
- `mistral:latest` - High-performance baseline Mistral model

### Specialized Models
- `mistral-nemo:custom` - 12.2B open weights language model
- `bakllava:latest` - Vision and language model
- `smollm2:135m` - Extremely lightweight assistant

## 🌍 Global Infrastructure

Our free API is powered by distributed community nodes:
- Fast response times
- Automatic load balancing and server selection
- Real-time availability checks

## 📄 API Reference

### Core Methods
```python
# List available models
api.list_models()  

# Get model details
api.get_model_info("mistral:latest")  

# Generate text
api.chat(model="llama3.2:3b", prompt="Your message")

# Stream responses
for chunk in api.stream_chat(prompt="Hello!", model="llama3:latest"):
    print(chunk, end='')
```

### Advanced Features
```python
# Check server locations
api.get_model_servers("deepseek-r1:latest")

# Generate raw API request
api.generate_api_request(model="llama3.2:3b", prompt="Hello")

# Get random model parameters (useful for LangChain integration)
api.get_llm_params()
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

Open-source MIT license - [View License](LICENSE)

## 🔗 Links

- [Documentation](docs/client.md)
- [Examples](docs/examples.md)
- [GitHub Issues](https://github.com/yourusername/ollamafreeapi/issues)
