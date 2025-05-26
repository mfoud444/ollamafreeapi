# OllamaFreeAPI 

[![PyPI Version](https://img.shields.io/pypi/v/ollamafreeapi)](https://pypi.org/project/ollamafreeapi/)
[![Python Versions](https://img.shields.io/pypi/pyversions/ollamafreeapi)](https://pypi.org/project/ollamafreeapi/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Free API](https://img.shields.io/badge/Free%20Forever-✓-success)](https://pypi.org/project/ollamafreeapi/)
[![Discord](https://img.shields.io/discord/your-server-id)](https://discord.gg/yourlink)

# Unlock AI Innovation for Free

**Access the world's best open language models in one place!**  

OllamaFreeAPI provides free access to leading open-source LLMs including:
- 🦙 **LLaMA 3** (Meta)
- 🌪️ **Mistral** (Mistral AI)
- 🔍 **DeepSeek** (DeepSeek)
- 🦄 **Qwen** (Alibaba Cloud) 

No payments. No credit cards. Just pure AI power at your fingertips.

```bash
pip install ollamafreeapi
```

## Why Choose OllamaFreeAPI?

| Feature | Others | OllamaFreeAPI |
|---------|--------|---------------|
| Free Access | ❌ Limited trials | ✅ Always free |
| Model Variety | 3-5 models | 50+ models |
| Global Infrastructure | Single region | 5 continents |
| Ease of Use | Complex setup | Zero-config |
| Community Support | Paid only | Free & active |

## Get Started in 30 Seconds

```python
from ollamafreeapi import OllamaFreeAPI

api = OllamaFreeAPI()

# Get instant responses
response = api.chat(
    model_name="llama3:latest",
    prompt="Explain neural networks like I'm five",
    temperature=0.7
)
print(response)
```

## Featured Model Catalog

### Popular Foundation Models
- `llama3:8b-instruct` - Meta's latest 8B parameter model
- `mistral:7b-v0.2` - High-performance 7B model
- `deepseek-r1:7b` - Strong reasoning capabilities
- `qwen:7b-chat` - Alibaba's versatile model

### Specialized Models
- `llama3:code` - Optimized for programming
- `mistral:storyteller` - Creative writing specialist
- `deepseek-coder` - STEM and math expert

## Global AI Infrastructure

Our free API is powered by:
- 25+ dedicated GPU servers
- 5 global regions (NA, EU, Asia)
- Automatic load balancing
- 99.5% uptime SLA

## Complete API Reference

### Core Methods
```python
# List available models
api.list_models()  

# Get model details
api.get_model_info("mistral:7b")  

# Generate text
api.chat(model_name="llama3:latest", prompt="Your message")

# Stream responses
for chunk in api.stream_chat(...):
    print(chunk, end='')
```

### Advanced Features
```python
# Check server locations
api.get_model_servers("deepseek-r1:7b")

# Generate raw API request
api.generate_api_request(...)

# Get performance metrics
api.get_server_status()
```

## Free Tier Limits

| Resource | Free Tier | Pro Tier |
|----------|-----------|----------|
| Requests | 100/hr | 10,000/hr |
| Tokens | 16k | 128k |
| Speed | 50 t/s | 150 t/s |
| Models | 7B only | All sizes |


## License

Open-source MIT license - [View License](LICENSE)

> 💡 Pro Tip: Star this repo to get updates on new model additions!
```
