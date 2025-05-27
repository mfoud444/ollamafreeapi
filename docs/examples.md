# OllamaFreeAPI Usage Examples

## Basic Setup
```python
from ollamafreeapi import OllamaFreeAPI

# Initialize the client
client = OllamaFreeAPI()
```

## Model Management Examples

### Listing Available Models
```python
# List all model families
families = client.list_families()
print("Available families:", families)

# List all models in a specific family
llama_models = client.list_models("llama")
print("Llama models:", llama_models)

# List all available models
all_models = client.list_models()
print("All models:", all_models)
```

### Getting Model Information
```python
# Get information about a specific model
model_info = client.get_model_info("llama2")
print("Model info:", model_info)

# Get available servers for a model
servers = client.get_model_servers("llama2")
print("Available servers:", servers)

# Get specific server information
server_info = client.get_server_info("llama2", "http://example.com:11434")
print("Server info:", server_info)
```

## Chat Examples

### Basic Chat
```python
# Simple chat with default parameters
response = client.chat(
    model_name="llama2",
    prompt="What is the capital of France?"
)
print("Response:", response)

# Chat with custom parameters
response = client.chat(
    model_name="llama2",
    prompt="Write a short poem about programming",
    temperature=0.8,
    top_p=0.95,
    num_predict=256
)
print("Response:", response)
```

### Streaming Chat
```python
# Stream the response
for chunk in client.stream_chat(
    model_name="llama2",
    prompt="Tell me a story about a robot",
    temperature=0.7
):
    print(chunk, end="", flush=True)
```

### Advanced API Request
```python
# Generate a custom API request
request = client.generate_api_request(
    model_name="llama2",
    prompt="Explain quantum computing",
    temperature=0.7,
    top_p=0.9,
    num_predict=512,
    repeat_penalty=1.1,
    stop=["Human:", "Assistant:"]
)
print("API Request:", request)
```

## Error Handling
```python
try:
    # Try to get info for non-existent model
    model_info = client.get_model_info("non_existent_model")
except ValueError as e:
    print(f"Error: {e}")

try:
    # Try to chat with a model that has no available servers
    response = client.chat("unavailable_model", "Hello")
except RuntimeError as e:
    print(f"Error: {e}")
```

## Complete Example
```python
from ollamafreeapi import OllamaFreeAPI

def main():
    # Initialize client
    client = OllamaFreeAPI()
    
    # List available models
    print("Available families:", client.list_families())
    
    # Choose a model
    model_name = "llama2"
    
    # Get model information
    try:
        model_info = client.get_model_info(model_name)
        print(f"\nModel {model_name} info:", model_info)
        
        # Get available servers
        servers = client.get_model_servers(model_name)
        print(f"\nAvailable servers for {model_name}:", servers)
        
        # Chat with the model
        prompt = "What are the three laws of robotics?"
        print(f"\nAsking: {prompt}")
        
        response = client.chat(
            model_name=model_name,
            prompt=prompt,
            temperature=0.7,
            num_predict=256
        )
        print(f"\nResponse: {response}")
        
    except (ValueError, RuntimeError) as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
```

These examples demonstrate the main features and common use cases of the OllamaFreeAPI client. Remember to handle exceptions appropriately in production code and adjust the parameters according to your specific needs.

Note: The actual model names and server URLs in these examples are placeholders. You should use the actual model names and server URLs available in your environment. 