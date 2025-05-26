from ollamafreeapi import OllamaFreeAPI

# Initialize the client
client = OllamaFreeAPI()

# List available model families
print(client.list_families())

# List all models in the 'llama' family
print(client.list_models(family='llama'))

# Get info about a specific model
print(client.get_model_info('deepseek-r1:7b'))

# Chat with a model
response = client.chat('deepseek-r1:7b', 'Hello! How are you?', temperature=0.8)
print(response)


# Stream responses
for chunk in client.stream_chat('deepseek-r1:7b', 'Tell me a story:'):
    print(chunk, end='', flush=True)
    
    