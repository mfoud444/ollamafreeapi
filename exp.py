from ollamafreeapi import OllamaFreeAPI
client = OllamaFreeAPI()
print(client.list_families())
print(client.list_models(family='others'))

# Get info about a specific model
# print(client.get_model_info('deepseek-r1:7b'))

# Chat with a model
# response = client.chat('deepseek-r1:7b', 'Hello! How are you?', temperature=0.8)
# print(response) others 


# # Stream responses
for chunk in client.stream_chat('deepseek-r1:7b', 'Tell me a story:'):
    print(chunk, end='', flush=True)
    
    