from ollamafreeapi.client import OllamaFreeAPI

def main():
    print("Initializing OllamaFreeAPI for Stream Testing...")
    client = OllamaFreeAPI()
    
    models = client.list_models()
    if not models:
        print("❌ No models available.")
        return
        
    # Pick the first available one (which worked in test_api.py)
    test_model = models[0]
    
    print(f"Available models loaded: {len(models)}")
    print(f"Using model: {test_model}")
    print("Sending streaming prompt: 'Write a short Python coding snippet to calculate Fibonacci numbers.'\n")
    print("Response stream:")
    print("-" * 50)
    
    try:
        # Stream the code generation
        for chunk in client.stream_chat(prompt='Write a short Python coding snippet to calculate Fibonacci numbers.', model=test_model):
            print(chunk, end='', flush=True)
            
        print("\n" + "-" * 50)
        print("\n✅ STREAMING TEST SUCCESSFUL!")
    except Exception as e:
        print(f"\n\n❌ Streaming test failed: {e}")

if __name__ == "__main__":
    main()
