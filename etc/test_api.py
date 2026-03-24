from ollamafreeapi.client import OllamaFreeAPI

def main():
    print("Initializing OllamaFreeAPI...")
    client = OllamaFreeAPI()
    
    models = client.list_models()
    print(f"\nSuccessfully loaded {len(models)} models from the new JSON files!")
    
    if models:
        test_model = models[0] # Just pick the first available one to test
        print(f"\nAvailable families: {client.list_families()}")
        print(f"Testing chat with model: '{test_model}'...")
        
        try:
            # Check servers to prove it knows where to route
            servers = client.get_model_servers(test_model)
            print(f"Found {len(servers)} servers routing for {test_model}")
            
            print(f"\nSending test prompt: 'What is 2+2? Answer in one word.'")
            response = client.chat(
                prompt="What is 2+2? Answer in one word.",
                model=test_model
            )
            print(f"\nResponse received:\n{response}")
            print("\n✅ API IS WORKING PERFECTLY WITH THE NEW UPDATE!")
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
    else:
        print("\n❌ No models were loaded. Something might be wrong with the JSON files.")

if __name__ == "__main__":
    main()
