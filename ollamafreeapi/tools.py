import requests
from bs4 import BeautifulSoup
import json
import os

# List of model names
models = ["llama", "deepseek", "gemma", "qwen", "mistral", "others"]

output_folder = "ollama_json"
os.makedirs(output_folder, exist_ok=True)

# Loop through each model
for model_name in models:
    print(f"Fetching model: {model_name}")
    
    url = f"https://ollama.zone/model/{model_name}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        script_tag = soup.find("script", id="__NEXT_DATA__")
        
        if not script_tag:
            print(f"❌ Could not find JSON script for model '{model_name}'")
            continue

        json_data = json.loads(script_tag.string)

       
        output_file = os.path.join(output_folder, f"{model_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        
        print(f"✅ Saved to {output_file}")
    
    except Exception as e:
        print(f"⚠️ Error processing model '{model_name}': {e}")
