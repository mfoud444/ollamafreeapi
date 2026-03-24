import os
import json
import shutil
from collections import defaultdict

base_dir = r"d:\my\ollamafreeapi"
resolved_file = os.path.join(base_dir, "resolved_valid_models.json")
json_dir = os.path.join(base_dir, "ollamafreeapi", "ollama_json")
backup_dir = os.path.join(base_dir, "ollamafreeapi", "ollama_json_backup")

def main():
    if not os.path.exists(resolved_file):
        print(f"Error: {resolved_file} not found. Please resolve models first.")
        return

    with open(resolved_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    models = data.get("props", {}).get("pageProps", {}).get("models", [])
    if not models:
        print("No models found in the resolved file.")
        return

    # Backup existing JSONs
    if os.path.exists(json_dir):
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        print(f"Backing up existing JSONs to {backup_dir}...")
        for file in os.listdir(json_dir):
            if file.endswith(".json"):
                src_file = os.path.join(json_dir, file)
                dst_file = os.path.join(backup_dir, file)
                shutil.move(src_file, dst_file)
    else:
        os.makedirs(json_dir)

    # Group models by family
    grouped_models = defaultdict(list)
    for model in models:
        family = model.get("family", "")
        # Handle cases where family is not provided or malformed
        if not family or family == "unknown":
            # try to infer from model name
            name = model.get("model_name", "").lower()
            if "mistral" in name:
                family = "mistral"
            elif "llama" in name:
                family = "llama"
            elif "gemma" in name:
                family = "gemma"
            elif "qwen" in name:
                family = "qwen"
            elif "deepseek" in name:
                family = "deepseek"
            else:
                family = "others"
                
        # Some families like qwen2 can be grouped into qwen
        if family.startswith("qwen"): family = "qwen"
        if family.startswith("gemma"): family = "gemma"
        if family.startswith("llama"): family = "llama"
        if family.startswith("mistral"): family = "mistral"
        
        grouped_models[family].append(model)
        
    # Write new JSONs
    print(f"Applying valid models back to {json_dir}...")
    for family, f_models in grouped_models.items():
        family_file = os.path.join(json_dir, f"{family}.json")
        outer_payload = {
            "props": {
                "pageProps": {
                    "models": f_models
                }
            }
        }
        with open(family_file, 'w', encoding='utf-8') as f:
            json.dump(outer_payload, f, indent=4, ensure_ascii=False)
        print(f"Created {family}.json with {len(f_models)} models.")
        
    print("\nSuccess! The Ollama Free API will now use only the verified working models.")

if __name__ == "__main__":
    main()
