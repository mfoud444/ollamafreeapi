import os
import json
import uuid

# Set up paths relative to the project directory
base_dir = r"d:\my\ollamafreeapi"
fetched_file = os.path.join(base_dir, "fetched_models.json")
old_json_folder = os.path.join(base_dir, "ollamafreeapi", "ollama_json")
output_file = os.path.join(base_dir, "resolved_valid_models.json")

def main():
    if not os.path.exists(fetched_file):
        print(f"Fetched models file not found at {fetched_file}")
        return
        
    # 1. Gather old IP metadata (location, connection info, etc.) from existing JSON files
    ip_metadata = {}
    if os.path.exists(old_json_folder):
        print(f"Reading existing metadata from {old_json_folder}...")
        for filename in os.listdir(old_json_folder):
            if filename.endswith(".json"):
                path = os.path.join(old_json_folder, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        models = data.get("props", {}).get("pageProps", {}).get("models", [])
                        for m in models:
                            ip_port = m.get("ip_port", "")
                            
                            # Standardize 'ip_port' strings by ensuring "http://" prefix
                            if ip_port and not ip_port.startswith("http"):
                                ip_port = "http://" + ip_port
                                
                            if ip_port and ip_port not in ip_metadata:
                                # Save everything that starts with ip_ or perf_ (or other keys as needed)
                                ip_metadata[ip_port] = {
                                    k: v for k, v in m.items() 
                                    if k.startswith("ip_") or k.startswith("perf_")
                                }
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
    else:
        print(f"Warning: Original ollama_json folder not found. Some metadata may be missing.")

    # 2. Read the newly fetched actual model data
    print("Loading valid models...")
    with open(fetched_file, 'r', encoding='utf-8') as f:
        fetched_data = json.load(f)
        
    resolved_models = []
    
    # 3. Assemble the models array
    for entry in fetched_data:
        source_url = entry.get("source_url", "")
        # Remove '/api/tags' to get back to the 'ip_port' base URL
        base_ip_port = source_url.replace("/api/tags", "")
        
        # Grab metadata for this IP, if available
        meta = ip_metadata.get(base_ip_port, {})
        
        for m in entry.get("models", []):
            # Create the data payload following the original structure
            new_model = {
                "id": str(uuid.uuid4()), # Generate a unique ID
                "ip_port": base_ip_port,
                "model_name": m.get("name"),
                "model": m.get("model"),
                "modified_at": m.get("modified_at"),
                "size": str(m.get("size")),
                "digest": m.get("digest"),
                "parent_model": m.get("details", {}).get("parent_model", ""),
                "format": m.get("details", {}).get("format", ""),
                "family": m.get("details", {}).get("family", ""),
                "parameter_size": m.get("details", {}).get("parameter_size", ""),
                "quantization_level": m.get("details", {}).get("quantization_level", ""),
                "date_added": meta.get("date_added", m.get("modified_at"))
            }
            
            # Map all the specific 'meta' fields like geo, perf_, etc.
            for k, v in meta.items():
                if k not in new_model:
                    new_model[k] = v
                    
            resolved_models.append(new_model)
            
    # 4. Wrap everything in the required outer JSON structure
    final_output = {
        "props": {
            "pageProps": {
                "models": resolved_models
            }
        }
    }
    
    # 5. Save the resolved file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"\nSuccessfully resolved {len(resolved_models)} individual models from {len(fetched_data)} valid endpoints.")
    print(f"Results have been saved to {output_file} in the original application format.")

if __name__ == "__main__":
    main()
