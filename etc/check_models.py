import os
import json
import urllib.request
from urllib.error import URLError, HTTPError
import socket
from concurrent.futures import ThreadPoolExecutor

# Make the paths relative to where the script is run, but with fallback for safety
base_dir = r"d:\my\ollamafreeapi"
folder_path = os.path.join(base_dir, "ollamafreeapi", "ollama_json")
output_file = os.path.join(base_dir, "valid_models.txt")

def check_url(base_url):
    # Ensure URL has http scheme
    if not base_url.startswith("http"):
        base_url = "http://" + base_url
        
    # Append /api/tags
    test_url = base_url.rstrip("/") + "/api/tags"
    
    try:
        # Send a GET request with a 5-second timeout
        req = urllib.request.Request(test_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.getcode() == 200:
                print(f"[VALID] {test_url}")
                return test_url
            else:
                print(f"[INVALID] {test_url} - Status: {response.getcode()}")
    except HTTPError as e:
        print(f"[INVALID] {test_url} - HTTP Error: {e.code}")
    except URLError as e:
        print(f"[ERROR] {test_url} - URL Error: {e.reason}")
    except socket.timeout:
        print(f"[ERROR] {test_url} - Timeout")
    except Exception as e:
        print(f"[ERROR] {test_url} - {str(e)}")
    return None

def main():
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}")
        print("Please ensure the script is pointing to the correct 'ollama_json' directory.")
        return

    unique_urls = set()
    
    print(f"Reading JSON files from {folder_path}...")
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Extract the models array from props -> pageProps -> models
                    models = data.get("props", {}).get("pageProps", {}).get("models", [])
                    for model in models:
                        ip_port = model.get("ip_port")
                        if ip_port:
                            unique_urls.add(ip_port)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
    print(f"Found {len(unique_urls)} unique IPs. Testing connections...")
    
    valid_urls = []
    
    # Use ThreadPoolExecutor to check URLs concurrently (much faster than sequentially)
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(check_url, unique_urls)
        for res in results:
            if res:
                valid_urls.append(res)
                
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for url in valid_urls:
                f.write(url + "\n")
        print(f"\nFinished! Saved {len(valid_urls)} valid URLs to {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()
