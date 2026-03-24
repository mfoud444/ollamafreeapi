import os
import json
import urllib.request
from urllib.error import URLError, HTTPError
import socket
from concurrent.futures import ThreadPoolExecutor

# Make the paths relative to where the script is run, but with fallback for safety
base_dir = r"d:\my\ollamafreeapi"
input_file = os.path.join(base_dir, "valid_models.txt")
output_file = os.path.join(base_dir, "fetched_models.json")

def fetch_models(url):
    url = url.strip()
    if not url:
        return None
        
    try:
        # Send a GET request with a 10-second timeout
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode('utf-8'))
                print(f"[SUCCESS] Fetched models from {url}")
                # We save both the source URL and the array of models it provides
                return {"source_url": url, "models": data.get("models", [])}
            else:
                print(f"[FAILED] {url} - Status: {response.getcode()}")
    except HTTPError as e:
        print(f"[FAILED] {url} - HTTP Error: {e.code}")
    except URLError as e:
        print(f"[ERROR] {url} - URL Error: {e.reason}")
    except socket.timeout:
        print(f"[ERROR] {url} - Timeout")
    except Exception as e:
        print(f"[ERROR] {url} - {str(e)}")
    return None

def main():
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    urls = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    urls.append(line)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
                
    print(f"Found {len(urls)} URLs in {input_file}. Fetching models...")
    
    results = []
    
    # Use ThreadPoolExecutor to check URLs concurrently (much faster than sequentially)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = executor.map(fetch_models, urls)
        for res in futures:
            if res:
                results.append(res)
                
    try:
        # Save as a consolidated JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\nFinished! Saved data from {len(results)} endpoints to {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()
