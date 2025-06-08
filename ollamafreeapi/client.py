import json
import random
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from ollama import Client

class OllamaFreeAPI:
    """
    A client for interacting with LLMs served via Ollama.
    Uses JSON filenames as the only source of family names.
    """
    
    def __init__(self) -> None:
        """Initialize the client and load model data."""
        self._models_data: Dict[str, List[Dict[str, Any]]] = self._load_models_data()
        self._families: Dict[str, List[str]] = self._extract_families()
        self._client: Optional[Client] = None

    @property
    def client(self) -> Client:
        """Lazy-loaded Ollama client."""
        if self._client is None:
            self._client = Client()
        return self._client

    def _load_models_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load model data from JSON files in the ollama_json directory.
        Models are sorted by size and digest/perf_response_text fields are removed.
        
        Returns:
            Dictionary mapping family names (from filenames) to lists of model data.
        """
        models_data: Dict[str, List[Dict[str, Any]]] = {}
        package_dir = Path(__file__).parent
        json_dir = package_dir / "ollama_json"
        
        for json_file in json_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    family_name = json_file.stem.lower()  # Get family name from filename only
                    
                    models = self._extract_models_from_data(data)
                    if models:
                        # Remove digest and perf_response_text fields
                        for model in models:
                            if isinstance(model, dict):
                                model.pop('digest', None)
                                model.pop('perf_response_text', None)
                        
                        # Sort models by size (largest first)
                        models.sort(key=lambda x: int(x.get('size', 0)) if isinstance(x.get('size'), (int, str)) else 0, reverse=True)
                        models_data[family_name] = models

            except (json.JSONDecodeError, OSError) as e:
                print(f"Error loading {json_file.name}: {str(e)}")
                continue
        
        return models_data

    def _extract_models_from_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract models list from different possible JSON structures."""
        if isinstance(data, list):
            return data
        if 'props' in data and 'pageProps' in data['props']:
            return data['props']['pageProps'].get('models', [])
        return data.get('models', [])

    def _extract_families(self) -> Dict[str, List[str]]:
        """
        Extract model families using ONLY the JSON filenames as family names.
        
        Returns:
            Dictionary mapping family names to lists of model names.
        """
        families: Dict[str, List[str]] = {}
        
        for family_name, models in self._models_data.items():
            model_names = []
            for model in models:
                if not isinstance(model, dict):
                    continue
                
                model_name = self._get_model_name(model)
                if model_name:
                    model_names.append(model_name)
            
            if model_names:
                families[family_name] = model_names
        
        return families

    def _get_model_name(self, model: Dict[str, Any]) -> Optional[str]:
        """Extract model name from model data using multiple possible fields."""
        return model.get('model_name') or model.get('model') or model.get('name')

    def list_families(self) -> List[str]:
        """
        List all available model families (from JSON filenames only).
        
        Returns:
            List of family names.
        """
        return list(self._families.keys())

    def list_models(self, family: Optional[str] = None) -> List[str]:
        """
        List all models, optionally filtered by family.
        
        Args:
            family: Filter models by family name (case insensitive)
            
        Returns:
            List of model names.
        """
        if family is None:
            return [model for models in self._families.values() for model in models]
        
        return self._families.get(family.lower(), [])

    def get_model_info(self, model: str) -> Dict:
        """Get full metadata for a specific model"""
        for models in self._models_data.values():
            for model_data in models:
                if isinstance(model_data, dict):
                    if model_data.get('model_name') == model or model_data.get('model') == model:
                        return model_data
        raise ValueError(f"Model '{model}' not found")
    
    def get_model_servers(self, model: str) -> List[Dict]:
        """
        Get all servers hosting a specific model
        
        Args:
            model: Name of the model
            
        Returns:
            List of server dictionaries containing url and metadata
        """
        servers = []
        for models in self._models_data.values():
            for model_data in models:
                if model_data['model_name'] == model:
                    server_info = {
                        'url': model_data['ip_port'],
                        'location': {
                            'city': model_data.get('ip_city_name_en'),
                            'country': model_data.get('ip_country_name_en'),
                            'continent': model_data.get('ip_continent_name_en')
                        },
                        'organization': model_data.get('ip_organization'),
                        'performance': {
                            'tokens_per_second': model_data.get('perf_tokens_per_second'),
                            'last_tested': model_data.get('perf_last_tested')
                        }
                    }
                    servers.append(server_info)
        return servers
    
    def get_server_info(self, model: str, server_url: Optional[str] = None) -> Dict:
        """
        Get information about a specific server hosting a model
        
        Args:
            model: Name of the model
            server_url: Specific server URL (if None, returns first available)
            
        Returns:
            Dictionary with server information
            
        Raises:
            ValueError: If model or server not found
        """
        servers = self.get_model_servers(model)
        if not servers:
            raise ValueError(f"No servers found for model '{model}'")
        
        if server_url:
            for server in servers:
                if server['url'] == server_url:
                    return server
            raise ValueError(f"Server '{server_url}' not found for model '{model}'")
        return servers[0]
    
    def generate_api_request(self, model: str, prompt: str, **kwargs) -> Dict:
        """
        Generate the JSON payload for an API request
        
        Args:
            model: Name of the model to use
            prompt: The input prompt
            **kwargs: Additional model parameters (temperature, top_p, etc.)
            
        Returns:
            Dictionary representing the API request payload
        """
        model_info = self.get_model_info(model)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": kwargs.get('temperature', 0.7),
                "top_p": kwargs.get('top_p', 0.9),
                "stop": kwargs.get('stop', []),
                "num_predict": kwargs.get('num_predict', 128)
            }
        }
        
        # Add any additional supported options
        supported_options = ['repeat_penalty', 'seed', 'tfs_z', 'mirostat']
        for opt in supported_options:
            if opt in kwargs:
                payload['options'][opt] = kwargs[opt]
                
        return payload
    
    def chat(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """
        Chat with a model using automatic server selection
        
        Args:
            prompt: The input prompt
            model: Name of the model to use (optional, will select random if not provided)
            **kwargs: Additional model parameters
            
        Returns:
            The generated response text
            
        Raises:
            RuntimeError: If no working server is found
        """
        if model is None:
            # Get all available models and select one randomly
            all_models = self.list_models()
            if not all_models:
                raise RuntimeError("No models available")
            model = random.choice(all_models)
            print(f"Selected model: {model}")
            
        servers = self.get_model_servers(model)
        if not servers:
            raise RuntimeError(f"No servers available for model '{model}'")
        
        # Try servers in random order (could be enhanced with priority/performance)
        random.shuffle(servers)
        
        last_error = None
        for server in servers:
            try:
                client = Client(host=server['url'])
                request = self.generate_api_request(model, prompt, **kwargs)
                response = client.generate(**request)
                return response['response']
            except Exception as e:
                last_error = e
                continue
        
        raise RuntimeError(f"All servers failed for model '{model}'. Last error: {str(last_error)}")
    
    def stream_chat(self, prompt: str, model: Optional[str] = None, **kwargs):
        """
        Stream chat response from a model
        
        Args:
            prompt: The input prompt
            model: Name of the model to use (optional, will select random if not provided)
            **kwargs: Additional model parameters
            
        Yields:
            Response chunks as they are generated
            
        Raises:
            RuntimeError: If no working server is found
        """
        if model is None:
            # Get all available models and select one randomly
            all_models = self.list_models()
            if not all_models:
                raise RuntimeError("No models available")
            model = random.choice(all_models)
            print(f"Selected model: {model}")
            
        servers = self.get_model_servers(model)
        if not servers:
            raise RuntimeError(f"No servers available for model '{model}'")
        
        random.shuffle(servers)
        last_error = None
        
        for server in servers:
            try:
                client = Client(host=server['url'])
                request = self.generate_api_request(model, prompt, **kwargs)
                request['stream'] = True
                
                for chunk in client.generate(**request):
                    yield chunk['response']
                return
            except Exception as e:
                last_error = e
                continue
        
        raise RuntimeError(f"All servers failed for model '{model}'. Last error: {str(last_error)}")
    
    def get_llm_params(self, model: Optional[str] = None) -> Dict[str, str]:
        """
        Get model and server parameters for OllamaLLM
        
        Args:
            model: Name of the model to use (optional, will select random if not provided)
            
        Returns:
            Dictionary containing model and base_url parameters for OllamaLLM
            
        Raises:
            RuntimeError: If no models or servers are available
            ValueError: If specified model is not found
        """
        if model is None:
            # Get random model
            all_models = self.list_models()
            if not all_models:
                raise RuntimeError("No models available")
            model = random.choice(all_models)
            print(f"Selected model: {model}")
        else:
            # Verify the specified model exists
            if model not in self.list_models():
                raise ValueError(f"Model '{model}' not found")
            
        servers = self.get_model_servers(model)
        if not servers:
            raise RuntimeError(f"No servers available for model '{model}'")
            
        server = random.choice(servers)
        print(f"Selected server: {server['url']}")
        
        return {
            "model": model,
            "base_url": server['url']
        }
        
    def get_random_llm_params(self) -> Dict[str, str]:
        """
        Get random model and server parameters for OllamaLLM
        
        Returns:
            Dictionary containing model and base_url parameters for OllamaLLM
            
        Raises:
            RuntimeError: If no models or servers are available
        """
        return self.get_llm_params()
    
    
    
    