import json
import random
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from ollama import Client

class OllamaFreeAPI:
    def __init__(self):
        self._models_data = self._load_models_data()
        self._families = self._extract_families()
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            self._client = Client()
        return self._client
    
    def _load_models_data(self) -> Dict[str, List[Dict]]:
        """Load all model data from JSON files with more flexible parsing"""
        models_data = {}
        package_dir = Path(__file__).parent
        json_dir = package_dir / "ollama_json"
        
        for json_file in json_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Flexible model type extraction
                model_type = os.path.splitext(json_file.name)[0]  # Use filename as fallback
                if 'props' in data and 'pageProps' in data['props']:
                    if 'modelType' in data['props']['pageProps']:
                        model_type = data['props']['pageProps']['modelType']
                    elif 'models' in data['props']['pageProps']:
                        # If we have models but no modelType, use filename
                        pass
                
                if 'props' in data and 'pageProps' in data['props'] and 'models' in data['props']['pageProps']:
                    models_data[model_type] = data['props']['pageProps']['models']
                else:
                    # Fallback: try to find models at root level
                    if 'models' in data:
                        models_data[model_type] = data['models']
        
        return models_data
    
    def _extract_families(self) -> Dict[str, List[str]]:
        """Extract model families and their models with more flexible family detection"""
        families = {}
        for model_type, models in self._models_data.items():
            family_models = {}
            for model in models:
                # More flexible family detection
                family = model.get('family', 
                                 model.get('family_name',
                                          model_type.lower()))
                model_name = model.get('model_name', 
                                     model.get('name', 
                                              model.get('model', '')))
                
                if not model_name:
                    continue
                
                if family not in family_models:
                    family_models[family] = []
                family_models[family].append(model_name)
            families[model_type] = family_models
        return families
    

    
    def list_families(self) -> List[str]:
        """Return all available model families"""
        return list(self._families.keys())
    
    def list_models(self, family: Optional[str] = None) -> List[str]:
        """
        List all models, optionally filtered by family
        
        Args:
            family: Filter models by family name
            
        Returns:
            List of model names
        """
        if family:
            return [model for models in self._families.values() 
                    for fam, mods in models.items() 
                    if fam.lower() == family.lower() 
                    for model in mods]
        return [model for models in self._families.values() 
                for fam_models in models.values() 
                for model in fam_models]
    
    def get_model_info(self, model_name: str) -> Dict:
        """
        Get full metadata for a specific model
        
        Args:
            model_name: Name of the model to get info for
            
        Returns:
            Dictionary containing model metadata
            
        Raises:
            ValueError: If model is not found
        """
        for models in self._models_data.values():
            for model in models:
                if model['model_name'] == model_name:
                    return model
        raise ValueError(f"Model '{model_name}' not found")
    
    def get_model_servers(self, model_name: str) -> List[Dict]:
        """
        Get all servers hosting a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of server dictionaries containing url and metadata
        """
        servers = []
        for models in self._models_data.values():
            for model in models:
                if model['model_name'] == model_name:
                    server_info = {
                        'url': model['ip_port'],
                        'location': {
                            'city': model.get('ip_city_name_en'),
                            'country': model.get('ip_country_name_en'),
                            'continent': model.get('ip_continent_name_en')
                        },
                        'organization': model.get('ip_organization'),
                        'performance': {
                            'tokens_per_second': model.get('perf_tokens_per_second'),
                            'last_tested': model.get('perf_last_tested')
                        }
                    }
                    servers.append(server_info)
        return servers
    
    def get_server_info(self, model_name: str, server_url: Optional[str] = None) -> Dict:
        """
        Get information about a specific server hosting a model
        
        Args:
            model_name: Name of the model
            server_url: Specific server URL (if None, returns first available)
            
        Returns:
            Dictionary with server information
            
        Raises:
            ValueError: If model or server not found
        """
        servers = self.get_model_servers(model_name)
        if not servers:
            raise ValueError(f"No servers found for model '{model_name}'")
        
        if server_url:
            for server in servers:
                if server['url'] == server_url:
                    return server
            raise ValueError(f"Server '{server_url}' not found for model '{model_name}'")
        return servers[0]
    
    def generate_api_request(self, model_name: str, prompt: str, **kwargs) -> Dict:
        """
        Generate the JSON payload for an API request
        
        Args:
            model_name: Name of the model to use
            prompt: The input prompt
            **kwargs: Additional model parameters (temperature, top_p, etc.)
            
        Returns:
            Dictionary representing the API request payload
        """
        model_info = self.get_model_info(model_name)
        
        payload = {
            "model": model_name,
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
    
    def chat(self, model_name: str, prompt: str, **kwargs) -> str:
        """
        Chat with a model using automatic server selection
        
        Args:
            model_name: Name of the model to use
            prompt: The input prompt
            **kwargs: Additional model parameters
            
        Returns:
            The generated response text
            
        Raises:
            RuntimeError: If no working server is found
        """
        servers = self.get_model_servers(model_name)
        if not servers:
            raise RuntimeError(f"No servers available for model '{model_name}'")
        
        # Try servers in random order (could be enhanced with priority/performance)
        random.shuffle(servers)
        
        last_error = None
        for server in servers:
            try:
                client = Client(host=server['url'])
                request = self.generate_api_request(model_name, prompt, **kwargs)
                response = client.generate(**request)
                return response['response']
            except Exception as e:
                last_error = e
                continue
        
        raise RuntimeError(f"All servers failed for model '{model_name}'. Last error: {str(last_error)}")
    
    def stream_chat(self, model_name: str, prompt: str, **kwargs):
        """
        Stream chat response from a model
        
        Args:
            model_name: Name of the model to use
            prompt: The input prompt
            **kwargs: Additional model parameters
            
        Yields:
            Response chunks as they are generated
            
        Raises:
            RuntimeError: If no working server is found
        """
        servers = self.get_model_servers(model_name)
        if not servers:
            raise RuntimeError(f"No servers available for model '{model_name}'")
        
        random.shuffle(servers)
        last_error = None
        
        for server in servers:
            try:
                client = Client(host=server['url'])
                request = self.generate_api_request(model_name, prompt, **kwargs)
                request['stream'] = True
                
                for chunk in client.generate(**request):
                    yield chunk['response']
                return
            except Exception as e:
                last_error = e
                continue
        
        raise RuntimeError(f"All servers failed for model '{model_name}'. Last error: {str(last_error)}")