# OllamaFreeAPI Client Documentation

## Overview
The `OllamaFreeAPI` class provides a client interface for interacting with LLMs served via Ollama. It uses JSON filenames as the source of family names and provides methods for model management and chat interactions.

## Class Initialization
```python
def __init__(self) -> None:
    """Initialize the client and load model data."""
```

## Properties

### client
```python
@property
def client(self) -> Client:
    """Lazy-loaded Ollama client."""
```
Returns an instance of the Ollama Client. The client is initialized only when first accessed.

## Model Management Methods

### list_families
```python
def list_families(self) -> List[str]:
    """
    List all available model families (from JSON filenames only).
    
    Returns:
        List of family names.
    """
```

### list_models
```python
def list_models(self, family: Optional[str] = None) -> List[str]:
    """
    List all models, optionally filtered by family.
    
    Args:
        family: Filter models by family name (case insensitive)
        
    Returns:
        List of model names.
    """
```

### get_model_info
```python
def get_model_info(self, model_name: str) -> Dict:
    """
    Get full metadata for a specific model.
    
    Args:
        model_name: Name of the model to retrieve information for
        
    Returns:
        Dictionary containing model metadata
        
    Raises:
        ValueError: If model is not found
    """
```

### get_model_servers
```python
def get_model_servers(self, model_name: str) -> List[Dict]:
    """
    Get all servers hosting a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        List of server dictionaries containing:
        - url: Server URL
        - location: Dictionary with city, country, and continent
        - organization: Organization name
        - performance: Dictionary with tokens_per_second and last_tested
    """
```

### get_server_info
```python
def get_server_info(self, model_name: str, server_url: Optional[str] = None) -> Dict:
    """
    Get information about a specific server hosting a model.
    
    Args:
        model_name: Name of the model
        server_url: Specific server URL (if None, returns first available)
        
    Returns:
        Dictionary with server information
        
    Raises:
        ValueError: If model or server not found
    """
```

## Chat Methods

### generate_api_request
```python
def generate_api_request(self, model_name: str, prompt: str, **kwargs) -> Dict:
    """
    Generate the JSON payload for an API request.
    
    Args:
        model_name: Name of the model to use
        prompt: The input prompt
        **kwargs: Additional model parameters:
            - temperature (default: 0.7)
            - top_p (default: 0.9)
            - stop (default: [])
            - num_predict (default: 128)
            - repeat_penalty (optional)
            - seed (optional)
            - tfs_z (optional)
            - mirostat (optional)
            
    Returns:
        Dictionary representing the API request payload
    """
```

### chat
```python
def chat(self, model_name: str, prompt: str, **kwargs) -> str:
    """
    Chat with a model using automatic server selection.
    
    Args:
        model_name: Name of the model to use
        prompt: The input prompt
        **kwargs: Additional model parameters (same as generate_api_request)
        
    Returns:
        The generated response text
        
    Raises:
        RuntimeError: If no working server is found
    """
```

### stream_chat
```python
def stream_chat(self, model_name: str, prompt: str, **kwargs):
    """
    Stream chat response from a model.
    
    Args:
        model_name: Name of the model to use
        prompt: The input prompt
        **kwargs: Additional model parameters (same as generate_api_request)
        
    Yields:
        Response chunks as they are generated
        
    Raises:
        RuntimeError: If no working server is found
    """
```

## Private Methods

### _load_models_data
```python
def _load_models_data(self) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load model data from JSON files in the ollama_json directory.
    Models are sorted by size and digest/perf_response_text fields are removed.
    
    Returns:
        Dictionary mapping family names to lists of model data.
    """
```

### _extract_models_from_data
```python
def _extract_models_from_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract models list from different possible JSON structures.
    
    Args:
        data: JSON data to extract models from
        
    Returns:
        List of model dictionaries
    """
```

### _extract_families
```python
def _extract_families(self) -> Dict[str, List[str]]:
    """
    Extract model families using ONLY the JSON filenames as family names.
    
    Returns:
        Dictionary mapping family names to lists of model names.
    """
```

### _get_model_name
```python
def _get_model_name(self, model: Dict[str, Any]) -> Optional[str]:
    """
    Extract model name from model data using multiple possible fields.
    
    Args:
        model: Model data dictionary
        
    Returns:
        Model name if found, None otherwise
    """
```
```

This documentation provides a comprehensive overview of all methods in the OllamaFreeAPI client, including their parameters, return values, and any exceptions they might raise. The documentation is organized into sections for better readability and includes both public and private methods.

You can save this content in a new file at `docs/client.md`. The markdown format makes it easy to read both in raw form and when rendered by markdown viewers.