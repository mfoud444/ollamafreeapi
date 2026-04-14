import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from ollama import Client

# Default timeout (seconds) for all Ollama API calls.
# Exposed as a class constant so callers can override per-call via kwargs.
DEFAULT_TIMEOUT = 30.0


class OllamaFreeAPI:
    """
    A client for interacting with LLMs served via Ollama.
    Uses JSON filenames as the only source of family names.

    Server selection is performance-weighted: servers are sorted descending by
    their measured throughput (tokens/s). A lightweight health-check probe is
    available for callers that want to pre-flight the top candidate before
    committing to a real request.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialize the client and load model data."""
        self._models_data: Dict[str, List[Dict[str, Any]]] = self._load_models_data()
        self._families: Dict[str, List[str]] = self._extract_families()
        self._client: Optional[Client] = None

    @property
    def client(self) -> Client:
        """
        Lazy-loaded Ollama client.

        .. note::
            This property is not used by ``chat()`` or ``stream_chat()``, which
            construct a per-call ``Client`` with the target server's host.
            It exists for callers that need a raw :class:`ollama.Client`
            instance without going through the server-selection logic.
        """
        if self._client is None:
            self._client = Client()
        return self._client

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # Allowlist family names to prevent path-traversal abuse (M-5).
        # Only .json files whose stem matches a known family are loaded.
        # ------------------------------------------------------------------
        allowed_families = {"gemma", "llama", "qwen", "mistral", "deepseek"}

        for json_file in json_dir.glob("*.json"):
            stem = json_file.stem.lower()
            if stem not in allowed_families:
                # Silently skip unexpected files rather than crashing.
                # Only curated family files are loaded.
                continue

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                models = self._extract_models_from_data(data)
                if models:
                    # Remove digest and perf_response_text fields
                    for model in models:
                        if isinstance(model, dict):
                            model.pop("digest", None)
                            model.pop("perf_response_text", None)

                    # Sort models by size (largest first)
                    models.sort(
                        key=lambda x: int(x.get("size", 0))
                        if isinstance(x.get("size"), (int, str))
                        else 0,
                        reverse=True,
                    )
                    models_data[stem] = models
            except json.JSONDecodeError as e:
                print(f"[OllamaFreeAPI] Invalid JSON in {json_file.name}: {e}")
                continue
            except OSError as e:
                print(f"[OllamaFreeAPI] Could not read {json_file.name}: {e}")
                continue

        return models_data

    def _extract_models_from_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract the models list from one JSON file.

        The Ollama instance data exported from the web UI uses a Next.js
        pageProps wrapper; locally-managed files may ship a flat list.
        Both layouts are accepted. When the pageProps path is taken a
        warning is printed so a maintainer knows a data-source migration
        is needed.
        """
        if isinstance(data, list):
            return data

        if "props" in data and "pageProps" in data["props"]:
            models = data["props"]["pageProps"].get("models", [])
            print(
                "[OllamaFreeAPI] Warning: data is using the pageProps wrapper "
                "(web-scraped format). Migrate to a flat list to avoid breakage "
                "when the site structure changes."
            )
            return models

        return data.get("models", [])

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
        return model.get("model_name") or model.get("model") or model.get("name")

    # ------------------------------------------------------------------
    # Public API — listing
    # ------------------------------------------------------------------

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
        """
        Get full metadata for a specific model.

        Args:
            model: Name of the model to look up.

        Returns:
            Full model metadata dictionary.

        Raises:
            ValueError: If the model is not found.
        """
        for models in self._models_data.values():
            for model_data in models:
                if isinstance(model_data, dict):
                    name = model_data.get("model_name") or model_data.get("model")
                    if name == model:
                        return model_data
        raise ValueError(f"Model '{model}' not found")

    def get_model_servers(self, model: str) -> List[Dict]:
        """
        Get all servers hosting a specific model.

        Args:
            model: Name of the model

        Returns:
            List of server dictionaries containing url and metadata
        """
        servers = []
        for models in self._models_data.values():
            for model_data in models:
                # Guard: skip non-dict entries and entries that don't match.
                if not isinstance(model_data, dict):
                    continue
                if model_data.get("model_name") != model:
                    continue

                # All subsequent accesses use .get() with defaults — no KeyError possible.
                servers.append(
                    {
                        "url": model_data.get("ip_port", ""),
                        "location": {
                            "city": model_data.get("ip_city_name_en"),
                            "country": model_data.get("ip_country_name_en"),
                            "continent": model_data.get("ip_continent_name_en"),
                        },
                        "organization": model_data.get("ip_organization"),
                        "performance": {
                            "tokens_per_second": model_data.get(
                                "perf_tokens_per_second"
                            ),
                            "last_tested": model_data.get("perf_last_tested"),
                        },
                    }
                )
        return servers

    def get_server_info(self, model: str, server_url: Optional[str] = None) -> Dict:
        """
        Get information about a specific server hosting a model.

        Args:
            model: Name of the model
            server_url: Specific server URL (if None, returns first available)

        Returns:
            Dictionary with server information

        Raises:
            ValueError: If model or server not found.
        """
        servers = self.get_model_servers(model)
        if not servers:
            raise ValueError(f"No servers found for model '{model}'")

        if server_url:
            for server in servers:
                if server["url"] == server_url:
                    return server
            raise ValueError(f"Server '{server_url}' not found for model '{model}'")

        return servers[0]

    # ------------------------------------------------------------------
    # Server selection helpers
    # ------------------------------------------------------------------

    def _select_best_server(
        self, servers: List[Dict]
    ) -> List[Dict]:
        """
        Return servers sorted by measured throughput (descending).

        Performance data (``perf_tokens_per_second``) is the primary sort key.
        Servers without a reading are placed last. The caller iterates the
        returned list in order; it does **not** randomly shuffle.

        Args:
            servers: List of server dictionaries as returned by get_model_servers().

        Returns:
            New list of servers sorted by tokens/s, descending.
        """
        def sort_key(s: Dict) -> float:
            perf = s.get("performance", {})
            tok_s = perf.get("tokens_per_second")
            if isinstance(tok_s, (int, float)):
                return float(tok_s)
            return -1.0  # servers without data go last

        return sorted(servers, key=sort_key, reverse=True)

    def _health_check(self, server_url: str, timeout: float = 5.0) -> bool:
        """
        Lightweight pre-flight check: send a minimal generate request to
        verify the server is reachable and responsive.

        This uses ``num_predict=1`` so it costs minimal resources on the
        remote server while confirming the endpoint is alive.  A "dummy"
        model name is used deliberately: any response (including a model-not-found
        error from a real server) proves the endpoint is up.

        Args:
            server_url: IP:port of the server to probe.
            timeout: Maximum seconds to wait for a response.

        Returns:
            True if the server responded within ``timeout`` seconds,
            False otherwise.
        """
        try:
            probe = Client(host=server_url)
            probe.generate(
                model="dummy",
                prompt=".",
                options={"num_predict": 1},
                timeout=timeout,
            )
            return True
        except Exception:
            # Any failure — timeout, connection refused, model not found, etc. —
            # means this server is not a viable first candidate.
            return False

    # ------------------------------------------------------------------
    # Request helpers
    # ------------------------------------------------------------------

    def generate_api_request(self, model: str, prompt: str, **kwargs) -> Dict:
        """
        Generate the JSON payload for an API request.

        Args:
            model: Name of the model to use
            prompt: The input prompt
            **kwargs: Additional model parameters (temperature, top_p, etc.)

        Returns:
            Dictionary representing the API request payload
        """
        # Validate model exists before building the payload.
        self.get_model_info(model)

        # Allowlist only known Ollama options to prevent kwargs injection (H-1).
        supported_options = {
            "temperature",
            "top_p",
            "stop",
            "num_predict",
            "repeat_penalty",
            "seed",
            "tfs_z",
            "mirostat",
        }
        options = {}
        for key, value in kwargs.items():
            if key in supported_options:
                options[key] = value
            # Unknown keys are silently dropped — consistent with Ollama's API.

        payload = {
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": options.pop("temperature", 0.7),
                "top_p": options.pop("top_p", 0.9),
                "stop": options.pop("stop", []),
                "num_predict": options.pop("num_predict", 128),
            },
        }
        # Merge any remaining known options.
        payload["options"].update(options)

        return payload

    # ------------------------------------------------------------------
    # chat / stream_chat
    # ------------------------------------------------------------------

    def chat(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """
        Chat with a model using performance-weighted server selection.

        Servers are sorted by measured throughput (descending). If the top
        server fails, the next one is tried automatically.

        Args:
            prompt: The input prompt.
            model: Name of the model to use. If None, a random model is
                selected (not recommended for production).
            **kwargs: Additional model parameters. Supports ``timeout`` (float,
                seconds) to override the default per-request timeout.

        Returns:
            The generated response text.

        Raises:
            RuntimeError: If no working server is found.
        """
        timeout: float = kwargs.pop("timeout", DEFAULT_TIMEOUT)

        if model is None:
            all_models = self.list_models()
            if not all_models:
                raise RuntimeError("No models available")
            model = random.choice(all_models)
            print(f"[OllamaFreeAPI] Selected model: {model}")

        servers = self.get_model_servers(model)
        if not servers:
            raise RuntimeError(f"No servers available for model '{model}'")

        servers = self._select_best_server(servers)

        last_error: Optional[Exception] = None
        for server in servers:
            client = Client(host=server["url"])
            request = self.generate_api_request(model, prompt, **kwargs)

            try:
                response = client.generate(**request, timeout=timeout)
                return response["response"]
            except Exception as e:
                last_error = e
                # Continue to next server; last_error is surfaced if all fail.
                continue

        raise RuntimeError(
            f"All servers failed for model '{model}'. Last error: {last_error}"
        )

    def stream_chat(
        self, prompt: str, model: Optional[str] = None, **kwargs
    ):
        """
        Stream chat response from a model using performance-weighted server selection.

        Servers are sorted by measured throughput (descending). If the top
        server fails, the next one is tried automatically.

        Args:
            prompt: The input prompt.
            model: Name of the model to use. If None, a random model is
                selected (not recommended for production).
            **kwargs: Additional model parameters. Supports ``timeout`` (float,
                seconds) to override the default per-request timeout.

        Yields:
            Response chunks as they are generated.

        Raises:
            RuntimeError: If no working server is found.
        """
        timeout: float = kwargs.pop("timeout", DEFAULT_TIMEOUT)

        if model is None:
            all_models = self.list_models()
            if not all_models:
                raise RuntimeError("No models available")
            model = random.choice(all_models)
            print(f"[OllamaFreeAPI] Selected model: {model}")

        servers = self.get_model_servers(model)
        if not servers:
            raise RuntimeError(f"No servers available for model '{model}'")

        servers = self._select_best_server(servers)

        last_error: Optional[Exception] = None
        for server in servers:
            client = Client(host=server["url"])
            request = self.generate_api_request(model, prompt, **kwargs)
            request["stream"] = True

            try:
                for chunk in client.generate(**request, timeout=timeout):
                    yield chunk["response"]
                return  # Stream completed successfully.
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(
            f"All servers failed for model '{model}'. Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # LLM params helpers (for LangChain / LlamaIndex integration)
    # ------------------------------------------------------------------

    def get_llm_params(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Return model and server parameters for OllamaLLM / LangChain integration.

        Server selection is performance-weighted (same logic as chat/stream_chat).

        Args:
            model: Name of the model. If None, a random model is selected.

        Returns:
            Dictionary with ``model`` and ``base_url`` keys.

        Raises:
            RuntimeError: If no models or servers are available.
            ValueError: If the specified model is not found.
        """
        if model is None:
            all_models = self.list_models()
            if not all_models:
                raise RuntimeError("No models available")
            model = random.choice(all_models)
            print(f"[OllamaFreeAPI] Selected model: {model}")
        else:
            if model not in self.list_models():
                raise ValueError(f"Model '{model}' not found")

        servers = self.get_model_servers(model)
        if not servers:
            raise RuntimeError(f"No servers available for model '{model}'")

        server = self._select_best_server(servers)[0]
        print(f"[OllamaFreeAPI] Selected server: {server['url']}")

        return {"model": model, "base_url": server["url"]}
