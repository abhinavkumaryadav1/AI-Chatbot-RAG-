"""
Ollama Client Module
Communicates with the local Ollama instance via HTTP API without requiring external keys.
"""

from typing import List, Dict, Any, Generator, Tuple
import json
import requests
from config import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL


class OllamaClientError(Exception):
    """Custom exception raised for Ollama API issues."""
    pass


class OllamaClient:
    """HTTP Client for interacting with a locally running Ollama LLM server."""

    def __init__(self, host: str = DEFAULT_OLLAMA_HOST, default_model: str = DEFAULT_OLLAMA_MODEL):
        """
        Initializes the Ollama HTTP client.

        Args:
            host: Ollama server base URL (e.g. 'http://localhost:11434').
            default_model: Default LLM identifier (e.g. 'llama3.2:3b').
        """
        self.host = host.rstrip("/")
        self.default_model = default_model

    def check_connection(self) -> Tuple[bool, str]:
        """
        Verifies if Ollama is running and responding to HTTP requests.

        Returns:
            Tuple of (is_connected: bool, message: str)
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=3)
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name") for m in data.get("models", [])]
                if models:
                    return True, f"Connected to Ollama ({len(models)} model(s) available)."
                else:
                    return True, "Connected to Ollama, but no models are downloaded yet."
            return False, f"Ollama responded with HTTP {response.status_code}."
        except requests.exceptions.ConnectionError:
            return False, "Could not connect to Ollama. Make sure Ollama is installed and running locally."
        except requests.exceptions.Timeout:
            return False, "Connection to Ollama timed out."
        except Exception as e:
            return False, f"Ollama connection check failed: {str(e)}"

    def list_models(self) -> List[str]:
        """
        Retrieves the list of installed local Ollama models.

        Returns:
            List of model names (e.g. ['llama3.2:3b', 'mistral:latest']).
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m.get("name") for m in data.get("models", []) if m.get("name")]
            return []
        except Exception:
            return []

    def is_model_available(self, model_name: str) -> bool:
        """
        Checks whether a specified model is downloaded and ready.

        Args:
            model_name: Name of the model.

        Returns:
            True if model is present, False otherwise.
        """
        models = self.list_models()
        # Direct match or partial tag match (e.g. 'llama3.2:3b' matching 'llama3.2:3b')
        return any(model_name == m or model_name in m for m in models)

    def generate(
        self,
        prompt: str,
        system: str = "",
        model: str = "",
        temperature: float = 0.1,
        timeout: int = 120
    ) -> str:
        """
        Generates a non-streaming text completion from Ollama.

        Args:
            prompt: User/context prompt text.
            system: System instructions.
            model: Model name (defaults to self.default_model).
            temperature: Sampling temperature (0.1 for high factual adherence).
            timeout: Maximum wait time in seconds.

        Returns:
            Generated text string.
        """
        target_model = model or self.default_model
        payload = {
            "model": target_model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=timeout
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()
            elif response.status_code == 404:
                raise OllamaClientError(
                    f"Model '{target_model}' was not found in your Ollama library. "
                    f"Please run `ollama pull {target_model}` in your terminal."
                )
            else:
                raise OllamaClientError(
                    f"Ollama returned HTTP error {response.status_code}: {response.text}"
                )

        except requests.exceptions.ConnectionError:
            raise OllamaClientError(
                "Unable to connect to Ollama. Please verify that the Ollama app is open and running."
            )
        except requests.exceptions.Timeout:
            raise OllamaClientError(
                f"Ollama generation timed out after {timeout} seconds."
            )
        except Exception as e:
            if isinstance(e, OllamaClientError):
                raise
            raise OllamaClientError(f"Error during Ollama inference: {str(e)}")

    def generate_stream(
        self,
        prompt: str,
        system: str = "",
        model: str = "",
        temperature: float = 0.1,
        timeout: int = 120
    ) -> Generator[str, None, None]:
        """
        Streams generated text tokens from Ollama for live UI feedback.

        Args:
            prompt: User/context prompt text.
            system: System instructions.
            model: Model name.
            temperature: Sampling temperature.
            timeout: Maximum initial connection timeout.

        Yields:
            Token text chunks as they arrive.
        """
        target_model = model or self.default_model
        payload = {
            "model": target_model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                stream=True,
                timeout=timeout
            )

            if response.status_code != 200:
                if response.status_code == 404:
                    raise OllamaClientError(
                        f"Model '{target_model}' not found. Run `ollama pull {target_model}`."
                    )
                raise OllamaClientError(f"Ollama returned HTTP {response.status_code}.")

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    yield chunk.get("response", "")
                    if chunk.get("done", False):
                        break

        except requests.exceptions.ConnectionError:
            raise OllamaClientError("Ollama service connection was lost during generation.")
        except requests.exceptions.Timeout:
            raise OllamaClientError(f"Ollama stream timed out after {timeout} seconds.")
        except Exception as e:
            if isinstance(e, OllamaClientError):
                raise
            raise OllamaClientError(f"Streaming error: {str(e)}")

