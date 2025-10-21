"""
Refactored LLM Client Module
Centralized configuration-based LLM client for OpenAI-compatible APIs
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from openai import OpenAI
from dotenv import load_dotenv


class ModelConfig:
    """Model configuration data class"""

    def __init__(self, config_dict: Dict[str, Any], model_id: str):
        self.model_id = model_id
        self.provider = config_dict.get("provider", "unknown")
        self.api_base = config_dict.get("api_base")
        self.api_key_env = config_dict.get("api_key_env")
        self.model_name = config_dict.get("model_name")
        self.description = config_dict.get("description", "")
        self.max_tokens = config_dict.get("max_tokens", 4096)
        self.temperature = config_dict.get("temperature", 0.7)

    def __repr__(self):
        return f"ModelConfig(id={self.model_id}, provider={self.provider}, model={self.model_name})"


class LLMClientManager:
    """Manages LLM clients based on centralized configuration"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize LLM client manager

        Args:
            config_path: Path to model_config.yaml. If None, uses default location.
        """
        # Load environment variables
        load_dotenv()

        # Determine config path
        if config_path is None:
            # Default: src/core/model_config.yaml
            config_path = Path(__file__).parent / "model_config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.models: Dict[str, ModelConfig] = {}
        self._load_models()

        # Default model
        self.default_model_id = self.config.get("default_model", "qwen3-32b-dashscope")

        # Timeout settings
        self.timeout = self.config.get("timeout", {
            "connect": 10,
            "read": 60,
            "write": 10
        })

        print(f"Loaded {len(self.models)} model configurations from {config_path}")

    def _load_models(self):
        """Load all model configurations"""
        models_dict = self.config.get("models", {})
        for model_id, model_config in models_dict.items():
            self.models[model_id] = ModelConfig(model_config, model_id)

    def list_models(self) -> List[str]:
        """List all available model IDs"""
        return list(self.models.keys())

    def get_model_info(self, model_id: Optional[str] = None) -> ModelConfig:
        """
        Get model configuration

        Args:
            model_id: Model identifier. If None, uses default model.

        Returns:
            ModelConfig object
        """
        if model_id is None:
            model_id = self.default_model_id

        if model_id not in self.models:
            available = ", ".join(self.models.keys())
            raise ValueError(f"Model '{model_id}' not found. Available: {available}")

        return self.models[model_id]

    def create_client(self, model_id: Optional[str] = None) -> tuple[OpenAI, ModelConfig]:
        """
        Create an OpenAI client for the specified model

        Args:
            model_id: Model identifier. If None, uses default model.

        Returns:
            Tuple of (OpenAI client, ModelConfig)
        """
        model_config = self.get_model_info(model_id)

        # Get API key from environment
        api_key = os.getenv(model_config.api_key_env)
        if not api_key:
            # For local models, use a dummy key if not set
            if model_config.provider == "sglang":
                api_key = "EMPTY"
            else:
                raise ValueError(
                    f"API key not found: {model_config.api_key_env}. "
                    f"Please set it in your .env file."
                )

        # Create OpenAI client
        client = OpenAI(
            api_key=api_key,
            base_url=model_config.api_base,
            timeout=self.timeout.get("read", 60)
        )

        return client, model_config

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate chat completion using specified model

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model_id: Model identifier. If None, uses default model.
            max_tokens: Maximum tokens to generate. If None, uses model default.
            temperature: Sampling temperature. If None, uses model default.
            **kwargs: Additional parameters to pass to the API

        Returns:
            Generated text response
        """
        client, model_config = self.create_client(model_id)

        # Use model defaults if not specified
        if max_tokens is None:
            max_tokens = model_config.max_tokens
        if temperature is None:
            temperature = model_config.temperature

        # Make API call
        response = client.chat.completions.create(
            model=model_config.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        # Extract content
        try:
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            print(f"Error extracting response content: {e}")
            return ""

    def generate_caption(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Convenience method for caption generation

        Args:
            system_prompt: System prompt
            user_prompt: User prompt with features
            model_id: Model identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated caption
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return self.chat_completion(
            messages=messages,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature
        )


# Singleton instance for convenience
_global_manager: Optional[LLMClientManager] = None


def get_manager(config_path: Optional[str] = None) -> LLMClientManager:
    """
    Get or create global LLM client manager

    Args:
        config_path: Path to model config (only used on first call)

    Returns:
        LLMClientManager instance
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = LLMClientManager(config_path)
    return _global_manager


def list_available_models() -> List[str]:
    """List all available models"""
    manager = get_manager()
    return manager.list_models()


def create_llm_client(model_id: Optional[str] = None) -> tuple[OpenAI, ModelConfig]:
    """
    Create LLM client for specified model

    Args:
        model_id: Model identifier (e.g., 'qwen3-32b-dashscope')

    Returns:
        Tuple of (OpenAI client, ModelConfig)
    """
    manager = get_manager()
    return manager.create_client(model_id)


def generate_text(
    messages: List[Dict[str, str]],
    model_id: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **kwargs
) -> str:
    """
    Generate text using specified model

    Args:
        messages: Chat messages
        model_id: Model identifier
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        **kwargs: Additional API parameters

    Returns:
        Generated text
    """
    manager = get_manager()
    return manager.chat_completion(
        messages=messages,
        model_id=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 80)
    print("LLM Client Manager - Test")
    print("=" * 80)

    # Initialize manager
    manager = LLMClientManager()

    # List available models
    print("\nAvailable models:")
    for i, model_id in enumerate(manager.list_models(), 1):
        config = manager.get_model_info(model_id)
        print(f"  {i}. {model_id}")
        print(f"     Provider: {config.provider}")
        print(f"     Model: {config.model_name}")
        print(f"     Description: {config.description}")

    # Test default model
    print(f"\nDefault model: {manager.default_model_id}")

    # Test chat completion
    print("\nTesting chat completion with default model...")
    try:
        response = manager.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello from LLM Client Manager!' in one sentence."}
            ],
            max_tokens=50
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 80)
