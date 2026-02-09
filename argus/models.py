"""
Model factory for Agno-based LLM access.
Supports OpenAI, Anthropic, and Google providers.
"""
from typing import Any

from .config import ModelConfig


def get_agno_model(config: ModelConfig) -> Any:
    """
    Factory to return an Agno model based on config.
    """
    common_kwargs = {
        "temperature": config.temperature,
    }
    
    if config.max_output_tokens is not None:
        common_kwargs["max_tokens"] = config.max_output_tokens

    if config.provider == "openai":
        from agno.models.openai import OpenAIChat
        
        return OpenAIChat(
            id=config.model_name,
            api_key=config.api_key.get_secret_value(),
            **common_kwargs,
        )
    elif config.provider == "anthropic":
        from agno.models.anthropic import Claude

        return Claude(
            id=config.model_name,
            api_key=config.api_key.get_secret_value(),
            **common_kwargs,
        )
    elif config.provider == "google":
        # Monkeypatch for google-genai < 1.0.0 or specific environments missing FileSearch
        try:
            import google.genai.types as genai_types
            if not hasattr(genai_types, 'FileSearch'):
                class DummyFileSearch:
                    def __init__(self, **kwargs): pass
                setattr(genai_types, 'FileSearch', DummyFileSearch)
        except ImportError:
            pass

        from agno.models.google import Gemini

        return Gemini(
            id=config.model_name,
            api_key=config.api_key.get_secret_value(),
            **common_kwargs,
        )

    raise ValueError(f"Unsupported provider: {config.provider}")
