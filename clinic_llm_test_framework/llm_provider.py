"""LLM provider wrapper.

This module encapsulates the logic for instantiating chat models from
different providers (currently OpenAI and Google) behind a unified
interface.  By isolating this complexity, the rest of the framework
interacts with models through a simple ``generate`` method and does
not need to be aware of provider‑specific details.

The implementation below is adapted from the user provided
``llm_provider.py`` example.  It loads API keys from environment
variables via ``python-dotenv`` and exposes a :class:`LLMProvider`
class with a single method :meth:`LLMProvider.get_llm` returning a
LangChain chat model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Attempt to load a .env file from the package's parent directory.  This makes
# specifying API keys convenient for local development and CI alike.  We
# deliberately only climb one directory up from the package (rather than two)
# to avoid accidentally reading unrelated .env files (e.g. higher up the
# filesystem hierarchy) that might contain binary data or invalid encodings.
CONFIG_DIR = Path(__file__).resolve().parent
# Load .env from the project root (one level above this package)
_env_path = CONFIG_DIR.parent / ".env"
if _env_path.exists():
    try:
        # Try with load_dotenv first
        load_dotenv(_env_path, override=True)
    except Exception:
        # If load_dotenv fails, manually parse the .env file
        try:
            with open(_env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        except Exception:
            # Last resort: try with different encoding
            try:
                with open(_env_path, 'r', encoding='latin-1') as f:
                    for line in f:
                        line = line.strip()
                        if line and '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
            except Exception:
                pass


class LLMProvider:
    """Factory for chat models.

    Parameters
    ----------
    provider : {"openai", "google"}, optional
        Which provider to instantiate.  Defaults to "openai".
    model_name : str, optional
        The model identifier to use.  If omitted, a sensible default
        is selected depending on the provider (``gpt-4`` for OpenAI,
        ``gemini-pro`` for Google).
    temperature : float, optional
        Sampling temperature passed to the model.  Lower values make
        outputs more deterministic.

    Notes
    -----
    API keys are read from the environment variables ``OPENAI_API_KEY``
    and ``GOOGLE_API_KEY`` respectively.  Missing keys will raise an
    exception at runtime.
    """

    def __init__(
        self,
        provider: Literal["openai", "google"] = "openai",
        model_name: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_llm(self) -> BaseChatModel:
        """Instantiate and return a chat model for the configured provider."""
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("Missing OPENAI_API_KEY")
            return ChatOpenAI(
                openai_api_key=api_key,
                model_name=self.model_name or "gpt-4o-mini",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("Missing GOOGLE_API_KEY")
            return ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=self.model_name or "gemini-pro",
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")