"""Shared LLM factory for the Text-to-SQL agents.

Centralizes construction of the chat model so every agent uses the same
provider, model, and temperature from config.py. Swapping providers later
is a single change here rather than in each agent.
"""

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import (
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_MODEL,
)
from src.logger import get_logger

logger = get_logger(__name__)


def get_llm() -> BaseChatModel:
    """Builds the configured chat model.

    Reads ``LLM_PROVIDER`` from config and returns the matching LangChain
    chat model, set to the configured temperature.

    Returns:
        A ready-to-use :class:`~langchain_core.language_models.chat_models.BaseChatModel`.

    Raises:
        ValueError: If ``LLM_PROVIDER`` is not a recognised value.
        ImportError: If the provider's integration package is not installed.
    """
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        logger.info("Using OpenAI model '%s'.", OPENAI_MODEL)
        return ChatOpenAI(model=OPENAI_MODEL, temperature=LLM_TEMPERATURE)

    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        logger.info("Using Ollama model '%s'.", OLLAMA_MODEL)
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'openai' or 'ollama'."
    )