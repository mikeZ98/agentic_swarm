"""Provider-agnostic chat-model factory.

Every role resolves to a LangChain `BaseChatModel`, so Langfuse's callback handler sees
token usage and latency uniformly regardless of vendor.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config import ModelConfig, Provider, Settings

type ChatRunnable = Runnable[LanguageModelInput, BaseMessage]


def _require(key: SecretStr | None, env_name: str, role: str) -> SecretStr:
    if key is None or not key.get_secret_value():
        raise RuntimeError(f"{env_name} is required for the '{role}' role but is not set.")
    return key


def build_chat_model(cfg: ModelConfig, settings: Settings, *, role: str) -> BaseChatModel:
    match cfg.provider:
        case Provider.ANTHROPIC:
            return ChatAnthropic(
                model=cfg.model,
                anthropic_api_key=_require(settings.anthropic_api_key, "ANTHROPIC_API_KEY", role),
                temperature=settings.temperature,
                max_tokens=8192,
                default_request_timeout=settings.request_timeout_s,
            )
        case Provider.OPENAI:
            return ChatOpenAI(
                model_name=cfg.model,
                openai_api_key=_require(settings.openai_api_key, "OPENAI_API_KEY", role),
                temperature=settings.temperature,
                request_timeout=settings.request_timeout_s,
            )
        case Provider.OPENROUTER:
            return ChatOpenAI(
                model_name=cfg.model,
                openai_api_key=_require(settings.openrouter_api_key, "OPENROUTER_API_KEY", role),
                openai_api_base=settings.openrouter_base_url,
                temperature=settings.temperature,
                request_timeout=settings.request_timeout_s,
            )
        case Provider.OLLAMA:
            return ChatOllama(
                model=cfg.model,
                base_url=settings.ollama_base_url,
                temperature=settings.temperature,
            )


def build_engineer_model(cfg: ModelConfig, settings: Settings, *, role: str) -> ChatRunnable:
    """Engineer model with an optional transparent local Ollama fallback."""
    fallback = ChatOllama(
        model=settings.ollama_fallback_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
    )
    if cfg.provider is Provider.OLLAMA:
        return build_chat_model(cfg, settings, role=role)
    try:
        primary = build_chat_model(cfg, settings, role=role)
    except RuntimeError:
        if not settings.ollama_fallback_enabled:
            raise
        # No credentials for the primary provider -> run fully local.
        return fallback
    if settings.ollama_fallback_enabled:
        return primary.with_fallbacks([fallback])
    return primary
