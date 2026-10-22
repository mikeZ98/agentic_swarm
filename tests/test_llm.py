"""Provider factory: routing per role, fail-fast keys and the local Ollama fallback.

Constructing LangChain chat models performs no network I/O, so these stay hermetic.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableWithFallbacks
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.agents import SwarmAgents, build_agents
from src.agents.llm import build_chat_model, build_engineer_model
from src.config import ModelConfig, Provider, Settings
from tests.fakes import FakeCodeTools


def _settings(*, keys: bool = True, **overrides: Any) -> Settings:
    secret = SecretStr if keys else (lambda _: None)
    return Settings(
        _env_file=None,
        ANTHROPIC_API_KEY=secret("sk-ant-test"),
        OPENAI_API_KEY=secret("sk-test"),
        OPENROUTER_API_KEY=secret("sk-or-test"),
        **overrides,
    )


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        (Provider.ANTHROPIC, ChatAnthropic),
        (Provider.OPENAI, ChatOpenAI),
        (Provider.OPENROUTER, ChatOpenAI),
        (Provider.OLLAMA, ChatOllama),
    ],
)
def test_each_provider_builds_its_chat_model(provider: Provider, expected: type) -> None:
    model = build_chat_model(ModelConfig(provider=provider, model="m"), _settings(), role="critic")

    assert isinstance(model, expected)


def test_openrouter_targets_the_openrouter_endpoint() -> None:
    model = build_chat_model(
        ModelConfig(provider=Provider.OPENROUTER, model="openai/gpt-4o"), _settings(), role="critic"
    )

    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_base == "https://openrouter.ai/api/v1"


@pytest.mark.parametrize(
    ("provider", "env_name"),
    [
        (Provider.ANTHROPIC, "ANTHROPIC_API_KEY"),
        (Provider.OPENAI, "OPENAI_API_KEY"),
        (Provider.OPENROUTER, "OPENROUTER_API_KEY"),
    ],
)
def test_a_missing_key_fails_fast_naming_the_role(provider: Provider, env_name: str) -> None:
    with pytest.raises(RuntimeError, match=f"{env_name} is required for the 'architect' role"):
        build_chat_model(
            ModelConfig(provider=provider, model="m"), _settings(keys=False), role="architect"
        )


def test_engineer_gets_a_transparent_ollama_fallback() -> None:
    cfg = ModelConfig(provider=Provider.ANTHROPIC, model="claude-sonnet-5")

    model = build_engineer_model(cfg, _settings(), role="backend")

    assert isinstance(model, RunnableWithFallbacks)
    assert isinstance(model.runnable, ChatAnthropic)
    assert [type(f) for f in model.fallbacks] == [ChatOllama]


def test_engineer_without_a_key_runs_fully_local() -> None:
    cfg = ModelConfig(provider=Provider.ANTHROPIC, model="claude-sonnet-5")

    model = build_engineer_model(cfg, _settings(keys=False), role="backend")

    assert isinstance(model, ChatOllama)
    assert model.model == "llama3.1:8b"


def test_engineer_without_a_key_and_without_fallback_fails() -> None:
    cfg = ModelConfig(provider=Provider.ANTHROPIC, model="claude-sonnet-5")
    settings = _settings(keys=False, ollama_fallback_enabled=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_engineer_model(cfg, settings, role="backend")


def test_engineer_with_fallback_disabled_uses_the_primary_only() -> None:
    cfg = ModelConfig(provider=Provider.OPENAI, model="gpt-4o")

    model = build_engineer_model(cfg, _settings(ollama_fallback_enabled=False), role="backend")

    assert isinstance(model, ChatOpenAI)


def test_an_ollama_engineer_needs_no_fallback() -> None:
    cfg = ModelConfig(provider=Provider.OLLAMA, model="qwen2.5-coder:7b")

    model = build_engineer_model(cfg, _settings(), role="frontend")

    assert isinstance(model, ChatOllama)
    assert model.model == "qwen2.5-coder:7b"


def test_build_agents_wires_all_four_roles() -> None:
    agents = build_agents(_settings(), FakeCodeTools())

    assert isinstance(agents, SwarmAgents)
    assert [type(a).__name__ for a in (agents.architect, agents.backend, agents.frontend)] == [
        "ArchitectAgent",
        "BackendAgent",
        "FrontendAgent",
    ]
    assert type(agents.critic).__name__ == "CriticAgent"
