"""Typed runtime configuration, sourced from environment variables / `.env`."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


class ModelConfig(BaseModel):
    """Provider + model id for a single agent role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Provider
    model: str


class SandboxSettings(BaseModel):
    """Limits for the throwaway container every generated snippet runs in."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    image: str = "agentic-swarm-sandbox:local"
    docker_bin: str = "docker"
    timeout_s: float = Field(default=20.0, gt=0, le=300)
    memory: str = Field(default="256m", pattern=r"^[1-9][0-9]*[kmg]$")
    cpus: float = Field(default=0.5, gt=0, le=8)
    pids_limit: int = Field(default=64, ge=8, le=4096)
    output_limit_bytes: int = Field(default=64 * 1024, ge=1024, le=16 * 1024 * 1024)
    max_code_chars: int = Field(default=200_000, ge=1, le=2_000_000)
    # Lint gate for generated code; security rules are bandit's job.
    ruff_select: tuple[str, ...] = ("E9", "F", "B", "ASYNC")
    bandit_severity: Literal["low", "medium", "high"] = "medium"
    bandit_confidence: Literal["low", "medium", "high"] = "medium"


class McpSettings(BaseModel):
    """How the agents reach the sandbox MCP server."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # stdio: spawn `python -m src.sandbox.server` as a child process (local default).
    # streamable-http: connect to an already-running server, e.g. the compose service.
    transport: Literal["stdio", "streamable-http"] = "stdio"
    url: str = "http://127.0.0.1:8765/mcp"
    call_timeout_s: float = Field(default=120.0, gt=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SWARM_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Role routing -----------------------------------------------------
    architect: ModelConfig = ModelConfig(provider=Provider.ANTHROPIC, model="claude-opus-5-5")
    backend: ModelConfig = ModelConfig(provider=Provider.ANTHROPIC, model="claude-sonnet-5")
    frontend: ModelConfig = ModelConfig(provider=Provider.ANTHROPIC, model="claude-sonnet-5")
    critic: ModelConfig = ModelConfig(provider=Provider.OPENAI, model="gpt-4o")

    # --- Local fallback for engineer nodes --------------------------------
    ollama_fallback_enabled: bool = True
    ollama_fallback_model: str = "llama3.1:8b"

    # --- Loop control -----------------------------------------------------
    max_iterations: int = Field(default=3, ge=0, le=10)
    acceptance_threshold: int = Field(default=80, ge=0, le=100)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    request_timeout_s: float = Field(default=180.0, gt=0)

    # --- Code-execution sandbox (reached only through MCP) ----------------
    sandbox: SandboxSettings = SandboxSettings()
    mcp: McpSettings = McpSettings()

    # --- Provider credentials (un-prefixed, conventional names) -----------
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENROUTER_API_KEY"
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias="OPENROUTER_BASE_URL",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL"
    )

    # --- Langfuse (the SDK reads LANGFUSE_* itself; we only gate on presence)
    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: SecretStr | None = Field(
        default=None, validation_alias="LANGFUSE_SECRET_KEY"
    )
    tracing_flush_timeout_s: float = Field(default=5.0, gt=0, le=60)

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
