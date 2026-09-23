"""Typed runtime configuration, sourced from environment variables / `.env`."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


class ModelConfig(BaseModel):
    """Provider + model id for a single agent role."""

    provider: Provider
    model: str


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

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
