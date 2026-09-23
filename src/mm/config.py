"""Runtime settings loaded from environment / .env. Secrets are SecretStr so they never print."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM providers — both reached through OpenAI-compatible endpoints.
    groq_api_key: SecretStr | None = None
    mm_primary_base_url: str = "https://api.groq.com/openai/v1"
    mm_primary_model: str = "openai/gpt-oss-120b"

    gemini_api_key: SecretStr | None = None
    mm_fallback_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    mm_fallback_model: str = "gemini-3.8-flash"

    # Mock bank target.
    mockbank_url: str = "http://127.0.0.1:8600"
    mockbank_username: str = "operator1"
    mockbank_password: SecretStr = SecretStr("change-me-local-only")
    mockbank_session_secret: SecretStr = SecretStr("dev-only-session-secret")
    mockbank_tenant: str = "tenant_a"

    # Operator console / runs.
    mm_operator_port: int = 8700
    mm_headless: bool = False
    mm_max_steps: int = 25
    mm_runs_dir: Path = Path("runs")


@lru_cache
def get_settings() -> Settings:
    return Settings()
