"""Runtime settings loaded from environment / .env. Secrets are SecretStr so they never print."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project files are found relative to the project, never to the directory `mm` happens to be run from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # LLM providers — both reached through OpenAI-compatible endpoints.
    groq_api_key: SecretStr | None = None
    mm_primary_base_url: str = "https://api.groq.com/openai/v1"
    mm_primary_model: str = "openai/gpt-oss-120b"

    gemini_api_key: SecretStr | None = None
    mm_fallback_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    mm_fallback_model: str = "gemini-3.8-flash,gemini-3.1-flash-lite"  # comma-separated, tried in order

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
    mm_runs_dir: Path = PROJECT_ROOT / "runs"
    mm_policy_path: Path = PROJECT_ROOT / "config" / "policy.yaml"
    mm_packs_dir: Path = PROJECT_ROOT / "packs"
    mm_capabilities_dir: Path = PROJECT_ROOT / "capabilities"
    mm_tenants_dir: Path = PROJECT_ROOT / "tenants"

    @field_validator("mm_runs_dir", "mm_policy_path", "mm_packs_dir", "mm_capabilities_dir", "mm_tenants_dir")
    @classmethod
    def _from_project_root(cls, path: Path) -> Path:
        """A relative path in .env means relative to the project, not to wherever `mm` was started."""
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
