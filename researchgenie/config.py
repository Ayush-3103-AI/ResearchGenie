"""Local configuration for the ResearchGenie CLI product.

Stores setup state and provider selection locally so `researchgenie` only
asks the one-time setup questions once per machine.

Secrets handling (explicit product requirement):
- Stored only in this local file, under the user's home directory.
- Never committed to the project's own git history (this file lives outside
  the repository entirely — see CONFIG_DIR below).
- Never printed to the terminal or written into any generated research
  artifact — callers must use `apply_to_environment()` to hand the API key
  to the existing pipeline code via an environment variable, not by passing
  the Config object (or its repr) anywhere near logs or output.
"""

from __future__ import annotations

import dataclasses
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path(os.getenv("RESEARCHGENIE_CONFIG_DIR", str(Path.home() / ".researchgenie")))
CONFIG_PATH = CONFIG_DIR / "config.json"


@dataclass
class Config:
    setup_complete: bool = False
    provider: str = "ollama"  # "ollama" (recommended, default) | "gemini"
    ollama_model: str = "qwen3.5:9b"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    # Scholarly access (#5), each optional. No Semantic Scholar key: the
    # project will not get an S2 plan, so S2 stays keyless.
    openalex_api_key: str | None = None
    contact_email: str | None = None
    ncbi_api_key: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def masked_summary(self) -> dict:
        """Safe-to-print view — every key, and the email, is masked."""
        data = self.to_dict()
        for name in ("gemini_api_key", "openalex_api_key", "ncbi_api_key"):
            if data.get(name):
                key = data[name]
                data[name] = f"{key[:4]}...{key[-2:]}" if len(key) > 8 else "****"
        if data.get("contact_email"):
            data["contact_email"] = "***@" + data["contact_email"].rpartition("@")[2]
        return data


def load() -> Config:
    if not CONFIG_PATH.exists():
        return Config()
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return Config()
    if not isinstance(data, dict):
        return Config()
    known_fields = {f.name for f in dataclasses.fields(Config)}
    return Config(**{key: value for key, value in data.items() if key in known_fields})


def save(config: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    try:
        # Best-effort owner-only permissions — a secret (the Gemini API key,
        # when set) lives in this file. Not meaningful on every filesystem
        # (e.g. most Windows filesystems), so failures here are non-fatal.
        os.chmod(CONFIG_PATH, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def is_setup_complete() -> bool:
    return load().setup_complete


def apply_to_environment(config: Config) -> None:
    """Push the saved config into the environment variables the existing
    pipeline (`shared/utilities/llm_provider.py`) already reads — the
    pipeline code needs zero awareness of this config file or the CLI."""
    os.environ["AI_PROVIDER"] = config.provider
    os.environ["LOCAL_AI_MODEL"] = config.ollama_model
    os.environ["GEMINI_MODEL"] = config.gemini_model
    if config.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = config.gemini_api_key
    apply_scholarly(config)


def apply_scholarly(config: Config) -> None:
    """The scholarly keys and email, read by shared/utilities/scholarly_gateway.py.
    A variable already set in the environment wins over the saved value."""
    for name, value in (("OPENALEX_API_KEY", config.openalex_api_key),
                        ("CONTACT_EMAIL", config.contact_email),
                        ("NCBI_API_KEY", config.ncbi_api_key)):
        if value and not os.environ.get(name):
            os.environ[name] = value
