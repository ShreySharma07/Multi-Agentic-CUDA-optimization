# config.py
"""
Loads karma.yaml into validated, per-agent LLM settings.

Design: every agent is configured independently (provider, model, api_key_env,
temperature). Values omitted for an agent are inherited from `defaults`. Secrets
are never stored here -- an agent names the environment variable holding its key
and we resolve it at build time, so karma.yaml stays safe to commit.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = "karma.yaml"

# Agents the pipeline knows how to build. Used to validate karma.yaml so a typo
# ("planer:") fails loudly instead of silently falling back to defaults.
KNOWN_AGENTS = ("coder", "classifier", "planner", "reflector")

# Providers that need no API key.
KEYLESS_PROVIDERS = ("ollama",)


class AgentSettings(BaseModel):
    """Fully-resolved settings for a single agent."""
    provider: str
    model: str
    api_key_env: str | None = None
    api_key: str | None = Field(default=None, repr=False)  # resolved, never printed
    temperature: float = 0.2
    max_retries: int = 5
    timeout: int = 120
    base_url: str | None = None  # ollama / self-hosted OpenAI-compatible

    def redacted(self) -> str:
        key = "set" if self.api_key else ("n/a" if self.provider in KEYLESS_PROVIDERS else "MISSING")
        return f"{self.provider}:{self.model} (temp={self.temperature}, key={key})"


class EvalSettings(BaseModel):
    """Torch-extension evaluation knobs (kernelbench path)."""
    warmup: int = 10
    runs: int = 30
    translate_retries: int = 3
    target_speedup_vs_compile: float = 1.05
    compile_timeout_s: int = 180
    ncu_timeout_s: int = 120
    # Skip tasks whose peak memory exceeds this fraction of VRAM. On Windows the
    # WDDM driver silently pages past VRAM into system RAM instead of raising, so
    # an oversized task yields plausible-looking timings that only measure PCIe.
    vram_headroom: float = 0.80


class KarmaConfig(BaseModel):
    agents: dict[str, AgentSettings]
    eval: EvalSettings = EvalSettings()

    def for_agent(self, name: str) -> AgentSettings:
        if name not in self.agents:
            raise KeyError(f"No config for agent {name!r}. Known: {sorted(self.agents)}")
        return self.agents[name]


class ConfigError(RuntimeError):
    pass


def _resolve_key(provider: str, api_key_env: str | None, agent: str) -> str | None:
    if provider in KEYLESS_PROVIDERS:
        return None
    if not api_key_env:
        raise ConfigError(
            f"agent {agent!r} (provider {provider!r}) has no `api_key_env`. "
            f"Name the env var holding its key, e.g. `api_key_env: GEMINI_API_KEY`."
        )
    key = os.getenv(api_key_env)
    if not key:
        raise ConfigError(
            f"agent {agent!r} needs env var {api_key_env!r}, which is unset. "
            f"Add `{api_key_env}=...` to your .env file."
        )
    return key


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> KarmaConfig:
    """
    Read karma.yaml, merge `defaults` into each agent, resolve API keys from the
    environment. Raises ConfigError with an actionable message on any problem --
    a misconfigured LLM should fail at startup, not 40 minutes into a run.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path.resolve()}")

    try:
        raw = yaml.safe_load(cfg_path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{cfg_path} is not valid YAML: {e}") from e

    defaults: dict = raw.get("defaults") or {}
    agents_raw: dict = raw.get("agents") or {}
    eval_raw: dict = raw.get("eval") or {}

    try:
        eval_settings = EvalSettings(**eval_raw)
    except Exception as e:
        raise ConfigError(f"invalid `eval:` section in {cfg_path}: {e}") from e

    unknown = set(agents_raw) - set(KNOWN_AGENTS)
    if unknown:
        raise ConfigError(
            f"Unknown agent(s) in {cfg_path}: {sorted(unknown)}. Known: {list(KNOWN_AGENTS)}"
        )

    agents: dict[str, AgentSettings] = {}
    for name in KNOWN_AGENTS:
        merged = {**defaults, **(agents_raw.get(name) or {})}

        provider = merged.get("provider")
        model = merged.get("model")
        if not provider or not model:
            raise ConfigError(
                f"agent {name!r} is missing `provider` and/or `model` "
                f"(set them under `agents.{name}` or under `defaults`)."
            )

        merged["api_key"] = _resolve_key(provider, merged.get("api_key_env"), name)
        agents[name] = AgentSettings(**merged)

    return KarmaConfig(agents=agents, eval=eval_settings)
