# brain/schema.py
"""
The brain's ACTION SPACE: every knob it may turn, with hard bounds.

Everything an agent proposes (LLM strategist, tuner, playbook rule) arrives as a
PATCH — a flat dict of dotted keys, e.g. {"engine.max_num_seqs": 128} — and goes
through `Deployment.patched()`. Unknown keys and out-of-range values raise
PatchError, so a hallucinated vLLM flag can never reach the simulator or a real
server. `extra="forbid"` on every model is what makes that guarantee hold.

Three levels, matching the brain's hierarchy:
  fleet   — which GPU, how many, which model variant / precision   (hours)
  engine  — vLLM engine config on each replica                     (minutes)
  routing — how requests are spread over replicas                  (per request)
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

_STRICT = ConfigDict(extra="forbid", validate_assignment=True)


class PatchError(ValueError):
    """A patch names an unknown knob or an invalid value."""


class SLA(BaseModel):
    """Service-level objective. The brain meets this first, then minimizes cost."""
    model_config = _STRICT

    ttft_p99_ms: float = Field(gt=0, description="p99 time to first token")
    tpot_p99_ms: float = Field(gt=0, description="p99 time per output token")
    max_quality_loss_pct: float = Field(
        default=1.0, ge=0, le=100,
        description="allowed relative accuracy loss from quantization / smaller variant",
    )


class FleetConfig(BaseModel):
    model_config = _STRICT

    gpu: str
    model: str
    precision: str = "bf16"
    tensor_parallel: Literal[1, 2, 4, 8] = 1
    num_replicas: int = Field(default=1, ge=1, le=256)


class EngineConfig(BaseModel):
    """vLLM engine arguments. Defaults approximate vLLM V1 online-serving defaults."""
    model_config = _STRICT

    max_num_seqs: int = Field(default=256, ge=1, le=2048)
    max_num_batched_tokens: int = Field(default=2048, ge=16, le=131072)
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    gpu_memory_utilization: float = Field(default=0.90, ge=0.30, le=0.98)
    kv_cache_dtype: Literal["auto", "fp8"] = "auto"
    # None = vLLM's default: the model's full context window.
    max_model_len: int | None = Field(default=None, ge=256)
    speculative: Literal["off", "ngram", "draft"] = "off"
    num_speculative_tokens: int = Field(default=4, ge=1, le=8)


class RoutingConfig(BaseModel):
    model_config = _STRICT

    policy: Literal["round_robin", "least_requests", "least_tokens", "prefix_affinity"] = (
        "round_robin"
    )


SECTIONS = ("fleet", "engine", "routing")


class Deployment(BaseModel):
    """One complete, concrete serving setup the brain can evaluate."""
    model_config = _STRICT

    fleet: FleetConfig
    engine: EngineConfig = EngineConfig()
    routing: RoutingConfig = RoutingConfig()

    def get(self, dotted: str) -> Any:
        section, knob = _split(dotted)
        return getattr(getattr(self, section), knob)

    def patched(self, patch: dict[str, Any]) -> "Deployment":
        """Return a new Deployment with `patch` applied; never mutates self.

        Values may be literals or {"scale": f} for numeric knobs (used by rules
        learned as "double max_num_seqs" rather than "set it to 512").
        """
        data = self.model_dump()
        for dotted, value in patch.items():
            section, knob = _split(dotted)
            if isinstance(value, dict):
                if set(value) != {"scale"}:
                    raise PatchError(f"{dotted}: only {{'scale': x}} is allowed, got {value!r}")
                current = data[section][knob]
                if not isinstance(current, (int, float)) or isinstance(current, bool):
                    raise PatchError(f"{dotted}: 'scale' needs a numeric knob, current={current!r}")
                scaled = current * float(value["scale"])
                value = int(round(scaled)) if isinstance(current, int) else round(scaled, 4)
            data[section][knob] = value
        try:
            return Deployment.model_validate(data)
        except ValidationError as e:
            raise PatchError(f"invalid patch {patch!r}: {_short(e)}") from e

    def key(self) -> str:
        """Stable identity, for caching and de-duplicating evaluations."""
        return self.model_dump_json()


def _split(dotted: str) -> tuple[str, str]:
    section, _, knob = dotted.partition(".")
    if section not in SECTIONS or not knob:
        raise PatchError(f"unknown knob {dotted!r}; keys look like 'engine.max_num_seqs'")
    model_cls = {"fleet": FleetConfig, "engine": EngineConfig, "routing": RoutingConfig}[section]
    if knob not in model_cls.model_fields:
        raise PatchError(
            f"unknown knob {dotted!r}. Valid {section} knobs: {sorted(model_cls.model_fields)}"
        )
    return section, knob


def _short(e: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
    )


def knob_catalog() -> dict[str, dict]:
    """Every patchable knob with its type and bounds — what an LLM is shown."""
    out = {}
    for section, cls in (("fleet", FleetConfig), ("engine", EngineConfig), ("routing", RoutingConfig)):
        props = cls.model_json_schema()["properties"]
        for knob, spec in props.items():
            spec = {k: v for k, v in spec.items() if k != "title"}
            out[f"{section}.{knob}"] = spec
    return out
