# brain/facts.py
"""
FACTS tier of the brain's knowledge base: what is physically true about GPUs,
models and weight formats, loaded from brain/catalog/*.yaml.

Strategy knowledge is expressed over TRAITS (e.g. "low-bandwidth GPU", "long
prompts"), never over raw names alone, so that something learned on an L4 can
transfer to any other bandwidth-starved GPU. This module is the single place
that turns catalog numbers into those traits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

CATALOG_DIR = Path(__file__).parent / "catalog"


class CatalogError(ValueError):
    """A catalog file is missing a field, or a lookup names something unknown."""


@dataclass(frozen=True)
class GPU:
    name: str
    arch: str
    vendor: str
    memory_gb: float
    mem_bw_gbps: float
    bf16_tflops: float
    fp8_tflops: float | None
    int8_tops: float
    fp4_tflops: float | None
    link_gbps: float
    max_tp: int
    price_per_hour: float

    @property
    def fp8(self) -> bool:
        return self.fp8_tflops is not None

    @property
    def fp4(self) -> bool:
        return self.fp4_tflops is not None

    def peak_tflops(self, compute: str) -> float:
        """Tensor-core peak for a compute format; falls back to bf16 if unsupported."""
        peak = {
            "bf16": self.bf16_tflops,
            "fp8": self.fp8_tflops,
            "int8": self.int8_tops,
            "fp4": self.fp4_tflops,
        }.get(compute)
        return peak if peak is not None else self.bf16_tflops

    def supports(self, capability: str | None) -> bool:
        if capability is None:
            return True
        return {"fp8": self.fp8, "fp4": self.fp4}.get(capability, False)


@dataclass(frozen=True)
class Model:
    name: str
    hf_id: str
    family: str
    params_b: float
    n_layers: int
    hidden: int
    n_kv_heads: int
    head_dim: int
    max_context: int
    quality: float
    active_params_b: float | None = None
    moe: dict | None = None

    @property
    def n_heads(self) -> int:
        return self.hidden // self.head_dim

    @property
    def active_b(self) -> float:
        """Parameters touched per token (differs from params_b only for MoE)."""
        return self.active_params_b or self.params_b

    def kv_bytes_per_token(self, dtype_bytes: float = 2.0) -> float:
        """Whole-model KV bytes for one token (K and V, every layer)."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * dtype_bytes


@dataclass(frozen=True)
class Precision:
    name: str
    weight_bytes: float
    compute: str
    flops_overhead: float
    quality_loss_pct: float
    requires: str | None
    vllm_flag: str | None


@dataclass
class Facts:
    gpus: dict[str, GPU]
    models: dict[str, Model]
    precisions: dict[str, Precision]
    gpu_notes: dict[str, str] = field(default_factory=dict)
    prior_rules: list[dict] = field(default_factory=list)

    # ── lookups that fail loudly with the valid choices ────────────────
    def gpu(self, name: str) -> GPU:
        return _lookup(self.gpus, name, "GPU")

    def model(self, name: str) -> Model:
        return _lookup(self.models, name, "model")

    def precision(self, name: str) -> Precision:
        return _lookup(self.precisions, name, "precision")

    def family_members(self, family: str) -> list[Model]:
        """Models of one family, smallest first (variant choice / draft models)."""
        return sorted(
            (m for m in self.models.values() if m.family == family),
            key=lambda m: m.params_b,
        )


def _lookup(table: dict, name: str, kind: str):
    if name not in table:
        raise CatalogError(f"Unknown {kind} {name!r}. Known: {sorted(table)}")
    return table[name]


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise CatalogError(f"Catalog file missing: {path}")
    return yaml.safe_load(path.read_text()) or {}


def _build(cls, name: str, raw: dict, source: Path):
    try:
        return cls(name=name, **raw)
    except TypeError as e:
        raise CatalogError(f"{source.name}: entry {name!r} is malformed: {e}") from e


def load_facts(catalog_dir: str | Path = CATALOG_DIR) -> Facts:
    d = Path(catalog_dir)
    gpus_p, models_p, prec_p = d / "gpus.yaml", d / "models.yaml", d / "precisions.yaml"
    gpus = {k: _build(GPU, k, v, gpus_p) for k, v in _read_yaml(gpus_p).items()}
    models = {k: _build(Model, k, v, models_p) for k, v in _read_yaml(models_p).items()}
    precisions = {k: _build(Precision, k, v, prec_p) for k, v in _read_yaml(prec_p).items()}

    priors_p = d / "priors.yaml"
    priors = _read_yaml(priors_p) if priors_p.exists() else {}
    notes = priors.get("gpu_notes") or {}
    unknown = set(notes) - set(gpus)
    if unknown:
        raise CatalogError(f"priors.yaml has notes for unknown GPU(s): {sorted(unknown)}")

    return Facts(gpus, models, precisions, gpu_notes=notes, prior_rules=priors.get("rules") or [])


@lru_cache(maxsize=1)
def default_facts() -> Facts:
    return load_facts()


# ── traits: the vocabulary strategy rules are written in ───────────────

BANDWIDTH_CLASSES = ("low", "mid", "high")
MEMORY_CLASSES = ("small", "medium", "large", "xlarge")
SIZE_CLASSES = ("small", "medium", "large")
LENGTH_CLASSES = ("short", "medium", "long")

# Every trait a rule may condition on, with its allowed values (None = any
# catalog name). Workload traits are computed from a workload spec (M2); they
# are declared here so rules referencing them can be validated today.
TRAIT_VOCAB: dict[str, tuple | None] = {
    "gpu.name": None,
    "gpu.vendor": ("nvidia", "amd"),
    "gpu.fp8": (True, False),
    "gpu.fp4": (True, False),
    "gpu.bandwidth_class": BANDWIDTH_CLASSES,
    "gpu.memory_class": MEMORY_CLASSES,
    "gpu.interconnect": ("pcie", "nvlink"),
    "model.family": None,
    "model.size_class": SIZE_CLASSES,
    "model.moe": (True, False),
    "workload.prompt_class": LENGTH_CLASSES,
    "workload.output_class": LENGTH_CLASSES,
    "workload.prefix_heavy": (True, False),
    "workload.bursty": (True, False),
    "workload.latency_strict": (True, False),
}


def gpu_traits(gpu: GPU) -> dict:
    bw = gpu.mem_bw_gbps
    mem = gpu.memory_gb
    return {
        "gpu.name": gpu.name,
        "gpu.vendor": gpu.vendor,
        "gpu.fp8": gpu.fp8,
        "gpu.fp4": gpu.fp4,
        "gpu.bandwidth_class": "low" if bw < 1000 else "mid" if bw < 2500 else "high",
        "gpu.memory_class": (
            "small" if mem <= 24 else "medium" if mem <= 48 else "large" if mem <= 80 else "xlarge"
        ),
        "gpu.interconnect": "nvlink" if gpu.link_gbps >= 200 else "pcie",
    }


def model_traits(model: Model) -> dict:
    p = model.params_b
    return {
        "model.family": model.family,
        "model.size_class": "small" if p < 5 else "medium" if p < 20 else "large",
        "model.moe": model.moe is not None,
    }


def ridge_point(gpu: GPU, compute: str = "bf16") -> float:
    """FLOPs per byte at which a kernel stops being bandwidth-bound.

    Decode at batch B does ~2*B FLOPs per weight byte (bf16), so B below
    ridge/2 means the GPU is waiting on memory — the core reason batching,
    quantization and speculative decoding help.
    """
    return gpu.peak_tflops(compute) * 1e12 / (gpu.mem_bw_gbps * 1e9)


def validate_condition(when: dict, facts: Facts) -> list[str]:
    """Problems with a rule's `when` clause (empty list = valid)."""
    problems = []
    for key, value in when.items():
        if key not in TRAIT_VOCAB:
            problems.append(f"unknown trait {key!r} (known: {sorted(TRAIT_VOCAB)})")
            continue
        allowed = TRAIT_VOCAB[key]
        if allowed is None:
            names = facts.gpus if key == "gpu.name" else {m.family for m in facts.models.values()}
            if value not in names:
                problems.append(f"{key}={value!r} is not in the catalog")
        elif value not in allowed:
            problems.append(f"{key}={value!r} not in {list(allowed)}")
    return problems
