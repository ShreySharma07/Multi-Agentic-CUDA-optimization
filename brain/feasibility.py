# brain/feasibility.py
"""
Can this deployment even start? And what does it cost?

This is the brain's first, cheapest filter. Before anything is simulated (M2) or
launched on a real GPU, a deployment must pass the checks vLLM itself would
fail on at startup — weights that do not fit, a KV cache too small for one
max-length sequence, tensor-parallel sizes that do not divide the attention
heads, precisions the GPU has no tensor cores for — plus the brain's own quality
budget. Each error carries a fix hint so an agent can act on it directly.

Memory model per GPU (all in bytes, inside gpu_memory_utilization * memory):
    weights / tp  +  activation peak  +  fixed runtime overhead  +  KV pool
The KV pool is whatever is left, and it decides how many tokens can be in
flight at once — the single most important number for throughput.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from brain.facts import Facts, GPU, Model, Precision, default_facts, validate_condition
from brain.schema import SLA, Deployment, FleetConfig, PatchError

GB = 1024 ** 3
KV_BLOCK_TOKENS = 16          # vLLM's default paged-attention block size
FIXED_OVERHEAD_GB = 1.0       # CUDA context, CUDA graphs, sampler buffers
ACTIVATION_FACTOR = 16        # bytes-per-(token*hidden) at the activation peak, bf16 MLP


@dataclass
class MemoryPlan:
    budget_gb: float               # gpu_memory_utilization * memory, per GPU
    weights_gb: float              # per GPU, after tensor-parallel sharding
    activation_gb: float
    overhead_gb: float
    kv_pool_gb: float              # per GPU; <= 0 means nothing fits
    kv_bytes_per_token: float      # per GPU
    kv_capacity_tokens: int        # tokens in flight per replica
    max_model_len: int


@dataclass
class Feasibility:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    memory: MemoryPlan | None = None
    quality_loss_pct: float = 0.0
    gpus_total: int = 0
    cost_per_hour: float = 0.0

    def summary(self) -> str:
        lines = [f"feasible: {self.ok}"]
        if self.memory:
            m = self.memory
            lines.append(
                f"memory/GPU: budget {m.budget_gb:.1f} GB = weights {m.weights_gb:.1f} + "
                f"activations {m.activation_gb:.2f} + overhead {m.overhead_gb:.1f} + "
                f"KV {max(m.kv_pool_gb, 0):.1f}"
            )
            lines.append(
                f"KV capacity: {m.kv_capacity_tokens:,} tokens/replica "
                f"(~{m.kv_capacity_tokens // max(m.max_model_len, 1)} full-length seqs at "
                f"max_model_len={m.max_model_len:,})"
            )
        lines.append(f"quality loss: {self.quality_loss_pct:.2f}%")
        lines.append(f"GPUs: {self.gpus_total}  cost: ${self.cost_per_hour:.2f}/hr")
        lines += [f"ERROR: {e}" for e in self.errors]
        lines += [f"warn:  {w}" for w in self.warnings]
        return "\n".join(lines)


def kv_bytes_per_token_per_gpu(model: Model, tp: int, kv_dtype: str) -> float:
    """KV heads shard across TP ranks; with tp > n_kv_heads each rank keeps a replica."""
    heads_per_gpu = max(1, model.n_kv_heads // tp)
    dtype_bytes = 1.0 if kv_dtype == "fp8" else 2.0
    return 2 * model.n_layers * heads_per_gpu * model.head_dim * dtype_bytes


def memory_plan(dep: Deployment, gpu: GPU, model: Model, prec: Precision) -> MemoryPlan:
    tp = dep.fleet.tensor_parallel
    eng = dep.engine
    budget = gpu.memory_gb * eng.gpu_memory_utilization
    weights = model.params_b * 1e9 * prec.weight_bytes / tp / GB
    activation = eng.max_num_batched_tokens * model.hidden * ACTIVATION_FACTOR / tp / GB
    kv_pool = budget - weights - activation - FIXED_OVERHEAD_GB
    kv_tok = kv_bytes_per_token_per_gpu(model, tp, eng.kv_cache_dtype)
    blocks = max(0, int(kv_pool * GB // (kv_tok * KV_BLOCK_TOKENS)))
    return MemoryPlan(
        budget_gb=budget,
        weights_gb=weights,
        activation_gb=activation,
        overhead_gb=FIXED_OVERHEAD_GB,
        kv_pool_gb=kv_pool,
        kv_bytes_per_token=kv_tok,
        kv_capacity_tokens=blocks * KV_BLOCK_TOKENS,
        max_model_len=eng.max_model_len or model.max_context,
    )


def quality_loss_pct(model: Model, prec: Precision, reference: Model | None) -> float:
    """Relative accuracy loss vs serving `reference` in bf16."""
    variant_loss = 0.0
    if reference is not None and reference.name != model.name:
        variant_loss = max(0.0, (reference.quality - model.quality) / reference.quality * 100)
    return variant_loss + prec.quality_loss_pct


def check(
    dep: Deployment,
    facts: Facts | None = None,
    *,
    sla: SLA | None = None,
    reference_model: str | None = None,
) -> Feasibility:
    """Run every startup and budget check. Never raises for a bad deployment."""
    facts = facts or default_facts()
    res = Feasibility(ok=False)
    f, eng = dep.fleet, dep.engine

    try:
        gpu, model, prec = facts.gpu(f.gpu), facts.model(f.model), facts.precision(f.precision)
        ref = facts.model(reference_model) if reference_model else None
    except Exception as e:  # CatalogError
        res.errors.append(str(e))
        return res

    res.gpus_total = f.tensor_parallel * f.num_replicas
    res.cost_per_hour = res.gpus_total * gpu.price_per_hour

    # ── hardware support ───────────────────────────────────────────────
    if not gpu.supports(prec.requires):
        res.errors.append(
            f"{f.precision} needs {prec.requires} tensor cores, which {gpu.name} lacks"
            f" — use bf16, int8_w8a8 or awq_int4"
        )
    if eng.kv_cache_dtype == "fp8" and not gpu.fp8:
        res.warnings.append(
            f"fp8 KV cache on {gpu.name} (no fp8 cores) works via conversion: "
            f"capacity doubles but attention is slower"
        )
    if gpu.vendor == "amd" and f.precision in ("awq_int4", "nvfp4"):
        res.errors.append(f"{f.precision} kernels are not available on vLLM ROCm for {gpu.name}")

    # ── tensor parallel ────────────────────────────────────────────────
    tp = f.tensor_parallel
    if tp > gpu.max_tp:
        res.errors.append(f"tensor_parallel={tp} exceeds {gpu.name} max of {gpu.max_tp} per node")
    if model.n_heads % tp:
        valid = [t for t in (1, 2, 4, 8) if model.n_heads % t == 0]
        res.errors.append(
            f"{model.name} has {model.n_heads} attention heads, not divisible by tp={tp}; "
            f"valid tp: {valid}"
        )
    if tp > 1 and gpu.link_gbps < 100:
        res.warnings.append(
            f"tp={tp} over PCIe ({gpu.link_gbps} GB/s): all-reduce will dominate; "
            f"prefer more replicas"
        )

    # ── vLLM engine-arg consistency (vLLM refuses to start otherwise) ──
    max_len = eng.max_model_len or model.max_context
    if eng.max_model_len and eng.max_model_len > model.max_context:
        res.errors.append(
            f"max_model_len={eng.max_model_len:,} exceeds {model.name} context of "
            f"{model.max_context:,}"
        )
    if eng.max_num_batched_tokens < eng.max_num_seqs:
        res.errors.append(
            f"max_num_batched_tokens ({eng.max_num_batched_tokens}) must be >= "
            f"max_num_seqs ({eng.max_num_seqs})"
        )
    if not eng.enable_chunked_prefill and eng.max_num_batched_tokens < max_len:
        res.errors.append(
            f"with chunked prefill off, max_num_batched_tokens ({eng.max_num_batched_tokens:,}) "
            f"must be >= max_model_len ({max_len:,}) — enable chunked prefill or lower max_model_len"
        )
    if eng.speculative == "draft":
        drafts = [m for m in facts.family_members(model.family) if m.params_b < model.params_b / 4]
        if not drafts:
            res.errors.append(
                f"no draft model in family {model.family!r} small enough for {model.name}; "
                f"use speculative=ngram"
            )
    if eng.gpu_memory_utilization > 0.95:
        res.warnings.append("gpu_memory_utilization > 0.95 risks OOM from fragmentation")

    # ── memory ─────────────────────────────────────────────────────────
    mem = memory_plan(dep, gpu, model, prec)
    res.memory = mem
    if mem.weights_gb >= mem.budget_gb - FIXED_OVERHEAD_GB:
        res.errors.append(
            f"weights need {mem.weights_gb:.1f} GB/GPU but only "
            f"{mem.budget_gb - FIXED_OVERHEAD_GB:.1f} GB is usable — raise tensor_parallel, "
            f"pick a lower precision, or a bigger GPU"
        )
    elif mem.kv_capacity_tokens < max_len:
        # vLLM's startup check: the KV cache must hold one max-length sequence.
        res.errors.append(
            f"KV cache holds {mem.kv_capacity_tokens:,} tokens but max_model_len is "
            f"{max_len:,} — set max_model_len <= {mem.kv_capacity_tokens:,}, use fp8 KV, "
            f"or add memory"
        )

    # ── quality budget ─────────────────────────────────────────────────
    if ref is not None and ref.family != model.family:
        res.errors.append(
            f"{model.name} ({model.family}) is not a variant of {ref.name} ({ref.family}); "
            f"quality is not comparable"
        )
    res.quality_loss_pct = quality_loss_pct(model, prec, ref)
    if sla is not None and res.quality_loss_pct > sla.max_quality_loss_pct:
        res.errors.append(
            f"quality loss {res.quality_loss_pct:.2f}% exceeds SLA budget "
            f"{sla.max_quality_loss_pct:.2f}%"
        )

    res.ok = not res.errors
    return res


def to_vllm_command(dep: Deployment, facts: Facts | None = None) -> dict:
    """Render the deployment as a real `vllm serve` invocation plus run notes."""
    facts = facts or default_facts()
    f, eng = dep.fleet, dep.engine
    model, prec = facts.model(f.model), facts.precision(f.precision)

    args = [
        f"vllm serve {model.hf_id}",
        f"--tensor-parallel-size {f.tensor_parallel}",
        f"--max-num-seqs {eng.max_num_seqs}",
        f"--max-num-batched-tokens {eng.max_num_batched_tokens}",
        f"--gpu-memory-utilization {eng.gpu_memory_utilization}",
        f"--max-model-len {eng.max_model_len or model.max_context}",
        "--enable-chunked-prefill" if eng.enable_chunked_prefill else "--no-enable-chunked-prefill",
        "--enable-prefix-caching" if eng.enable_prefix_caching else "--no-enable-prefix-caching",
    ]
    notes = []
    if eng.kv_cache_dtype == "fp8":
        args.append("--kv-cache-dtype fp8")
    if prec.vllm_flag and prec.vllm_flag.startswith("--"):
        args.append(prec.vllm_flag.split(" (")[0])
    if prec.vllm_flag and "checkpoint" in prec.vllm_flag:
        notes.append(f"{f.precision}: point the command at a {prec.vllm_flag.strip('()')}")
    if eng.speculative != "off":
        spec = {"num_speculative_tokens": eng.num_speculative_tokens}
        if eng.speculative == "ngram":
            spec.update(method="ngram", prompt_lookup_max=4)
        else:
            drafts = [m for m in facts.family_members(model.family) if m.params_b < model.params_b / 4]
            spec["model"] = drafts[0].hf_id if drafts else "<draft-model>"
        args.append(f"--speculative-config '{json.dumps(spec)}'")
    if f.num_replicas > 1:
        notes.append(
            f"run {f.num_replicas} replicas behind a router using '{dep.routing.policy}'"
        )
    return {"command": " \\\n  ".join(args), "notes": notes}


def fit_table(
    model_name: str,
    facts: Facts | None = None,
    *,
    max_model_len: int = 8192,
    sla: SLA | None = None,
) -> list[dict]:
    """Every (GPU, precision, tp) that can host one replica of `model_name`.

    This is the seed of the fleet planner: it answers "where can this model run
    at all, how much KV room does it get, and what does one replica cost?"
    Sorted by $/hr per 1k KV tokens — a crude but honest capacity-per-dollar.
    """
    facts = facts or default_facts()
    rows = []
    for gpu in facts.gpus.values():
        for prec_name in facts.precisions:
            for tp in (1, 2, 4, 8):
                dep = Deployment(
                    fleet=FleetConfig(gpu=gpu.name, model=model_name, precision=prec_name,
                                      tensor_parallel=tp),
                )
                dep = dep.patched({"engine.max_model_len": max_model_len})
                fz = check(dep, facts, sla=sla, reference_model=model_name)
                if not fz.ok:
                    continue
                kv = fz.memory.kv_capacity_tokens
                rows.append({
                    "gpu": gpu.name, "precision": prec_name, "tp": tp,
                    "kv_tokens": kv,
                    "cost_per_hour": fz.cost_per_hour,
                    "usd_per_hr_per_1k_kv": fz.cost_per_hour / (kv / 1000),
                    "quality_loss_pct": fz.quality_loss_pct,
                })
                break  # smallest tp that fits is the one worth listing
    return sorted(rows, key=lambda r: r["usd_per_hr_per_1k_kv"])


def validate_priors(facts: Facts | None = None) -> list[str]:
    """Problems in priors.yaml rules: unknown traits, or patches the schema rejects."""
    facts = facts or default_facts()
    probe = Deployment(fleet=FleetConfig(gpu="H100-SXM", model="llama-3.1-8b"))
    problems = []
    for i, rule in enumerate(facts.prior_rules):
        tag = f"rule[{i}] {rule.get('statement', '')[:40]!r}"
        for p in validate_condition(rule.get("when") or {}, facts):
            problems.append(f"{tag}: {p}")
        patch = rule.get("patch") or {}
        if not patch:
            problems.append(f"{tag}: empty patch")
            continue
        if "fleet.precision" in patch and patch["fleet.precision"] not in facts.precisions:
            problems.append(f"{tag}: unknown precision {patch['fleet.precision']!r}")
        try:
            probe.patched(patch)
        except PatchError as e:
            problems.append(f"{tag}: {e}")
    return problems


__all__ = [
    "Feasibility", "MemoryPlan", "check", "memory_plan", "to_vllm_command",
    "fit_table", "validate_priors", "quality_loss_pct", "kv_bytes_per_token_per_gpu",
]
