"""M1 — facts tier, action space, feasibility. Runs on CPU, no GPU or API key."""
import pytest

from brain.__main__ import main as cli
from brain.facts import (
    CatalogError, TRAIT_VOCAB, default_facts, gpu_traits, model_traits, validate_condition,
)
from brain.feasibility import check, fit_table, to_vllm_command, validate_priors
from brain.schema import SLA, Deployment, EngineConfig, FleetConfig, PatchError, knob_catalog

FACTS = default_facts()


def dep(gpu="H100-SXM", model="llama-3.1-8b", precision="bf16", tp=1, replicas=1, **engine):
    engine.setdefault("max_model_len", 8192)
    return Deployment(
        fleet=FleetConfig(gpu=gpu, model=model, precision=precision,
                          tensor_parallel=tp, num_replicas=replicas),
        engine=EngineConfig(**engine),
    )


# ── catalog / facts ───────────────────────────────────────────────────

def test_catalog_loads_and_is_consistent():
    assert {"L4", "A100-80GB", "H100-SXM", "MI300X"} <= set(FACTS.gpus)
    for g in FACTS.gpus.values():
        assert g.memory_gb > 0 and g.mem_bw_gbps > 0 and g.price_per_hour > 0
        assert g.fp8 == (g.fp8_tflops is not None)
    for m in FACTS.models.values():
        assert m.hidden % m.head_dim == 0
        assert m.active_b <= m.params_b


def test_kv_bytes_per_token_matches_llama_8b():
    # 2 * 32 layers * 8 kv heads * 128 dim * 2 bytes = 128 KiB
    assert FACTS.model("llama-3.1-8b").kv_bytes_per_token() == 131072


def test_unknown_names_fail_loudly_with_choices():
    with pytest.raises(CatalogError, match="Known"):
        FACTS.gpu("H1OO")


def test_gpu_traits_classify_hardware():
    l4, h100 = gpu_traits(FACTS.gpu("L4")), gpu_traits(FACTS.gpu("H100-SXM"))
    assert l4["gpu.bandwidth_class"] == "low" and l4["gpu.interconnect"] == "pcie"
    assert h100["gpu.bandwidth_class"] == "high" and h100["gpu.interconnect"] == "nvlink"
    assert not gpu_traits(FACTS.gpu("A100-80GB"))["gpu.fp8"]
    assert model_traits(FACTS.model("mixtral-8x7b"))["model.moe"]
    for t in (l4, h100):
        assert set(t) <= set(TRAIT_VOCAB)


def test_priors_are_valid_and_every_gpu_has_a_note():
    assert validate_priors(FACTS) == []
    assert set(FACTS.gpu_notes) == set(FACTS.gpus)


def test_condition_validation_rejects_bad_traits():
    assert validate_condition({"gpu.bandwidth_class": "low"}, FACTS) == []
    assert validate_condition({"gpu.colour": "green"}, FACTS)
    assert validate_condition({"gpu.bandwidth_class": "ultra"}, FACTS)
    assert validate_condition({"gpu.name": "H1OO"}, FACTS)


# ── action space: hallucination guard ─────────────────────────────────

def test_patch_applies_without_mutating_original():
    base = dep()
    new = base.patched({"engine.max_num_seqs": 128, "routing.policy": "least_tokens"})
    assert new.engine.max_num_seqs == 128 and new.routing.policy == "least_tokens"
    assert base.engine.max_num_seqs == 256


def test_patch_scale_on_numeric_knob():
    assert dep().patched({"engine.max_num_seqs": {"scale": 0.5}}).engine.max_num_seqs == 128


@pytest.mark.parametrize("patch", [
    {"engine.max_batch_size": 64},               # hallucinated knob
    {"engine.max_num_seqs": -4},                 # out of range
    {"engine.kv_cache_dtype": "int3"},           # invalid enum
    {"fleet.tensor_parallel": 3},                # not a legal TP size
    {"gpu.name": "L4"},                          # wrong section
    {"engine.enable_chunked_prefill": {"scale": 2}},  # scale on a bool
])
def test_invalid_patches_are_rejected(patch):
    with pytest.raises(PatchError):
        dep().patched(patch)


def test_knob_catalog_exposes_bounds():
    knobs = knob_catalog()
    assert knobs["engine.max_num_seqs"]["maximum"] == 2048
    assert "fleet.gpu" in knobs and "routing.policy" in knobs


# ── feasibility: the startup failures vLLM would hit ──────────────────

def test_healthy_deployment_is_feasible():
    fz = check(dep())
    assert fz.ok, fz.errors
    assert fz.memory.kv_capacity_tokens > 100_000
    assert fz.cost_per_hour == pytest.approx(2.50)


def test_default_full_context_does_not_fit_on_l4():
    fz = check(dep(gpu="L4", max_model_len=None))
    assert not fz.ok
    assert any("max_model_len" in e for e in fz.errors)


def test_70b_needs_tensor_parallel_on_80gb():
    assert not check(dep(model="llama-3.1-70b")).ok
    assert check(dep(model="llama-3.1-70b", tp=4)).ok


def test_70b_fits_single_mi300x():
    assert check(dep(gpu="MI300X", model="llama-3.1-70b")).ok


def test_fp8_weights_need_fp8_hardware():
    fz = check(dep(gpu="A100-80GB", precision="fp8"))
    assert not fz.ok and any("fp8" in e for e in fz.errors)


def test_fp8_kv_on_ampere_is_warning_not_error():
    fz = check(dep(gpu="A100-80GB", kv_cache_dtype="fp8"))
    assert fz.ok and any("fp8 KV" in w for w in fz.warnings)


def test_fp8_kv_doubles_capacity():
    a = check(dep(kv_cache_dtype="auto")).memory.kv_capacity_tokens
    b = check(dep(kv_cache_dtype="fp8")).memory.kv_capacity_tokens
    assert b == pytest.approx(2 * a, rel=0.01)


def test_tp_must_divide_attention_heads():
    fz = check(dep(model="qwen2.5-7b", tp=8))  # 28 heads
    assert not fz.ok and any("not divisible" in e for e in fz.errors)


def test_tp_beyond_kv_heads_replicates_kv():
    # qwen2.5-7b has 4 KV heads: at tp=4 each GPU holds one; capacity math stays sane
    fz = check(dep(model="qwen2.5-7b", tp=4))
    assert fz.ok and fz.memory.kv_capacity_tokens > 0


def test_chunked_prefill_off_requires_big_token_budget():
    fz = check(dep(enable_chunked_prefill=False, max_num_batched_tokens=2048))
    assert not fz.ok and any("chunked prefill" in e for e in fz.errors)
    assert check(dep(enable_chunked_prefill=False, max_num_batched_tokens=8192)).ok


def test_batched_tokens_must_cover_max_num_seqs():
    assert not check(dep(max_num_seqs=512, max_num_batched_tokens=256)).ok


def test_max_model_len_cannot_exceed_model_context():
    assert not check(dep(model="qwen2.5-7b", max_model_len=65536)).ok


def test_draft_speculation_needs_small_family_member():
    assert check(dep(speculative="draft")).ok                     # llama has 1B
    assert not check(dep(model="qwen2.5-14b", speculative="draft")).ok  # smallest is 7B


def test_quality_budget_blocks_aggressive_quantization():
    sla = SLA(ttft_p99_ms=1000, tpot_p99_ms=50, max_quality_loss_pct=1.0)
    assert check(dep(precision="fp8"), sla=sla).ok
    assert not check(dep(precision="awq_int4"), sla=sla).ok


def test_smaller_variant_quality_is_priced_against_reference():
    sla = SLA(ttft_p99_ms=1000, tpot_p99_ms=50, max_quality_loss_pct=5.0)
    fz = check(dep(model="llama-3.2-3b"), sla=sla, reference_model="llama-3.1-8b")
    assert fz.quality_loss_pct > 5 and not fz.ok
    cross = check(dep(model="qwen2.5-7b"), reference_model="llama-3.1-8b")
    assert any("not comparable" in e for e in cross.errors)


def test_amd_rejects_cuda_only_formats():
    assert not check(dep(gpu="MI300X", precision="awq_int4")).ok


def test_replicas_scale_cost():
    assert check(dep(gpu="L4", replicas=4, precision="fp8")).cost_per_hour == pytest.approx(2.80)


# ── outputs ───────────────────────────────────────────────────────────

def test_vllm_command_renders_every_knob():
    d = dep(gpu="H100-SXM", precision="fp8", kv_cache_dtype="fp8", speculative="ngram",
            enable_prefix_caching=False, replicas=2)
    out = to_vllm_command(d)
    cmd = out["command"]
    for flag in ("vllm serve meta-llama/Llama-3.1-8B-Instruct", "--quantization fp8",
                 "--kv-cache-dtype fp8", "--no-enable-prefix-caching", '"method": "ngram"',
                 "--max-model-len 8192"):
        assert flag in cmd
    assert any("2 replicas" in n for n in out["notes"])


def test_fit_table_sorted_and_feasible():
    rows = fit_table("llama-3.1-70b", max_model_len=8192)
    assert rows
    costs = [r["usd_per_hr_per_1k_kv"] for r in rows]
    assert costs == sorted(costs)
    # 16 GB cards can only host 70B sharded (4x T4 with int4 fits: ~9.4 GB/GPU)
    assert all(r["tp"] >= 4 for r in rows if r["gpu"] == "T4")
    assert {"gpu": "MI300X", "tp": 1}.items() <= next(r for r in rows if r["gpu"] == "MI300X").items()


def test_cli_smoke(capsys):
    assert cli(["catalog"]) == 0
    assert cli(["fit", "--model", "llama-3.1-8b", "--top", "3"]) == 0
    assert cli(["check", "--gpu", "L4", "--model", "llama-3.1-8b"]) == 1
    assert cli(["check", "--gpu", "L4", "--model", "llama-3.1-8b", "--precision", "fp8",
                "--max-model-len", "8192"]) == 0
    assert "vllm serve" in capsys.readouterr().out
