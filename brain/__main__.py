# brain/__main__.py
"""
KARMA brain CLI (milestone M1: facts + feasibility).

    python -m brain catalog                      GPUs, models, precisions + traits
    python -m brain fit --model llama-3.1-8b     where can this model run, per $
    python -m brain check --gpu L4 --model llama-3.1-8b --precision fp8 \
        --max-model-len 8192                     will vLLM start? memory + command
    python -m brain knobs                        the action space agents may patch
"""
from __future__ import annotations

import argparse
import json
import sys

from brain.facts import default_facts, gpu_traits, model_traits, ridge_point
from brain.feasibility import check, fit_table, to_vllm_command, validate_priors
from brain.schema import SLA, Deployment, EngineConfig, FleetConfig, PatchError, knob_catalog


def cmd_catalog(_args) -> int:
    facts = default_facts()
    print(f"{'GPU':<11}{'mem':>6}{'GB/s':>7}{'bf16':>7}{'fp8':>7}{'ridge':>7}{'$/hr':>7}  traits")
    for g in facts.gpus.values():
        t = gpu_traits(g)
        tags = f"bw={t['gpu.bandwidth_class']} mem={t['gpu.memory_class']} {t['gpu.interconnect']}"
        print(f"{g.name:<11}{g.memory_gb:>6.0f}{g.mem_bw_gbps:>7.0f}{g.bf16_tflops:>7.0f}"
              f"{(g.fp8_tflops or 0):>7.0f}{ridge_point(g):>7.0f}{g.price_per_hour:>7.2f}  {tags}")
    print(f"\n{'model':<15}{'params':>8}{'KV KB/tok':>11}{'ctx':>9}  traits")
    for m in facts.models.values():
        t = model_traits(m)
        print(f"{m.name:<15}{m.params_b:>7.1f}B{m.kv_bytes_per_token() / 1024:>11.0f}"
              f"{m.max_context:>9,}  size={t['model.size_class']} moe={t['model.moe']}")
    print(f"\n{'precision':<11}{'B/param':>8}{'compute':>9}{'loss%':>7}  requires")
    for p in facts.precisions.values():
        print(f"{p.name:<11}{p.weight_bytes:>8.2f}{p.compute:>9}{p.quality_loss_pct:>7.1f}"
              f"  {p.requires or '-'}")
    problems = validate_priors(facts)
    print(f"\npriors: {len(facts.prior_rules)} rules, {len(facts.gpu_notes)} GPU notes, "
          f"{len(problems)} problems")
    for p in problems:
        print(f"  ! {p}")
    return 1 if problems else 0


def _sla(args) -> SLA | None:
    if args.max_quality_loss is None:
        return None
    # Latency targets are not checked in M1 (no simulator yet); only quality is.
    return SLA(ttft_p99_ms=1e9, tpot_p99_ms=1e9, max_quality_loss_pct=args.max_quality_loss)


def cmd_fit(args) -> int:
    rows = fit_table(args.model, max_model_len=args.max_model_len, sla=_sla(args))
    if not rows:
        print(f"{args.model} fits nowhere at max_model_len={args.max_model_len}")
        return 1
    print(f"Where {args.model} can run (max_model_len={args.max_model_len:,}), "
          f"cheapest KV capacity first:\n")
    print(f"{'GPU':<11}{'prec':<11}{'tp':>3}{'KV tokens':>12}{'$/hr':>8}{'$/hr/1kKV':>11}{'loss%':>7}")
    for r in rows[: args.top]:
        print(f"{r['gpu']:<11}{r['precision']:<11}{r['tp']:>3}{r['kv_tokens']:>12,}"
              f"{r['cost_per_hour']:>8.2f}{r['usd_per_hr_per_1k_kv']:>11.4f}"
              f"{r['quality_loss_pct']:>7.2f}")
    return 0


def cmd_check(args) -> int:
    try:
        dep = Deployment(
            fleet=FleetConfig(gpu=args.gpu, model=args.model, precision=args.precision,
                              tensor_parallel=args.tp, num_replicas=args.replicas),
            engine=EngineConfig(
                max_num_seqs=args.max_num_seqs,
                max_num_batched_tokens=args.max_num_batched_tokens,
                enable_chunked_prefill=not args.no_chunked_prefill,
                gpu_memory_utilization=args.gpu_memory_utilization,
                kv_cache_dtype=args.kv_cache_dtype,
                max_model_len=args.max_model_len,
            ),
        )
    except Exception as e:  # pydantic ValidationError
        print(f"invalid deployment: {e}")
        return 2
    fz = check(dep, sla=_sla(args), reference_model=args.reference_model)
    print(fz.summary())
    if fz.ok:
        cmd = to_vllm_command(dep)
        print("\n" + cmd["command"])
        for n in cmd["notes"]:
            print(f"# {n}")
    return 0 if fz.ok else 1


def cmd_knobs(_args) -> int:
    print(json.dumps(knob_catalog(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m brain", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("catalog", help="list facts tier").set_defaults(fn=cmd_catalog)
    sub.add_parser("knobs", help="print the action space").set_defaults(fn=cmd_knobs)

    fit = sub.add_parser("fit", help="where can a model run")
    fit.add_argument("--model", required=True)
    fit.add_argument("--max-model-len", type=int, default=8192)
    fit.add_argument("--max-quality-loss", type=float, default=None)
    fit.add_argument("--top", type=int, default=15)
    fit.set_defaults(fn=cmd_fit)

    ck = sub.add_parser("check", help="feasibility of one deployment")
    ck.add_argument("--gpu", required=True)
    ck.add_argument("--model", required=True)
    ck.add_argument("--precision", default="bf16")
    ck.add_argument("--tp", type=int, default=1, choices=[1, 2, 4, 8])
    ck.add_argument("--replicas", type=int, default=1)
    ck.add_argument("--max-num-seqs", type=int, default=256)
    ck.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ck.add_argument("--no-chunked-prefill", action="store_true")
    ck.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ck.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8"])
    ck.add_argument("--max-model-len", type=int, default=None)
    ck.add_argument("--reference-model", default=None,
                    help="the model you'd serve in bf16; prices smaller-variant quality loss")
    ck.add_argument("--max-quality-loss", type=float, default=None)
    ck.set_defaults(fn=cmd_check)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (PatchError, ValueError) as e:
        print(f"error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
