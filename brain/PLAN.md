# KARMA Brain — Plan

The brain is the control plane that decides **how to serve a model on vLLM**:
which GPU, how many, which precision, which vLLM engine settings, and how to route
requests. It does not replace vLLM; it drives it. Over time it should learn enough
to make a vLLM fleet faster, more scalable and cheaper than a hand-tuned one.

> **Analogy.** vLLM is the engine. The brain is the race engineer: it reads the
> telemetry, changes the setup, watches the lap time and keeps a notebook of what
> worked on which track and which car. The notebook is what lets it eventually beat
> the factory settings.

## Design decisions (chosen)

| Decision | Choice | Why |
|---|---|---|
| How it decides | **Hybrid**: an LLM strategist (Gemini) proposes strategies, a numeric tuner refines values, and a simulator scores candidates | The LLM brings intuition and the tuner brings precision. Neither alone is enough |
| Feedback without a GPU | **Event-driven queue simulator + roofline step costs** | Models queueing, p99 latency, KV pressure, preemption and cost. Calibrated against real runs later |
| Knowledge base | **3 tiers + a learned predictor**: Facts → Experience → Playbook, plus a surrogate model | Facts are physics, experience is raw trials, the playbook holds distilled rules, and the predictor skips unpromising simulations |
| Self-evolution | **Rule lifecycle**: hypothesis → candidate → promoted, or demoted on contrary evidence | Protects against the memory poisoning the kernel KB hit earlier |
| Structure | **Hierarchical**: fleet (hours) → replica/engine (minutes) → request routing (milliseconds) | Each level acts at its natural speed |
| Objective | **Meet the SLA first, then minimize $/1M tokens**, reporting the Pareto front | Cost is the goal; latency and quality are hard constraints |
| GPU-awareness | Strategy is written over **traits** (e.g. `gpu.bandwidth_class=low`), not GPU names alone | Knowledge learned on an L4 transfers to any bandwidth-starved GPU |

## Architecture

```
                 ┌──────────────────── Brain ─────────────────────┐
 workload + SLA →│ Fleet planner ─→ Engine tuner ─→ Router choice │→ deployment
                 │      ↑   Strategist (Gemini) + Tuner     ↓     │   + vllm serve cmd
                 │      └──── Predictor ←── Simulator ──────┘     │
                 └───────────────────────┬────────────────────────┘
                     Knowledge base:  Facts │ Experience │ Playbook
```

## Milestones

| # | Milestone | Delivers | Exit criteria | Status |
|---|---|---|---|---|
| **M1** | **Facts tier + action space** | GPU/model/precision catalogs; GPU and model traits; per-GPU strategy priors; strict patch schema (hallucination guard); feasibility checks that mirror vLLM startup failures; memory/KV planner; quality budget; $/hr; `vllm serve` rendering; CLI `catalog`/`fit`/`check`/`knobs` | Catalog and priors validate; every vLLM startup failure covered has a test; 35/35 tests pass on CPU | ✅ **Complete** |
| M2 | Simulator | Workload generator (presets: chat, RAG long-context, code completion, batch summarize, bursty agentic); roofline step cost (compute/memory/TP comm/overhead, MoE, quantization); continuous-batching replica (chunked prefill, prefix cache, KV blocks, preemption, speculative decoding); multi-replica cluster with routing; p50/p99 TTFT/TPOT, goodput, $/1M tokens | Physical sanity tests pass: more replicas → lower TTFT; chunked prefill → lower TPOT p99 on long prompts; fp8 KV → fewer preemptions; deterministic per seed; a 60 s simulation finishes in under 2 s on CPU | ⬜ Not started |
| M3 | Objective + diagnosis | SLA-then-cost ranking; violation score for infeasible setups; Pareto front; bottleneck diagnosis (KV-bound, queueing, prefill interference, compute/memory/comm-bound, over-provisioned, load imbalance) with evidence | Ranking is a strict total order; each diagnosis is triggered by a crafted scenario in tests | ⬜ Not started |
| M4 | Experience + Playbook | SQLite experience store (context, deployment, metrics, source=sim/real); playbook with rule lifecycle driven by A/B ablations; priors from M1 loaded as hypotheses | A planted false prior is demoted; a true rule is promoted only after wins in ≥2 distinct contexts | ⬜ Not started |
| M5 | Predictor + Tuner | numpy kNN + ridge surrogate with uncertainty; numeric neighbourhood search screened by the predictor (exploit + explore); online prediction-error log | Predictor MAE falls as experience grows; the tuner reaches the same best with ≥30% fewer simulations than unscreened search | ⬜ Not started |
| M6 | Strategist + hierarchical orchestrator | Gemini strategist (via existing `Agents/providers`) with a heuristic fallback when no key is set; fleet planner (GPU × precision × variant × TP shortlist); engine loop; routing selection; baseline = vLLM defaults sized to meet the SLA; report (JSON + Markdown + `vllm serve`); `python -m brain optimize` | On all 5 workload presets the brain meets the SLA at lower $/hr than the defaults baseline, and the result holds on a held-out random seed | ⬜ Not started |
| M7 | Self-evolution + sim-to-real | `python -m brain evolve` curriculum across GPUs, models and workloads; tracks rules promoted, predictor error and simulations-to-best per session; calibration hook to ingest real vLLM benchmark runs (`source=real`) and correct per-GPU sim factors | Later sessions need measurably fewer simulations to reach their best; real measurements override simulated rules | ⬜ Not started |

## Edge cases designed for

| Edge case | Where handled |
|---|---|
| The LLM proposes a knob vLLM doesn't have, or an out-of-range value | M1 strict schema (`PatchError`) |
| vLLM refuses to start (KV cache can't hold one `max_model_len` sequence; batched tokens < max_num_seqs; chunked prefill off with a small token budget; TP not dividing heads) | M1 feasibility |
| A precision unsupported by the hardware (fp8 on Ampere, fp4 off Blackwell, AWQ on ROCm) | M1 feasibility |
| Quantization or a smaller variant silently costs accuracy | M1 quality budget vs SLA |
| Tensor parallel over PCIe | M1 warning; M2 comm cost |
| Mean looks fine but p99 breaks the SLA | M2 metrics, M3 objective |
| Prefill bursts stall decodes | M2 scheduler model |
| KV exhaustion → preemption → latency cliffs | M2 replica, M3 diagnosis |
| Bursty traffic and load imbalance | M2 workload/router |
| Overfitting to one random seed or one workload | M6 held-out seed; M4 multi-context promotion |
| A wrong prior or a lucky trial poisons memory | M4 lifecycle (evidence-based, demotion) |
| Sim-to-real gap | M4 `source` tag; M7 calibration; real data outranks simulated data |
| Price changes | Prices only live in `catalog/gpus.yaml` |

## Using M1 today

```
python -m brain catalog                                  # facts + traits + priors check
python -m brain fit --model llama-3.1-70b                # where it can run, per $
python -m brain check --gpu L4 --model llama-3.1-8b \
    --precision fp8 --kv-cache-dtype fp8 --max-model-len 8192 --replicas 3
python -m brain knobs                                    # action space shown to agents
python -m pytest tests/brain -q
```

## Known limits of M1

- Catalog numbers are approximate datasheet values (dense FLOPs) and approximate Oct 2026 marketplace prices. Edit `brain/catalog/*.yaml`.
- The memory model (activation factor, fixed overhead) is a calibrated guess, not a vLLM profile run. M7 calibration replaces it with measured values.
- Latency SLA fields are validated but not yet evaluated; that needs the M2 simulator.
