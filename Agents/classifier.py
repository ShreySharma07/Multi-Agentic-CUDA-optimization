# Agents/classifier.py
"""
ClassifierAgent — one LLM call per kernel that labels it (kernel_type,
bottleneck, preferred_strategies, confidence) *before* the optimization loop
begins.

Design rule (important): the classification is a SOFT PRIOR, never a hard
filter. `preferred_strategies` is a preference signal handed to the
PlanningAgent, which is explicitly free to ignore it (see Agents/planner.py).
A failed / low-confidence classification degrades to "other" with an empty
strategy list, which means "planner has full freedom" — the pipeline must
behave exactly as it did before this agent existed. It can prune, never block.
"""
import csv
from pathlib import Path
from datetime import datetime

from Agents.json_utils import extract_json
from Agents.providers import LLMProvider


# ── Strategy taxonomy ──────────────────────────────────────────────────
# Per kernel_type, the strategies we PREFER to try. "other" is deliberately
# empty so the planner is unconstrained for kernels we can't confidently type.
STRATEGY_TAXONOMY = {
    "elementwise":  ["vectorized_loads_float4", "fast_math_intrinsics",
                     "grid_stride_loop", "occupancy_tuning"],
    "reduction":    ["warp_shuffle_reduction", "shared_memory_tree_reduction",
                     "vectorized_loads_float4", "multi_element_per_thread"],
    "matmul":       ["shared_memory_tiling", "register_blocking",
                     "vectorized_loads_float4", "loop_unrolling"],
    "attention":    ["online_softmax", "shared_memory_tiling",
                     "warp_shuffle_reduction"],
    "convolution":  ["shared_memory_tiling", "constant_memory_filters",
                     "loop_unrolling"],
    "other":        [],  # planner gets full freedom
}

VALID_TYPES = list(STRATEGY_TAXONOMY.keys())
VALID_BOTTLENECKS = ("memory-bound", "compute-bound", "unknown")
VALID_CONFIDENCE = ("high", "medium", "low")

# Flat vocabulary of every known strategy name (used to reject hallucinations).
ALL_STRATEGIES = sorted({s for v in STRATEGY_TAXONOMY.values() for s in v})

CLASSIFIER_LOG_CSV = "results/classifier_log.csv"
CLASSIFIER_LOG_FIELDS = [
    "timestamp", "kernel", "predicted_type", "predicted_bottleneck",
    "confidence", "manual_type", "match",
]


CLASSIFIER_PROMPT = """You are a CUDA kernel classifier. Given the CUDA source and its
profiler metrics, classify the kernel so a downstream planner can pick an
optimization strategy.

Respond with ONLY a raw JSON object — no markdown, no code fences, no prose
before or after. EXACTLY these keys:
{
  "kernel_type": one of ["elementwise","reduction","matmul","attention","convolution","other"],
  "bottleneck": one of ["memory-bound","compute-bound","unknown"],
  "preferred_strategies": [3 to 5 strategy names you would prefer to try],
  "confidence": one of ["high","medium","low"],
  "rationale": "one short sentence"
}

Guidance:
- kernel_type is about the COMPUTE PATTERN, not the variable names.
- If dram_throughput > compute_throughput the kernel is memory-bound.
- If you are not confident about the type, answer "other".
"""


def _safe_float(val, default=0.0) -> float:
    try:
        return float(str(val).replace("%", "").strip())
    except Exception:
        return default


def bottleneck_from_metrics(metrics: dict) -> str:
    """Metrics are authoritative: dram vs compute throughput decides the bound."""
    dram = _safe_float((metrics or {}).get("dram_throughput", 0))
    comp = _safe_float((metrics or {}).get("compute_throughput", 0))
    if dram == 0.0 and comp == 0.0:
        return "unknown"
    return "memory-bound" if dram > comp else "compute-bound"


def _clean_strategies(raw_list, kernel_type: str) -> list[str]:
    """
    Ground the LLM's suggested strategies against the known vocabulary and top
    up from the taxonomy so we always hand the planner 3-5 valid names — except
    for "other", which stays empty (full freedom).
    """
    if kernel_type == "other":
        return []

    canonical = STRATEGY_TAXONOMY.get(kernel_type, [])
    picked: list[str] = []
    if isinstance(raw_list, list):
        for s in raw_list:
            if isinstance(s, str) and s in ALL_STRATEGIES and s not in picked:
                picked.append(s)

    # top up to at least 3 from this type's canonical list
    for s in canonical:
        if len(picked) >= 3:
            break
        if s not in picked:
            picked.append(s)

    return picked[:5] if picked else list(canonical)


def append_classifier_log(kernel: str, classification: dict,
                          log_path: str = CLASSIFIER_LOG_CSV) -> None:
    """
    Append one row per kernel for offline accuracy scoring. `manual_type` and
    `match` are left blank on purpose — fill `manual_type` by hand for the
    KernelBench kernels, then run scripts/score_classifier.py.
    """
    Path("results").mkdir(exist_ok=True)
    path = Path(log_path)
    exists = path.exists()
    row = {
        "timestamp":            datetime.now().isoformat(),
        "kernel":               kernel,
        "predicted_type":       classification.get("kernel_type", ""),
        "predicted_bottleneck": classification.get("bottleneck", ""),
        "confidence":           classification.get("confidence", ""),
        "manual_type":          "",
        "match":                "",
    }
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CLASSIFIER_LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class ClassifierAgent:
    """One-shot kernel classifier. Never raises into the pipeline — any failure
    degrades to a low-confidence "other" classification (no pruning applied)."""

    def __init__(self, provider: LLMProvider, session_id: str = "classifier"):
        self.provider = provider
        self.session_id = session_id

    def _fallback(self, metrics: dict) -> dict:
        return {
            "kernel_type":          "other",
            "bottleneck":           bottleneck_from_metrics(metrics),
            "preferred_strategies": [],
            "confidence":           "low",
            "rationale":            "classification failed, no pruning applied",
        }

    async def classify(self, kernel_source: str, metrics: dict) -> dict:
        prompt = (
            CLASSIFIER_PROMPT
            + "\nPROFILER METRICS:\n"
            + f"  occupancy: {(metrics or {}).get('occupancy', 'n/a')}\n"
            + f"  compute_throughput: {(metrics or {}).get('compute_throughput', 'n/a')}\n"
            + f"  dram_throughput: {(metrics or {}).get('dram_throughput', 'n/a')}\n\n"
            + f"CUDA SOURCE (truncated):\n{(kernel_source or '')[:2500]}\n"
        )

        try:
            raw = await self.provider.complete(prompt, session_id=self.session_id)
        except Exception as e:
            print(f"    [classifier] LLM call failed: {e}")
            raw = ""

        data = extract_json(raw)
        if not data:
            return self._fallback(metrics)

        kernel_type = data.get("kernel_type")
        if kernel_type not in VALID_TYPES:
            kernel_type = "other"

        # Metrics win on bottleneck; fall back to the LLM's guess only when the
        # profiler gave us nothing (dram == compute == 0).
        bottleneck = bottleneck_from_metrics(metrics)
        if bottleneck == "unknown":
            llm_b = data.get("bottleneck")
            bottleneck = llm_b if llm_b in VALID_BOTTLENECKS else "unknown"

        confidence = data.get("confidence")
        if confidence not in VALID_CONFIDENCE:
            confidence = "medium"

        rationale = data.get("rationale") or "no rationale provided"

        return {
            "kernel_type":          kernel_type,
            "bottleneck":           bottleneck,
            "preferred_strategies": _clean_strategies(data.get("preferred_strategies"), kernel_type),
            "confidence":           confidence,
            "rationale":            str(rationale)[:300],
        }
