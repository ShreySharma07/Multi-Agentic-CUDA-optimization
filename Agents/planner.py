# Agents/planner.py
"""
PlanningAgent — one LLM call per round that turns the classifier's soft prior,
the profiler metrics, the KnowledgeBase evidence, the optimization playbook and
this session's history into a concrete plan for the CoderAgent.

CUMULATIVE, NOT SUBSTITUTIVE
A fast kernel is a STACK of techniques applied together. An earlier version of
this agent chose exactly ONE strategy per round and the coder implemented it in
isolation, which meant round N+1 routinely destroyed what round N had bought
(measured: register blocking took a matmul to 4.0 TFLOPS, then a round that
"applied shared_memory_tiling" replaced it and fell back to 0.9). So a plan now
carries:
  add_techniques  -- 1..3 techniques to ADD this round
  keep_techniques -- everything already in the kernel, which must survive
and the coder is told the change is additive.

SOFT PRIOR, NOT A CAGE
The classifier's preferred_strategies and the playbook are both priors. The
planner is explicitly free to choose techniques that appear in neither, and to
restructure the algorithm itself; it just has to say why. Measured speedup is the
arbiter. A failed / unparseable plan degrades to "agent_choice" with an empty
avoid list -- i.e. the coder gets the same freedom it had before this agent
existed, and the pipeline is never blocked.
"""
import json

from Agents.classifier import ALL_STRATEGIES
from Agents.json_utils import extract_json
from Agents.playbook import format_playbook
from Agents.providers import LLMProvider


PLAN_PROMPT = """You are a CUDA optimization planner. Decide what to ADD to this kernel
this round, and describe exactly what to change.

Respond with ONLY a raw JSON object — no markdown, no code fences, no prose
before or after. EXACTLY these keys:
{
  "add_techniques": ["1 to 3 technique names to ADD this round"],
  "keep_techniques": ["every technique already in the kernel that must be preserved"],
  "strategy": "short name for this round's main technique (the headline of add_techniques)",
  "strategy_is_other": true or false,
  "other_justification": "one sentence (required and non-empty if strategy_is_other, else \\"\\")",
  "rationale": "1-2 sentences citing the metrics and/or KB evidence",
  "specific_changes": ["2 to 5 concrete code-level instructions"],
  "avoid": ["techniques or pitfalls to avoid this round"],
  "fallback": "one technique to try instead if this plan fails"
}

RULES:
- The change is CUMULATIVE. Everything in keep_techniques stays in the kernel;
  add_techniques is layered ON TOP. Never propose replacing a technique that is
  already working unless you explicitly say it is being subsumed and why.
- Techniques compose. Prefer adding a technique that STACKS with what is already
  there over swapping in an unrelated one.
- Respect prerequisites: do not add a technique whose "needs" are not yet present.
- Do NOT repeat a technique that already FAILED this session.
- Do NOT pick anything in the avoid list below.
- set strategy_is_other=true if your headline technique is not a named playbook
  technique (a novel combination, or an algorithmic restructuring). That is
  allowed and sometimes correct — just justify it.
"""


def _fmt_kb(kb_results) -> str:
    if not kb_results:
        return "  (no relevant KnowledgeBase entries)\n"
    lines = []
    for k in kb_results:
        outcome = k.get("result", "?")
        try:
            sp = float(k.get("speedup", 0) or 0)
        except (TypeError, ValueError):
            sp = 0.0
        techs = str(k.get("techniques", "")).strip() or "(unrecorded)"
        # State the outcome and the technique IDs plainly. A failed round used to be
        # listed under "Relevant past optimizations" with no indication it had failed.
        lines.append(
            f"  - [{outcome}, {sp:.2f}x vs PyTorch] techniques: {techs}\n"
            f"      insight: {k.get('insight', '?')}"
        )
    return "\n".join(lines) + "\n"


def _fmt_history(history) -> str:
    if not history:
        return "  (this is the first optimization round)\n"
    lines = []
    for h in history:
        strat = h.get("strategy", "?")
        res = h.get("result", "?")
        if res == "success":
            lines.append(f"  - {strat}: SUCCESS ({h.get('speedup', 0):.2f}x vs eager)")
        else:
            lines.append(f"  - {strat}: FAILED ({res})")
    return "\n".join(lines) + "\n"


def _failed_strategies(history) -> list[str]:
    out = []
    for h in history:
        if h.get("result") != "success":
            s = h.get("strategy")
            if s and s not in ("agent_choice", "?") and s not in out:
                out.append(s)
    return out


def _kb_avoids(kb_results) -> list[str]:
    """
    TECHNIQUE IDs the KB says failed — and nothing else.

    This list is injected under a prompt line that says "Do NOT pick any of these",
    so only things that ARE pickable belong in it. It used to be filled with the
    reflector's free-text `avoid_if` sentences, which meant the planner was being
    told not to pick a paragraph — and one of those paragraphs advised against the
    very technique stack that had reached 98% of cuBLAS. Prose caveats are still
    surfaced, but as advisory context (see _kb_cautions), not as hard constraints.
    """
    out = []
    for k in (kb_results or []):
        if k.get("result") == "success":
            continue  # a technique that WORKED is not a thing to avoid
        for t in str(k.get("techniques", "")).split(","):
            t = t.strip()
            if t and t in ALL_STRATEGIES and t not in out:
                out.append(t)
    return out


def _kb_cautions(kb_results) -> list[str]:
    """The reflector's prose `avoid_if` notes — advisory context, never a hard rule."""
    out = []
    for k in (kb_results or []):
        a = str(k.get("avoid_if", "")).strip()
        if a and a.lower() != "unknown" and a not in out:
            out.append(a)
    return out


class PlanningAgent:
    """Per-round planner. Never raises into the pipeline — any failure degrades to
    a full-freedom 'agent_choice' plan."""

    def __init__(self, provider: LLMProvider, session_id: str = "planner"):
        self.provider = provider
        self.session_id = session_id

    def _fallback(self, applied: list[str] | None = None) -> dict:
        return {
            "add_techniques":      [],
            "keep_techniques":     list(applied or []),
            "strategy":            "agent_choice",
            "strategy_is_other":   True,
            "other_justification": "planner failed",
            "rationale":           "planner failed, coder has full freedom",
            "specific_changes":    [],
            "avoid":               [],
            "fallback":            "",
        }

    async def plan(
        self,
        classification: dict,
        metrics: dict,
        kb_results: list[dict],
        history: list[dict],
        best_speedup: float | None,
        applied: list[str] | None = None,
        cc: int | None = None,
    ) -> dict:
        classification = classification or {}
        applied = list(applied or [])
        preferred = classification.get("preferred_strategies", []) or []
        kernel_type = classification.get("kernel_type", "other")
        bottleneck = classification.get("bottleneck", "unknown")

        base_avoid = list(dict.fromkeys(_failed_strategies(history) + _kb_avoids(kb_results)))

        prompt = (
            PLAN_PROMPT
            + "\n" + format_playbook(kernel_type, bottleneck, applied, cc)
            + "\nKERNEL CLASSIFICATION (a prior — the metrics and your own reading may override it):\n"
            + f"  kernel_type: {kernel_type}\n"
            + f"  bottleneck: {bottleneck}\n"
            + f"  confidence: {classification.get('confidence', 'unknown')}\n"
            + f"  classifier prefers: {json.dumps(preferred)}\n\n"
            + "PROFILER METRICS (Nsight Compute):\n"
            + f"  occupancy: {(metrics or {}).get('occupancy', 'n/a')}%\n"
            + f"  compute_throughput: {(metrics or {}).get('compute_throughput', 'n/a')}%\n"
            + f"  dram_throughput: {(metrics or {}).get('dram_throughput', 'n/a')}%\n\n"
            + "KNOWLEDGEBASE — past rounds on similar kernels. Speedups are vs PyTorch\n"
            + "eager (cuBLAS/cuDNN), so ~1.0x means matching a hand-tuned vendor library:\n"
            + _fmt_kb(kb_results)
            + "\nTHIS SESSION'S HISTORY (technique → outcome):\n"
            + _fmt_history(history)
            + "\nCurrent best: "
            + (f"{best_speedup:.2f}x vs PyTorch eager\n" if best_speedup else "none yet\n")
            + "\nAVOID LIST — techniques that FAILED before. Do not pick any of these:\n"
            + ("  " + "; ".join(base_avoid) + "\n" if base_avoid else "  (empty)\n")
            + "\nCAUTIONS from past reflections — advisory only, NOT hard rules. Weigh them\n"
            + "against the metrics; if the evidence says otherwise, ignore them:\n"
            + ("".join(f"  ~ {c}\n" for c in _kb_cautions(kb_results)) or "  (none)\n")
        )

        try:
            raw = await self.provider.complete(prompt, session_id=self.session_id)
        except Exception as e:
            print(f"    [planner] LLM call failed: {e}")
            raw = ""

        data = extract_json(raw)
        if not data:
            return self._fallback(applied)

        add = data.get("add_techniques")
        if not isinstance(add, list):
            add = []
        add = [str(t).strip() for t in add if str(t).strip()][:3]

        keep = data.get("keep_techniques")
        if not isinstance(keep, list):
            keep = []
        # Everything already applied must survive, whatever the model said.
        keep = list(dict.fromkeys([str(t).strip() for t in keep if str(t).strip()] + applied))

        strategy = str(data.get("strategy") or (add[0] if add else "agent_choice")).strip()

        is_other = bool(data.get("strategy_is_other")) or (strategy not in ALL_STRATEGIES)
        justification = str(data.get("other_justification") or "").strip()
        if is_other and not justification:
            justification = "planner selected an off-playbook technique without justification"

        changes = data.get("specific_changes")
        if not isinstance(changes, list):
            changes = []

        llm_avoid = data.get("avoid")
        if not isinstance(llm_avoid, list):
            llm_avoid = []
        avoid = list(dict.fromkeys(base_avoid + [str(a) for a in llm_avoid]))

        return {
            "add_techniques":      add,
            "keep_techniques":     keep,
            "strategy":            strategy,
            "strategy_is_other":   is_other,
            "other_justification": justification if is_other else "",
            "rationale":           str(data.get("rationale") or "no rationale provided")[:400],
            "specific_changes":    [str(c) for c in changes][:5],
            "avoid":               avoid,
            "fallback":            str(data.get("fallback") or ""),
        }
