# Agents/planner.py
"""
PlanningAgent — one LLM call per round that turns the classifier's soft prior,
the profiler metrics, the KnowledgeBase evidence and this session's history
into a single concrete plan for the CoderAgent to implement.

The classifier's preferred_strategies are a PRIOR, not a constraint. The plan
prompt hands them over with an explicit escape hatch: if the metrics or KB
evidence point elsewhere, the planner is told to choose "other" and justify it
in one sentence. A failed / unparseable plan degrades to strategy="agent_choice"
with an empty avoid list — i.e. the coder gets the same full freedom it had
before this agent existed.
"""
import json

from Agents.classifier import ALL_STRATEGIES, _extract_json


PLAN_PROMPT = """You are a CUDA optimization planner. Choose ONE strategy for the coder to
implement this round and describe exactly what to change.

Respond with ONLY a raw JSON object — no markdown, no code fences, no prose
before or after. EXACTLY these keys:
{
  "strategy": "a strategy name, or 'other'",
  "strategy_is_other": true or false,
  "other_justification": "one sentence (required and non-empty if strategy_is_other is true, else \\"\\")",
  "rationale": "1-2 sentences citing the metrics and/or KB evidence",
  "specific_changes": ["2 to 4 concrete code-level instructions"],
  "avoid": ["strategies or pitfalls to avoid this round"],
  "fallback": "one strategy name to try if this plan fails"
}

RULES:
- The classifier PREFERS these strategies: {preferred}. These are a prior, not
  a constraint. If the profiler metrics or KnowledgeBase evidence points
  elsewhere, choose "other" and justify it in one sentence in
  "other_justification".
- Do NOT repeat a strategy that already FAILED this session.
- Do NOT pick anything in the avoid list below.
- Prefer the single highest-impact change given the bottleneck.
"""


def _fmt_kb(kb_results) -> str:
    if not kb_results:
        return "  (no relevant KnowledgeBase entries)\n"
    lines = []
    for k in kb_results:
        lines.append(
            f"  - {k.get('strategy_used', '?')} "
            f"→ {k.get('speedup', 0)}x "
            f"→ insight: {k.get('insight', '?')} "
            f"| avoid_if: {k.get('avoid_if', '?')}"
        )
    return "\n".join(lines) + "\n"


def _fmt_history(history) -> str:
    if not history:
        return "  (this is the first round)\n"
    lines = []
    for h in history:
        strat = h.get("strategy", "?")
        res = h.get("result", "?")
        if res == "success":
            lines.append(f"  - {strat}: SUCCESS ({h.get('speedup', 0):.2f}x)")
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
    out = []
    for k in (kb_results or []):
        a = k.get("avoid_if")
        if a and a != "unknown" and a not in out:
            out.append(a)
    return out


class PlanningAgent:
    """Per-round strategy planner. Never raises into the pipeline — any failure
    degrades to a full-freedom 'agent_choice' plan."""

    def __init__(self, safe_chat_fn, runner, user_id: str, session_id: str):
        self.safe_chat = safe_chat_fn
        self.runner = runner
        self.user_id = user_id
        self.session_id = session_id

    def _fallback(self) -> dict:
        return {
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
    ) -> dict:
        classification = classification or {}
        preferred = classification.get("preferred_strategies", []) or []

        # Programmatic avoid list: strategies that already failed this session
        # plus every KB avoid_if hint. Merged with whatever the LLM returns.
        base_avoid = list(dict.fromkeys(_failed_strategies(history) + _kb_avoids(kb_results)))

        prompt = (
            PLAN_PROMPT.replace("{preferred}", json.dumps(preferred))
            + "\nKERNEL CLASSIFICATION:\n"
            + f"  kernel_type: {classification.get('kernel_type', 'unknown')}\n"
            + f"  bottleneck: {classification.get('bottleneck', 'unknown')}\n"
            + f"  confidence: {classification.get('confidence', 'unknown')}\n\n"
            + "PROFILER METRICS:\n"
            + f"  occupancy: {(metrics or {}).get('occupancy', 'n/a')}\n"
            + f"  compute_throughput: {(metrics or {}).get('compute_throughput', 'n/a')}\n"
            + f"  dram_throughput: {(metrics or {}).get('dram_throughput', 'n/a')}\n\n"
            + "KNOWLEDGEBASE EVIDENCE (strategy → speedup → insight → avoid_if):\n"
            + _fmt_kb(kb_results)
            + "\nTHIS SESSION'S HISTORY (strategy → outcome):\n"
            + _fmt_history(history)
            + f"\nCurrent best this session: "
            + (f"{best_speedup:.2f}x\n" if best_speedup else "none yet\n")
            + "\nAVOID LIST (do not pick any of these):\n"
            + ("  " + "; ".join(base_avoid) + "\n" if base_avoid else "  (empty)\n")
        )

        try:
            raw = await self.safe_chat(prompt, self.runner, self.user_id, self.session_id)
        except Exception as e:
            print(f"    [planner] LLM call failed: {e}")
            raw = ""

        data = _extract_json(raw)
        if not data:
            return self._fallback()

        strategy = data.get("strategy") or "agent_choice"
        strategy = str(strategy).strip()

        # A strategy is "other" if the LLM flagged it OR it isn't in the known
        # vocabulary. "agent_choice" (our own sentinel) counts as other too.
        is_other = bool(data.get("strategy_is_other")) or (strategy not in ALL_STRATEGIES)
        justification = str(data.get("other_justification") or "").strip()
        if is_other and not justification:
            justification = "planner selected an off-taxonomy strategy without justification"

        specific_changes = data.get("specific_changes")
        if not isinstance(specific_changes, list):
            specific_changes = []

        llm_avoid = data.get("avoid")
        if not isinstance(llm_avoid, list):
            llm_avoid = []
        avoid = list(dict.fromkeys(base_avoid + [str(a) for a in llm_avoid]))

        return {
            "strategy":            strategy,
            "strategy_is_other":   is_other,
            "other_justification": justification if is_other else "",
            "rationale":           str(data.get("rationale") or "no rationale provided")[:400],
            "specific_changes":    [str(c) for c in specific_changes][:4],
            "avoid":               avoid,
            "fallback":            str(data.get("fallback") or ""),
        }
