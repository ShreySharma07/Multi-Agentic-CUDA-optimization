# pipeline/dims.py
"""
Extracts a problem-size fingerprint from generated CUDA source so an agent
can't silently shrink the workload (e.g. N_TEST) between the baseline and an
"optimized" round to fake a speedup. The prompt already instructs the agent
to preserve dimensions exactly; this makes that check enforced instead of
trusted.
"""
import re

_DIM_NAME = r"\w*(?:N|SIZE|DIM|WIDTH|HEIGHT|ROWS?|COLS?|BATCH|CHANNELS?)\w*"

_DEFINE_RE = re.compile(rf"#define\s+({_DIM_NAME})\s+(\d+)", re.IGNORECASE)
_CONST_RE = re.compile(rf"\b(?:const\s+)?int\s+({_DIM_NAME})\s*=\s*(\d+)\s*;", re.IGNORECASE)


def extract_problem_dims(source: str) -> dict[str, int]:
    """Best-effort {dimension_name: value} map from #define / const-int declarations."""
    dims: dict[str, int] = {}
    for pattern in (_DEFINE_RE, _CONST_RE):
        for name, value in pattern.findall(source):
            dims[name.upper()] = int(value)
    return dims


def dims_match(baseline_src: str, candidate_src: str) -> tuple[bool, str]:
    """
    True iff every dimension constant present in BOTH sources agrees in value.

    Constants that only appear on one side (e.g. a renamed loop variable) are
    ignored -- this deliberately only catches the case that matters: a
    shared-named constant whose value changed, which is what happens when a
    kernel quietly shrinks N_TEST to look faster.
    """
    base_dims = extract_problem_dims(baseline_src)
    cand_dims = extract_problem_dims(candidate_src)

    mismatches = [
        f"{name}: baseline={base_dims[name]} candidate={cand_dims[name]}"
        for name in base_dims
        if name in cand_dims and cand_dims[name] != base_dims[name]
    ]
    return (len(mismatches) == 0), "; ".join(mismatches)
