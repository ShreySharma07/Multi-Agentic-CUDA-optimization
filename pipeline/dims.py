# pipeline/dims.py
"""
Extracts a problem-size fingerprint from generated CUDA source so an agent
can't silently shrink the workload (e.g. N_TEST) between the baseline and an
"optimized" round to fake a speedup. The prompt already instructs the agent
to preserve dimensions exactly; this makes that check enforced instead of
trusted.

Crucially this must distinguish PROBLEM dimensions (how much work there is --
N, N_TEST, BATCH, CHANNELS) from TUNING knobs (how the work is scheduled --
TILE_SIZE, BLOCK_DIM, WARP_SIZE, UNROLL_FACTOR). Changing a tuning knob is a
legitimate optimization -- it is literally half of the strategy taxonomy --
while changing a problem dimension invalidates the benchmark. An earlier
version of this file matched any identifier containing the letters "N" or
"SIZE", which classified TILE_SIZE as a problem dimension and therefore
rejected correct retiling optimizations.
"""
import re

# Tokens that mean "this constant sizes the PROBLEM".
DIM_TOKENS = {
    "N", "M", "K",
    "SIZE", "DIM", "DIMS",
    "WIDTH", "HEIGHT", "DEPTH",
    "ROW", "ROWS", "COL", "COLS",
    "BATCH", "CHANNEL", "CHANNELS",
    "LEN", "LENGTH", "NUM", "COUNT",
    "ELEMENTS", "SAMPLES", "FEATURES",
}

# Tokens that mean "this constant tunes the SCHEDULE". These win over
# DIM_TOKENS: TILE_SIZE contains SIZE but is a tuning knob, not a dimension.
TUNING_TOKENS = {
    "TILE", "BLOCK", "WARP", "THREAD", "THREADS", "GRID",
    "UNROLL", "COARSEN", "VEC", "VECTOR", "PAD", "PADDING",
    "SMEM", "SHARED", "STRIDE", "CHUNK", "FACTOR", "RADIUS",
    "PER",  # THREADS_PER_BLOCK
}

_DEFINE_RE = re.compile(r"#define\s+([A-Za-z_]\w*)\s+(\d+)\b")
# Capture the const/constexpr qualifier so we can tell a real constant from a
# loop induction variable: `for (int k = 0; ...)` also matches `int <name> = <int>;`
# and would otherwise register a phantom problem dimension named K.
_CONST_RE = re.compile(
    r"\b(const\s+|constexpr\s+)?(?:int|unsigned|size_t|long)\s+([A-Za-z_]\w*)\s*=\s*(\d+)\s*;"
)

# split camelCase / PascalCase into word boundaries before tokenizing
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokens(name: str) -> set[str]:
    """NUM_ELEMENTS -> {NUM, ELEMENTS};  numElements -> {NUM, ELEMENTS}."""
    spaced = _CAMEL_RE.sub("_", name)
    return {t for t in spaced.upper().split("_") if t}


def is_problem_dim(name: str) -> bool:
    """True iff `name` sizes the problem rather than tuning the schedule."""
    toks = _tokens(name)
    if toks & TUNING_TOKENS:
        return False
    return bool(toks & DIM_TOKENS)


def _is_constant_decl(qualifier: str, name: str) -> bool:
    """
    A real compile-time constant, not a loop variable. Either it is explicitly
    const/constexpr, or it is written in CONSTANT_CASE (no lowercase letters),
    which excludes the `i`/`j`/`k` of `for (int k = 0; ...)`.
    """
    return bool(qualifier) or name.isupper()


def extract_problem_dims(source: str) -> dict[str, int]:
    """{dimension_name: value} for problem-size constants only (no tuning knobs)."""
    dims: dict[str, int] = {}

    for name, value in _DEFINE_RE.findall(source):
        if is_problem_dim(name):
            dims[name.upper()] = int(value)

    for qualifier, name, value in _CONST_RE.findall(source):
        if _is_constant_decl(qualifier, name) and is_problem_dim(name):
            dims[name.upper()] = int(value)

    return dims


def dims_match(baseline_src: str, candidate_src: str) -> tuple[bool, str]:
    """
    Reject a candidate that changed the amount of work.

    Two failure modes are caught:
      1. A shared problem-size constant whose value changed (the classic
         "shrink N_TEST to look faster").
      2. Every baseline problem-size constant vanished from the candidate --
         i.e. they were renamed or inlined, which defeats check (1) silently.

    Tuning knobs (TILE_SIZE, BLOCK_DIM, ...) are ignored entirely; changing
    them is the whole point of the optimizer.
    """
    base_dims = extract_problem_dims(baseline_src)
    cand_dims = extract_problem_dims(candidate_src)

    if not base_dims:
        return True, ""  # nothing to enforce

    shared = set(base_dims) & set(cand_dims)
    if not shared:
        return False, (
            "all problem-size constants disappeared from the candidate "
            f"(baseline had {sorted(base_dims)}); they must keep their names and values"
        )

    mismatches = [
        f"{name}: baseline={base_dims[name]} candidate={cand_dims[name]}"
        for name in sorted(shared)
        if cand_dims[name] != base_dims[name]
    ]
    return (len(mismatches) == 0), "; ".join(mismatches)
