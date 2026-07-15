# Agents/playbook.py
"""
CUDA optimization playbook — domain knowledge the planner reasons over.

THE PRINCIPLE THIS ENCODES
No single technique makes a kernel fast. A competitive kernel is a STACK: several
techniques applied together, each covering a different bottleneck (global traffic,
arithmetic intensity, latency hiding, bank conflicts, instruction scheduling).
Applying technique B *instead of* technique A therefore tends to lose whatever A
bought you. Optimizations must COMPOSE, and the planner's job is to grow a stack
across rounds, not to swap one technique for another.

TWO RULES THIS FILE MUST OBEY
1. Hardware-agnostic. Nothing here assumes a particular GPU. Techniques that need
   a minimum compute capability declare it (`min_cc`) and are filtered against
   whatever device is actually detected at run time. Tile sizes, unroll factors
   and block sizes are never prescribed as constants — they are things to SWEEP,
   because the right value differs per architecture, per problem size, and is
   exactly what the agent is here to discover.
2. A PRIOR, NOT A CAGE. This list is a starting reference, not an allow-list. The
   planner is explicitly told it may propose techniques that do not appear here
   and combine them freely; the KnowledgeBase and the measured speedup are the
   arbiters, not this file. Consistent with the classifier's soft-prior design:
   it can inform, never restrict.
"""
from __future__ import annotations

# ── Technique reference ────────────────────────────────────────────────
# bottleneck : regime it helps ("memory-bound" | "compute-bound" | "any")
# min_cc     : minimum compute capability x10 (80 = Ampere/sm_80). None = universal.
# stacks_with: techniques it is designed to be combined with
# requires   : should already be present for this to make sense
# pitfall    : how it most often goes wrong, or silently loses its benefit
PLAYBOOK: dict[str, dict] = {
    # ---- memory movement -------------------------------------------------
    "coalesced_global_access": {
        "what": "Make consecutive threads in a warp touch consecutive addresses so a warp's "
                "accesses collapse into the fewest possible memory transactions.",
        "when": "Any kernel whose global reads/writes are strided or transposed.",
        "bottleneck": "memory-bound", "min_cc": None,
        "stacks_with": ["vectorized_loads", "shared_memory_tiling"],
        "requires": [],
        "pitfall": "For a transposed operand, do not fix coalescing by transposing in global "
                   "memory — stage through shared memory instead.",
    },
    "vectorized_loads": {
        "what": "Move 64/128 bits per thread (float2/float4-style wide loads) instead of one "
                "scalar, cutting memory instruction count and improving bus utilisation.",
        "when": "Contiguous, suitably aligned data. Usually worth it once tiling exists.",
        "bottleneck": "memory-bound", "min_cc": None,
        "stacks_with": ["shared_memory_tiling", "register_blocking", "double_buffering"],
        "requires": [],
        "pitfall": "Needs alignment and a divisible length — guard the tail. The best vector "
                   "width is architecture- and dtype-dependent: sweep it, don't assume float4.",
    },
    "read_only_cache": {
        "what": "Mark never-written inputs `const T* __restrict__` (or use __ldg) so they route "
                "through the read-only path and the compiler can keep more in registers.",
        "when": "Inputs the kernel never writes.",
        "bottleneck": "memory-bound", "min_cc": None,
        "stacks_with": ["vectorized_loads", "register_blocking"],
        "requires": [],
        "pitfall": "__restrict__ is a promise of no aliasing. If in/out can overlap it is UB.",
    },
    "shared_memory_tiling": {
        "what": "Stage a tile of each operand in shared memory so each global element is read "
                "once per block rather than once per thread.",
        "when": "Any kernel with data reuse across threads in a block (matmul, conv, stencil).",
        "bottleneck": "memory-bound", "min_cc": None,
        "stacks_with": ["register_blocking", "vectorized_loads", "double_buffering",
                        "bank_conflict_avoidance", "warp_tiling"],
        "requires": [],
        "pitfall": "Tiling ALONE can be SLOWER than register blocking alone: if each thread still "
                   "computes a single output element you have added shared-memory traffic without "
                   "raising arithmetic intensity. Tiling pays off when COMBINED with register "
                   "blocking so each thread computes a micro-tile.",
    },
    "bank_conflict_avoidance": {
        "what": "Pad or swizzle the shared-memory tile so threads in a warp hit distinct banks; "
                "an n-way conflict serialises the access n times.",
        "when": "Any shared-memory tile accessed column-wise.",
        "bottleneck": "memory-bound", "min_cc": None,
        "stacks_with": ["shared_memory_tiling", "double_buffering"],
        "requires": ["shared_memory_tiling"],
        "pitfall": "Padding costs shared memory and can cut occupancy — check the smem budget.",
    },
    "double_buffering": {
        "what": "Prefetch the next tile into a second shared-memory buffer while computing on the "
                "current one, hiding global latency behind math.",
        "when": "A loop over tiles where load and math can overlap.",
        "bottleneck": "memory-bound", "min_cc": None,
        "stacks_with": ["shared_memory_tiling", "register_blocking", "async_copy"],
        "requires": ["shared_memory_tiling"],
        "pitfall": "Doubles shared-memory usage. A __syncthreads() in the wrong place serialises "
                   "exactly the overlap you were trying to create.",
    },
    "async_copy": {
        "what": "Asynchronous global->shared copy (cp.async / memcpy_async) that bypasses registers "
                "so the transfer proceeds in the background.",
        "when": "Double-buffered tile loads, on architectures that support it.",
        "bottleneck": "memory-bound", "min_cc": 80,
        "stacks_with": ["double_buffering", "shared_memory_tiling"],
        "requires": ["shared_memory_tiling"],
        "pitfall": "Requires commit/wait-group discipline; without it you read a tile that has not "
                   "landed. Falls back to a normal copy on older architectures.",
    },
    "grid_stride_loop": {
        "what": "Each thread handles several elements in a strided loop, decoupling grid size from "
                "problem size.",
        "when": "Elementwise / reduction kernels over large 1-D data.",
        "bottleneck": "memory-bound", "min_cc": None,
        "stacks_with": ["vectorized_loads", "multi_element_per_thread"],
        "requires": [],
        "pitfall": "Size the grid to fill the device (a multiple of SM count), not to the data.",
    },
    "multi_element_per_thread": {
        "what": "Thread coarsening: give each thread several elements to amortise index math, "
                "launch overhead and reduction tree depth.",
        "when": "Memory-bound elementwise and reductions.",
        "bottleneck": "any", "min_cc": None,
        "stacks_with": ["grid_stride_loop", "vectorized_loads", "warp_shuffle_reduction"],
        "requires": [],
        "pitfall": "Too much coarsening cuts parallelism and occupancy — sweep the factor.",
    },

    # ---- op-chain fusion (Level 2/3: a heavy op wrapped in cheap ops) -----
    "vendor_op_fused_epilogue": {
        "what": "For a CHAIN of ops dominated by one heavy op (gemm/conv), call the VENDOR "
                "library for that op -- at::linear / at::matmul / at::conv2d, or cuBLAS via "
                "at::cuda::getCurrentCUDABlasHandle() -- and fuse EVERY surrounding "
                "elementwise/reduction op (scale, bias, add, clamp, activation, logsumexp, "
                "norm, ...) into ONE custom CUDA kernel. Do NOT reimplement the heavy op.",
        "when": "A fused op-chain whose runtime is dominated by a gemm or conv -- i.e. most of "
                "KernelBench Level 2 and Level 3. This is essentially the ONLY way to win here.",
        "bottleneck": "any", "min_cc": None,
        "stacks_with": ["vectorized_loads", "fast_math_intrinsics", "warp_shuffle_reduction",
                        "shared_memory_tree_reduction"],
        "requires": [],
        "pitfall": "Your own gemm/conv CANNOT beat cuBLAS/cuDNN -- writing one (shared_memory_tiling "
                   "+ register_blocking on the heavy op) is the classic mistake here and loses badly. "
                   "The entire win is fusing the cheap epilogue so it costs no extra kernel launches "
                   "or DRAM round-trips. Pointless for a PURE heavy op with no surrounding ops (you'd "
                   "just match the vendor lib). Feed the vendor op's output straight into your fused "
                   "kernel without materialising an intermediate.",
    },

    # ---- compute / ILP ---------------------------------------------------
    "register_blocking": {
        "what": "Each thread computes a small micro-tile of the output with accumulators held in "
                "registers, so one shared-memory read feeds many FMAs. This is what raises "
                "arithmetic intensity.",
        "when": "Matmul and any kernel with 2-D output reuse.",
        "bottleneck": "compute-bound", "min_cc": None,
        "stacks_with": ["shared_memory_tiling", "warp_tiling", "vectorized_loads",
                        "double_buffering", "loop_unrolling"],
        "requires": [],
        "pitfall": "Register pressure: too large a micro-tile spills to local memory and collapses "
                   "performance. The right size is architecture-dependent — sweep it.",
    },
    "warp_tiling": {
        "what": "Three-level tile hierarchy (block tile -> warp tile -> thread tile) so each warp "
                "owns a contiguous sub-tile and its shared-memory reads are broadcast-friendly.",
        "when": "High-performance GEMM, once tiling and register blocking are in place.",
        "bottleneck": "compute-bound", "min_cc": None,
        "stacks_with": ["shared_memory_tiling", "register_blocking", "double_buffering",
                        "bank_conflict_avoidance"],
        "requires": ["shared_memory_tiling", "register_blocking"],
        "pitfall": "It organises micro-tiles; it does not replace them. Meaningless without "
                   "register blocking underneath.",
    },
    "loop_unrolling": {
        "what": "Unroll the inner loop so the compiler can schedule independent FMAs back to back "
                "and hide their latency.",
        "when": "Small fixed trip counts (the inner tile loop).",
        "bottleneck": "compute-bound", "min_cc": None,
        "stacks_with": ["register_blocking", "warp_tiling"],
        "requires": [],
        "pitfall": "Unrolling a large loop explodes code size and register use, cutting occupancy.",
    },
    "instruction_level_parallelism": {
        "what": "Keep several independent accumulators so the pipeline is never stalled on one "
                "dependency chain.",
        "when": "Reductions and dot-product inner loops.",
        "bottleneck": "compute-bound", "min_cc": None,
        "stacks_with": ["register_blocking", "loop_unrolling"],
        "requires": [],
        "pitfall": "Changes float summation order — fine within tolerance, but not bitwise equal.",
    },
    "fast_math_intrinsics": {
        "what": "Hardware approximations for transcendentals (__expf, __logf, __fdividef, ...), "
                "several times faster than the IEEE versions.",
        "when": "Transcendental-heavy elementwise kernels (GELU, softmax, sigmoid).",
        "bottleneck": "compute-bound", "min_cc": None,
        "stacks_with": ["vectorized_loads", "grid_stride_loop"],
        "requires": [],
        "pitfall": "Lower precision. Correctness is checked against PyTorch — if validation starts "
                   "failing after this change, this is the first suspect.",
    },
    "tensor_core_mma": {
        "what": "Use the matrix-multiply-accumulate units (WMMA / mma) instead of the scalar FMA "
                "pipeline.",
        "when": "GEMM-shaped math on architectures with MMA units, WHEN the task's dtype and "
                "precision policy permit it.",
        "bottleneck": "compute-bound", "min_cc": 70,
        "stacks_with": ["shared_memory_tiling", "double_buffering", "warp_tiling"],
        "requires": [],
        "pitfall": "Reduced-precision paths (TF32/FP16 accumulate) change numerics. The kernel is "
                   "validated against PyTorch at the task's tolerance — if the reference runs in "
                   "full precision, a reduced-precision MMA path will FAIL validation. Check the "
                   "dtype and tolerance before reaching for this.",
    },
    "minimize_divergence": {
        "what": "Restructure branches so threads in a warp take the same path; prefer predication "
                "or arithmetic (fmaxf) over if/else.",
        "when": "Masked kernels (triangular, ReLU, bounds checks).",
        "bottleneck": "compute-bound", "min_cc": None,
        "stacks_with": [],
        "requires": [],
        "pitfall": "Never `return` early in a block that still has a __syncthreads() ahead — that is "
                   "undefined behaviour and the classic cause of a wrong tiled kernel.",
    },
    "occupancy_tuning": {
        "what": "Tune block size, shared-memory use and launch bounds so enough warps are resident "
                "to hide latency.",
        "when": "Always worth a sweep, especially when measured occupancy is low.",
        "bottleneck": "any", "min_cc": None,
        "stacks_with": [],
        "requires": [],
        "pitfall": "Maximum occupancy is not the goal. A register-blocked GEMM deliberately trades "
                   "occupancy for ILP — do not shrink the micro-tile merely to raise it.",
    },

    # ---- reductions ------------------------------------------------------
    "warp_shuffle_reduction": {
        "what": "Reduce within a warp using shuffle instructions — registers only, no shared "
                "memory, no barrier.",
        "when": "The final intra-warp stage of any reduction.",
        "bottleneck": "any", "min_cc": None,
        "stacks_with": ["shared_memory_tree_reduction", "multi_element_per_thread"],
        "requires": [],
        "pitfall": "Use the _sync variants with a correct mask; implicit warp-synchronous behaviour "
                   "is not guaranteed on modern architectures.",
    },
    "shared_memory_tree_reduction": {
        "what": "Log-depth tree reduction in shared memory down to one value per block.",
        "when": "Block-level reduction above warp size.",
        "bottleneck": "any", "min_cc": None,
        "stacks_with": ["warp_shuffle_reduction", "multi_element_per_thread"],
        "requires": [],
        "pitfall": "Every thread must reach every __syncthreads() in the tree — no early return.",
    },
    "two_pass_reduction": {
        "what": "One kernel reduces to a partial per block; a second reduces the partials. Avoids "
                "global atomic contention.",
        "when": "Large reductions where atomics dominate.",
        "bottleneck": "memory-bound", "min_cc": None,
        "stacks_with": ["warp_shuffle_reduction", "multi_element_per_thread"],
        "requires": [],
        "pitfall": "Costs a second launch; only worth it when the reduction is genuinely large.",
    },
    "online_softmax": {
        "what": "Single-pass max+sum with a running rescale, instead of multiple passes over the row.",
        "when": "Softmax / attention rows too large to keep in registers.",
        "bottleneck": "memory-bound", "min_cc": None,
        "stacks_with": ["shared_memory_tiling", "warp_shuffle_reduction"],
        "requires": [],
        "pitfall": "The rescale must be applied to the accumulator too, not only the denominator.",
    },

    # ---- convolution -----------------------------------------------------
    "constant_memory_filters": {
        "what": "Place small uniform filter weights in constant memory so broadcast reads are served "
                "from the constant cache.",
        "when": "Convolution with small fixed filters that fit the constant-memory limit.",
        "bottleneck": "memory-bound", "min_cc": None,
        "stacks_with": ["shared_memory_tiling", "loop_unrolling"],
        "requires": [],
        "pitfall": "Limited capacity, and it only pays when the whole warp reads the SAME element.",
    },
    "im2col_gemm": {
        "what": "Lower convolution to a matmul, then apply the whole GEMM stack to it.",
        "when": "Convolutions where GEMM-class efficiency is the goal.",
        "bottleneck": "compute-bound", "min_cc": None,
        "stacks_with": ["shared_memory_tiling", "register_blocking", "warp_tiling"],
        "requires": [],
        "pitfall": "The im2col buffer costs memory and bandwidth; can lose to direct conv on small "
                   "inputs.",
    },
    "halo_tiling": {
        "what": "Load a tile plus its halo into shared memory so stencil neighbours are read once.",
        "when": "Direct convolution and stencils.",
        "bottleneck": "memory-bound", "min_cc": None,
        "stacks_with": ["shared_memory_tiling", "bank_conflict_avoidance", "loop_unrolling"],
        "requires": [],
        "pitfall": "Halo handling at block edges is the usual source of off-by-one bugs.",
    },
}


# ── Canonical stacks ───────────────────────────────────────────────────
# What a competitive kernel of this type applies TOGETHER, in dependency order.
# This is the antidote to swapping one technique per round. It is a suggested
# ordering, not a script: the planner may reorder, skip, or go outside it.
CANONICAL_STACKS: dict[str, list[str]] = {
    "matmul": [
        # If the matmul is embedded in a chain (Level 2/3), delegate it to cuBLAS
        # and fuse the epilogue -- that is the win. The write-your-own-gemm ladder
        # below only applies to a STANDALONE matmul, where it still cannot beat
        # cuBLAS but is the only thing to try.
        "vendor_op_fused_epilogue",
        "shared_memory_tiling",
        "register_blocking",
        "vectorized_loads",
        "bank_conflict_avoidance",
        "loop_unrolling",
        "warp_tiling",
        "double_buffering",
        "async_copy",
        "tensor_core_mma",
    ],
    "reduction": [
        "multi_element_per_thread",
        "vectorized_loads",
        "shared_memory_tree_reduction",
        "warp_shuffle_reduction",
        "instruction_level_parallelism",
    ],
    "elementwise": [
        "grid_stride_loop",
        "vectorized_loads",
        "read_only_cache",
        "fast_math_intrinsics",
        "occupancy_tuning",
    ],
    "attention": [
        "shared_memory_tiling",
        "online_softmax",
        "register_blocking",
        "warp_shuffle_reduction",
        "double_buffering",
    ],
    "convolution": [
        # Same as matmul: a conv embedded in a chain should be delegated to cuDNN
        # (at::conv2d/at::conv3d) with the epilogue fused; the direct-conv ladder
        # below is for a standalone conv.
        "vendor_op_fused_epilogue",
        "shared_memory_tiling",
        "halo_tiling",
        "constant_memory_filters",
        "vectorized_loads",
        "loop_unrolling",
        "register_blocking",
    ],
    "other": [],
}


def technique_names() -> list[str]:
    return sorted(PLAYBOOK)


def supported(name: str, cc: int | None) -> bool:
    """Is this technique available on a device of compute capability `cc` (e.g. 89)?"""
    t = PLAYBOOK.get(name)
    if not t:
        return True          # unknown / model-invented technique: never block it
    if t["min_cc"] is None or cc is None:
        return True
    return cc >= t["min_cc"]


def format_technique(name: str) -> str:
    t = PLAYBOOK.get(name)
    if not t:
        return f"- {name}: (not in the playbook — judge it on its merits)"
    lines = [f"- {name} [{t['bottleneck']}]",
             f"    what:    {t['what']}",
             f"    when:    {t['when']}"]
    if t["requires"]:
        lines.append(f"    needs:   {', '.join(t['requires'])} to be present already")
    if t["stacks_with"]:
        lines.append(f"    stacks:  {', '.join(t['stacks_with'])}")
    lines.append(f"    pitfall: {t['pitfall']}")
    return "\n".join(lines)


def format_playbook(
    kernel_type: str,
    bottleneck: str,
    applied: list[str] | None = None,
    cc: int | None = None,
) -> str:
    """
    Scenario-specific reference for the planner prompt: the canonical stack for this
    kernel type, what the kernel already has, what remains, and the full entry for
    each technique still on the table — filtered to what this device can actually run.

    Explicitly framed as a prior, not an allow-list.
    """
    applied = applied or []
    stack = [t for t in CANONICAL_STACKS.get(kernel_type, []) if supported(t, cc)]
    remaining = [t for t in stack if t not in applied]

    extras = [
        n for n, t in PLAYBOOK.items()
        if n not in stack and n not in applied
        and t["bottleneck"] in (bottleneck, "any")
        and supported(n, cc)
    ]

    out = ["CUDA OPTIMIZATION PLAYBOOK",
           "",
           "KEY IDEA: a fast kernel is a STACK of techniques applied TOGETHER, each covering a",
           "different bottleneck. Applying one technique INSTEAD OF another usually destroys what",
           "the previous one bought. Grow the stack; do not swap.",
           ""]

    if stack:
        out.append(f"Typical stack for a '{kernel_type}' kernel (a suggested order, not a script):")
        for i, t in enumerate(stack, 1):
            mark = "[APPLIED]" if t in applied else "[  todo ]"
            out.append(f"  {i}. {mark} {t}")
    else:
        out.append(f"No canonical stack for kernel_type '{kernel_type}' — reason from the "
                   f"techniques below and from the profiler metrics.")

    out += ["",
            f"ALREADY IN THE KERNEL: {', '.join(applied) if applied else '(nothing yet)'}",
            "",
            "TECHNIQUES STILL AVAILABLE (filtered to this device's capability):"]
    for t in remaining:
        out.append(format_technique(t))
    for t in extras[:6]:
        out.append(format_technique(t))

    out += ["",
            "THIS LIST IS NOT EXHAUSTIVE. It is a prior, not a menu. If the profiler metrics, the",
            "KnowledgeBase, or your own reading of the kernel point somewhere else — a technique",
            "not listed here, an unusual combination, a restructuring of the algorithm itself —",
            "propose it and say why. Measured speedup decides, not this file."]

    return "\n".join(out) + "\n"
