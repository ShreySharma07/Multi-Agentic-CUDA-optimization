# pipeline/torch_eval.py
"""
PyTorch-extension evaluation (CUDA Agent / ByteDance methodology).

Python owns the inputs, the correctness oracle and the clock; the LLM writes
ONLY the kernel + pybind binding. This removes the structural cheats the old
standalone-.cu path had to police: the model can no longer shrink the problem,
weaken a tolerance, or grade itself against a reference it wrote. PyTorch is the
reference, and it is executed here, not described in a prompt.

WINDOWS: torch.utils.cpp_extension shells out to cl.exe, so this must be run
from a VS Developer prompt (or after vcvars64.bat). Two Windows-specific flags
are required and are set below:
  * -Xcompiler /Zc:preprocessor -- CUDA's CCCL headers hard-error on MSVC's
    traditional preprocessor.
A CUDA 13.x toolkit against a cu126 torch build is tolerated in practice.
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

# Correctness tolerances by dtype. float32 reductions reassociate on GPU, so a
# bit-exact match is not expected; fp16/bf16 need far more headroom.
DTYPE_TOLERANCES = {
    torch.float32: (1e-4, 1e-5),
    torch.float64: (1e-6, 1e-7),
    torch.float16: (1e-2, 1e-3),
    torch.bfloat16: (1e-2, 1e-3),
}
DEFAULT_TOL = (1e-3, 1e-4)

VALIDATION_SEEDS = (42, 123, 7)
BASELINE_SEED = 42


def gpu_arch() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def _cuda_flags() -> list[str]:
    return ["-O3", f"-arch={gpu_arch()}", "-Xcompiler", "/Zc:preprocessor"]


def _cpp_flags() -> list[str]:
    return ["/Zc:preprocessor"] if sys.platform == "win32" else []


# ── Task ───────────────────────────────────────────────────────────────
@dataclass
class Task:
    name: str
    path: Path
    source: str
    model_cls: type
    get_inputs: callable
    get_init_inputs: callable
    _ref_model: torch.nn.Module | None = field(default=None, repr=False)

    @property
    def ref_model(self) -> torch.nn.Module:
        """Lazily instantiate the reference model on the GPU."""
        if self._ref_model is None:
            init = self.get_init_inputs() or []
            self._ref_model = self.model_cls(*init).cuda().eval()
        return self._ref_model

    def inputs(self, seed: int = BASELINE_SEED) -> list[torch.Tensor]:
        """Deterministic inputs on the GPU for a given seed."""
        torch.manual_seed(seed)
        return [
            x.cuda() if isinstance(x, torch.Tensor) else x
            for x in self.get_inputs()
        ]

    @torch.no_grad()
    def expected(self, seed: int = BASELINE_SEED) -> torch.Tensor:
        return self.ref_model(*self.inputs(seed))


def load_task(py_file: Path) -> Task:
    """importlib-load a KernelBench task file."""
    py_file = Path(py_file)
    spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return Task(
        name=py_file.stem,
        path=py_file,
        source=py_file.read_text(),
        model_cls=mod.Model,
        get_inputs=mod.get_inputs,
        get_init_inputs=getattr(mod, "get_init_inputs", lambda: []),
    )


# ── Timing ─────────────────────────────────────────────────────────────
@torch.no_grad()
def _time_callable(fn, args, warmup: int, runs: int) -> tuple[float, float]:
    """(mean_ms, std_ms) via cuda events."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    samples = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    return statistics.fmean(samples), (statistics.pstdev(samples) if len(samples) > 1 else 0.0)


def torch_compile_available() -> bool:
    """torch.compile's inductor backend needs triton, which has no Windows wheel.
    Checked once so the caller can pick a sensible stop condition instead of
    silently never hitting a compile-relative target."""
    return importlib.util.find_spec("triton") is not None


def vram_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def probe_task(task: Task, headroom: float = 0.80) -> tuple[bool, str, float]:
    """
    Can this task be evaluated honestly on this GPU? Returns (ok, reason, peak_gb).

    Two failure modes, both fatal to a trustworthy measurement:
      * hard OOM -- cannot run at all;
      * peak > VRAM -- on Windows the WDDM driver silently oversubscribes into
        system RAM instead of failing, so the task *appears* to run while every
        timing actually measures PCIe paging. That is worse than an OOM, because
        it yields a plausible-looking number that means nothing.

    Called before any LLM work so an unusable task costs zero tokens.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    budget = vram_gb() * headroom

    try:
        inputs = task.inputs(BASELINE_SEED)
        with torch.no_grad():
            _ = task.ref_model(*inputs)
        torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, f"OOM: task does not fit in {vram_gb():.1f}GB VRAM", 0.0
    except Exception as e:
        torch.cuda.empty_cache()
        return False, f"{e.__class__.__name__}: {e}", 0.0

    peak = torch.cuda.max_memory_allocated() / 1e9
    del inputs
    torch.cuda.empty_cache()

    if peak > budget:
        return False, (
            f"peak {peak:.1f}GB exceeds {budget:.1f}GB usable of {vram_gb():.1f}GB VRAM — "
            f"would page to system RAM and every timing would measure PCIe, not the kernel"
        ), peak

    return True, "", peak


def measure_baselines(task: Task, warmup: int = 10, runs: int = 30) -> dict:
    """Time PyTorch eager and torch.compile. torch.compile is unavailable on
    Windows (no triton) and legitimately fails for some ops -- record that and
    carry on rather than killing the run.

    Also records peak memory. This matters on Windows: WDDM lets the driver
    oversubscribe VRAM into system RAM, so a task far larger than the card does
    not OOM -- it silently pages over PCIe and every timing taken from it
    measures bus bandwidth, not the kernel. Such a task is flagged `spilled` so
    its "speedup" is never mistaken for a real result.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    inputs = task.inputs(BASELINE_SEED)
    model = task.ref_model

    eager_ms, eager_std = _time_callable(model, inputs, warmup, runs)

    compile_ms = compile_std = None
    compile_note = ""
    if not torch_compile_available():
        compile_note = "triton not installed (no Windows wheel) — torch.compile baseline unavailable"
    else:
        try:
            compiled = torch.compile(model)
            compile_ms, compile_std = _time_callable(compiled, inputs, warmup, runs)
        except Exception as e:
            compile_note = f"torch.compile failed: {e.__class__.__name__}"

    if compile_note:
        print(f"    {compile_note}")

    with torch.no_grad():
        expected = model(*inputs)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    spilled = peak_gb > vram_gb()
    if spilled:
        print(f"    WARNING: peak {peak_gb:.1f}GB > {vram_gb():.1f}GB VRAM — paging to "
              f"system RAM; timings measure PCIe, not the kernel")

    return {
        "eager_ms": eager_ms,
        "eager_std": eager_std,
        "compile_ms": compile_ms,
        "compile_std": compile_std,
        "compile_note": compile_note,
        "peak_gb": peak_gb,
        "spilled": spilled,
        "expected": expected,
        "inputs": inputs,
    }


# ── Compilation ────────────────────────────────────────────────────────
def extension_name(cuda_source: str, cpp_source: str) -> str:
    """Content-addressed so torch's on-disk cache dedupes identical rebuilds."""
    digest = hashlib.sha256((cuda_source + cpp_source).encode("utf-8")).hexdigest()
    return f"karma_{digest[:12]}"


PYBIND_TEMPLATE = """
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("forward", &forward, "KARMA kernel forward");
}}
"""


def _ensure_pybind(cpp_source: str) -> str:
    """
    Without a PYBIND11_MODULE block the extension builds fine and then fails at
    import with a baffling 'does not define module export function' ImportError.
    If the model forgot it, append the standard one rather than surfacing that.
    """
    if "PYBIND11_MODULE" in cpp_source:
        return cpp_source
    return cpp_source.rstrip() + "\n" + PYBIND_TEMPLATE


def compile_extension(
    cuda_source: str, cpp_source: str, name: str | None = None, timeout_s: int = 180
) -> tuple[object | None, str]:
    """
    Build a CUDA extension. Returns (module, "") or (None, full_error_text).

    The error text is the compile-feedback payload handed back to the coder, so
    it is never truncated here. A hung cl.exe cannot stall the batch: the build
    runs on a worker thread with a hard timeout.
    """
    cpp_source = _ensure_pybind(cpp_source)
    name = name or extension_name(cuda_source, cpp_source)

    def _build():
        return load_inline(
            name=name,
            cpp_sources=[cpp_source],
            cuda_sources=[cuda_source],
            extra_cuda_cflags=_cuda_flags(),
            extra_cflags=_cpp_flags(),
            verbose=False,
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_build).result(timeout=timeout_s), ""
    except FuturesTimeout:
        return None, f"Compilation timed out after {timeout_s}s (hung compiler)."
    except Exception as e:
        return None, f"{e.__class__.__name__}: {e}"


# ── Correctness ────────────────────────────────────────────────────────
def _tol(t: torch.Tensor) -> tuple[float, float]:
    return DTYPE_TOLERANCES.get(t.dtype, DEFAULT_TOL)


def _as_tuple(x):
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)


@torch.no_grad()
def validate(mod, task: Task) -> tuple[bool, str]:
    """
    Gate A -- correctness against PyTorch on several seeds (a kernel that only
    works on one input distribution is not correct).
    Gate B -- determinism: same inputs twice must give bit-identical output. A
    race condition shows up here and nowhere else.

    Intermediates are freed between seeds: these tasks are large enough that
    holding three seeds' worth of inputs+outputs at once will OOM on a small card.
    """
    # Gate A
    for seed in VALIDATION_SEEDS:
        try:
            inputs = task.inputs(seed)
            expected = task.ref_model(*inputs)
            got = mod.forward(*inputs)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return False, f"Out of memory validating seed {seed} (task too large for this GPU)."
        except Exception as e:
            return False, f"Kernel raised on seed {seed}: {e.__class__.__name__}: {e}"

        for i, (g, e) in enumerate(zip(_as_tuple(got), _as_tuple(expected))):
            if not isinstance(g, torch.Tensor):
                return False, f"forward() returned {type(g).__name__}, expected Tensor (seed {seed})"
            if g.shape != e.shape:
                return False, f"Shape mismatch on seed {seed}: got {tuple(g.shape)}, expected {tuple(e.shape)}"

            rtol, atol = _tol(e)
            try:
                torch.testing.assert_close(g, e, rtol=rtol, atol=atol, check_dtype=False)
            except AssertionError:
                diff = (g.float() - e.float()).abs().max().item()
                return False, (
                    f"Incorrect on seed {seed} (output {i}): max abs diff {diff:.3e} "
                    f"exceeds tolerance rtol={rtol:g} atol={atol:g}."
                )

        del inputs, expected, got
        torch.cuda.empty_cache()

    # Gate B
    try:
        inputs = task.inputs(BASELINE_SEED)
        first = _as_tuple(mod.forward(*inputs))
        second = _as_tuple(mod.forward(*inputs))
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, "Out of memory during determinism check (task too large for this GPU)."

    for i, (a, b) in enumerate(zip(first, second)):
        if not torch.equal(a, b):
            diff = (a.float() - b.float()).abs().max().item()
            return False, (
                f"Nondeterministic kernel: two runs on identical inputs differ "
                f"(output {i}, max diff {diff:.3e}). Likely a race condition — every "
                f"thread in a block must reach every __syncthreads()."
            )

    del inputs, first, second
    torch.cuda.empty_cache()
    return True, "Correct on 3 seeds and deterministic."


# ── Benchmark ──────────────────────────────────────────────────────────
@torch.no_grad()
def benchmark_interleaved(
    mod, task: Task, warmup: int = 10, runs: int = 30, block: int = 5
) -> dict:
    """
    Time the kernel against eager, interleaved in blocks so that thermal drift
    and clock boost affect both arms equally -- measuring one fully, then the
    other, silently attributes any drift to the second.
    """
    inputs = task.inputs(BASELINE_SEED)
    model = task.ref_model

    for _ in range(warmup):
        model(*inputs)
        mod.forward(*inputs)
    torch.cuda.synchronize()

    def _one(fn) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*inputs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    eager, karma = [], []
    while len(karma) < runs:
        for _ in range(min(block, runs - len(eager))):
            eager.append(_one(model))
        for _ in range(min(block, runs - len(karma))):
            karma.append(_one(mod.forward))

    karma_ms = statistics.fmean(karma)
    eager_ms = statistics.fmean(eager)

    if karma_ms <= 0:
        return {"karma_ms": 0.0, "karma_std": 0.0, "eager_ms": eager_ms,
                "eager_std": 0.0, "speedup_vs_eager": 0.0,
                "warning": "kernel timed at <=0ms"}

    return {
        "karma_ms": karma_ms,
        "karma_std": statistics.pstdev(karma) if len(karma) > 1 else 0.0,
        "eager_ms": eager_ms,
        "eager_std": statistics.pstdev(eager) if len(eager) > 1 else 0.0,
        "speedup_vs_eager": eager_ms / karma_ms,
        "warning": "",
    }
