# karma_kernels.py
"""
Use a kernel that KARMA optimized, from your own code.

    import torch
    from karma_kernels import load

    ln = load("40_LayerNorm")          # a real torch extension module
    y = ln.forward(x)                  # torch.Tensor in, torch.Tensor out

That is the whole API. The optimized kernel IS a PyTorch operator: it takes and
returns torch tensors, runs on the current CUDA stream, and composes with the
rest of your model like any other op.

TWO LOAD PATHS, and the difference matters in practice:

  fast path  -- torch caches the compiled .pyd, which is an ordinary Python
                extension module. If it is already built we import it directly.
                Needs NOTHING but torch: no nvcc, no cl.exe, no VS prompt.
  build path -- first time only (or after the source changes). Compiles with
                torch.utils.cpp_extension, so it needs ninja + cl.exe, i.e. a
                Visual Studio Developer prompt on Windows. Takes ~50s.

So: build once from a Developer prompt, then use it anywhere.

CLI:
    python karma_kernels.py                 # list available kernels
    python karma_kernels.py 40_LayerNorm    # validate + benchmark vs PyTorch
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

RESULTS_DIR = Path(__file__).parent / "kernels" / "results"


def available() -> list[str]:
    """
    Kernels that are loadable as torch extensions.

    kernels/results/ also holds output from the LEGACY standalone path — those
    .cu files are whole programs with their own main() and CANNOT be imported as
    an extension. A torch-extension kernel is the one with a _binding.cpp
    companion, so that is the discriminator.
    """
    return sorted(
        p.stem for p in RESULTS_DIR.glob("*.cu")
        if not p.stem.endswith("_binding")
        and (RESULTS_DIR / f"{p.stem}_binding.cpp").exists()
    )


def legacy_standalone() -> list[str]:
    """Old standalone programs in kernels/results/ — compile with nvcc and run;
    they are not importable."""
    return sorted(
        p.stem for p in RESULTS_DIR.glob("*.cu")
        if not p.stem.endswith("_binding")
        and not (RESULTS_DIR / f"{p.stem}_binding.cpp").exists()
    )


def _sources(name: str) -> tuple[str, str]:
    cu = RESULTS_DIR / f"{name}.cu"
    cpp = RESULTS_DIR / f"{name}_binding.cpp"
    if not cu.exists():
        raise FileNotFoundError(
            f"No optimized kernel named {name!r}. Available: {available()}"
        )
    return cu.read_text(), (cpp.read_text() if cpp.exists() else "")


def _cached_pyd(cuda_src: str, cpp_src: str):
    """Import the already-built extension directly, bypassing the toolchain."""
    from torch.utils.cpp_extension import _get_build_directory

    from pipeline.torch_eval import _ensure_pybind, extension_name

    ext = extension_name(cuda_src, _ensure_pybind(cpp_src))
    build_dir = Path(_get_build_directory(ext, False))

    for pyd in build_dir.glob(f"{ext}*.pyd"):        # Windows
        spec = importlib.util.spec_from_file_location(ext, pyd)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    for so in build_dir.glob(f"{ext}*.so"):          # Linux
        spec = importlib.util.spec_from_file_location(ext, so)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None


def load(name: str, force_build: bool = False):
    """The optimized kernel as a torch extension module. `mod.forward(...)`."""
    cuda_src, cpp_src = _sources(name)

    if not force_build:
        try:
            mod = _cached_pyd(cuda_src, cpp_src)
            if mod is not None:
                return mod
        except Exception:
            pass  # fall through and build

    from pipeline.torch_eval import compile_extension

    mod, err = compile_extension(cuda_src, cpp_src)
    if mod is None:
        raise RuntimeError(
            f"Could not build {name!r}. On Windows this needs a Visual Studio "
            f"Developer prompt (cl.exe + ninja).\n\n{err[:600]}"
        )
    return mod


# ── demo ───────────────────────────────────────────────────────────────
def _demo(name: str) -> None:
    import statistics

    sys.path.insert(0, str(Path(__file__).parent))
    from pipeline.torch_eval import load_task, validate

    bench = Path(__file__).parent / "KernelBench/KernelBench/level1" / f"{name}.py"
    if not bench.exists():
        print(f"(no KernelBench task for {name}; loading the kernel only)")
        print("loaded:", load(name))
        return

    print(f"loading optimized kernel: {name}")
    mod = load(name)
    print(f"  -> {mod}\n")

    task = load_task(bench)
    ok, msg = validate(mod, task)
    print(f"correctness vs PyTorch: {'PASS' if ok else 'FAIL'}  ({msg})\n")
    if not ok:
        return

    eager, inputs = task.ref_model, task.inputs(42)

    def _time(fn, n=30):
        with torch.no_grad():
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            xs = []
            for _ in range(n):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); fn(); e.record()
                torch.cuda.synchronize()
                xs.append(s.elapsed_time(e))
        return statistics.fmean(xs), statistics.pstdev(xs)

    e_ms, e_sd = _time(lambda: eager(*inputs))
    k_ms, k_sd = _time(lambda: mod.forward(*inputs))

    print(f"{'PyTorch':10} {e_ms:8.3f} ms  (+/- {e_sd:.3f})")
    print(f"{'KARMA':10} {k_ms:8.3f} ms  (+/- {k_sd:.3f})")
    print(f"{'speedup':10} {e_ms / k_ms:8.2f}x\n")

    print("use it in your own code:")
    print("    from karma_kernels import load")
    print(f"    k = load({name!r})")
    print("    y = k.forward(x)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("torch-extension kernels (importable — use load()):\n")
        for n in available():
            print(f"  {n}")
        legacy = legacy_standalone()
        if legacy:
            print("\nlegacy standalone programs (nvcc + run; NOT importable):\n")
            for n in legacy:
                print(f"  {n}")
        print(f"\nusage: python {Path(__file__).name} <name>")
    else:
        _demo(sys.argv[1])
