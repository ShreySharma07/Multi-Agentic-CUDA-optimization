# scripts/demo_kernel.py
"""
Showable demo: drop a KARMA-optimized kernel into a real nn.Module in place of
the PyTorch op, and prove it is (a) correct and (b) faster.

    python scripts/demo_kernel.py

No compiler needed — the kernel was already built, so this loads the cached
extension directly. Run it from any shell.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from karma_kernels import load

SHAPE = (64, 256, 256)
BATCH = 16


class KarmaLayerNorm(nn.Module):
    """Drop-in replacement for nn.LayerNorm using the KARMA kernel.

    Same signature, same shapes, same dtype -- it is an ordinary torch op, so it
    composes with the rest of a model exactly like the thing it replaces.
    """

    def __init__(self, normalized_shape):
        super().__init__()
        self._k = load("40_LayerNorm")
        self.ln = nn.LayerNorm(normalized_shape)   # holds weight/bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._k.forward(x)


def bench(fn, n: int = 30) -> tuple[float, float]:
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


def main() -> None:
    torch.manual_seed(0)
    x = torch.rand(BATCH, *SHAPE, device="cuda")

    torch_ln = nn.LayerNorm(SHAPE).cuda().eval()
    karma_ln = KarmaLayerNorm(SHAPE).cuda().eval()

    print(f"input: {tuple(x.shape)}  ({x.numel() * 4 / 1e6:.0f} MB fp32)\n")

    with torch.no_grad():
        y_torch = torch_ln(x)
        y_karma = karma_ln(x)

    same = torch.allclose(y_karma, y_torch, rtol=1e-4, atol=1e-5)
    diff = (y_karma - y_torch).abs().max().item()
    print(f"correctness : {'MATCH' if same else 'MISMATCH'}  (max abs diff {diff:.2e})")
    assert same, "kernel disagrees with PyTorch"

    t_ms, t_sd = bench(lambda: torch_ln(x))
    k_ms, k_sd = bench(lambda: karma_ln(x))

    # LayerNorm is memory-bound: two-pass moves read+read+write.
    traffic_gb = x.numel() * 4 * 3 / 1e9
    print()
    print(f"{'':12} {'ms':>9} {'+/-':>7} {'GB/s':>8}")
    print("-" * 40)
    for label, ms, sd in (("PyTorch", t_ms, t_sd), ("KARMA", k_ms, k_sd)):
        print(f"{label:12} {ms:9.3f} {sd:7.3f} {traffic_gb / (ms / 1000):8.1f}")
    print("-" * 40)
    print(f"{'speedup':12} {t_ms / k_ms:9.2f}x")


if __name__ == "__main__":
    main()
