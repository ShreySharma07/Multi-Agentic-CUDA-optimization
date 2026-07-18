"""Render images/KARMA_README_Benchmarks.png — the benchmark figure in the README.

Two panels so a 3ms..1200ms latency spread stays legible:
  left  — absolute latency per backend (log scale)
  right — speedup vs PyTorch eager (normalised, 1.0 = eager baseline)

Numbers come from the torch-extension run-off (eager / torch.compile / KARMA
raced back-to-back on an RTX 4050 Laptop, sm_89). Edit BENCH and re-run.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# kernel -> (eager_ms, compile_ms | None, karma_ms)
BENCH = {
    "LayerNorm":            (8.31,    10.09,  6.23),
    "MinGPT GELU":          (28.25,   3.41,   3.14),
    "Matmul + Residual":    (26.17,   24.40,  23.84),
    "Conv2d + InstanceNorm": (1192.51, 609.25, 493.98),
}

EAGER   = "#9AA0A6"   # gray — the baseline
COMPILE = "#4285F4"   # blue — torch.compile
KARMA   = "#7C3AED"   # violet — ours, made to pop
TEXT    = "#1F2328"

kernels = list(BENCH)
eager   = [BENCH[k][0] for k in kernels]
compile = [BENCH[k][1] for k in kernels]
karma   = [BENCH[k][2] for k in kernels]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": TEXT,
    "axes.edgecolor": "#D0D7DE", "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT, "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6))
fig.suptitle("KARMA vs PyTorch eager and torch.compile",
             fontsize=15, fontweight="bold", y=0.99)

y = range(len(kernels))
h = 0.26

# ---- left: absolute latency, log x so tiny + huge kernels coexist ----
def barset(ax, offset, vals, color, label):
    ys = [i + offset for i in y]
    drawn = [(v if v is not None else 0) for v in vals]
    bars = ax.barh(ys, drawn, height=h, color=color, label=label, zorder=3)
    return bars, ys

for off, vals, color, label in [
    (h, eager, EAGER, "PyTorch eager"),
    (0, compile, COMPILE, "torch.compile"),
    (-h, karma, KARMA, "KARMA (ours)"),
]:
    bars, ys = barset(axL, off, vals, color, label)
    for yi, v in zip(ys, vals):
        if v is None:
            axL.text(1.05, yi, "n/a", va="center", ha="left", fontsize=8.5,
                     color="#8A9199", zorder=4)
        else:
            axL.text(v * 1.08, yi, f"{v:,.1f}", va="center", ha="left",
                     fontsize=8.5, color=TEXT, zorder=4)

axL.set_xscale("log")
axL.set_xlim(1, 3000)
axL.set_yticks([i for i in y])
axL.set_yticklabels(kernels)
axL.invert_yaxis()
axL.set_xlabel("Latency (ms, log scale) — lower is better")
axL.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
axL.grid(axis="x", color="#EAEEF2", zorder=0)
axL.set_title("Absolute latency", fontsize=12, pad=8)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# ---- right: speedup vs eager (normalised) ----
sp_compile = [(e / c if c else None) for e, c in zip(eager, compile)]
sp_karma   = [e / k for e, k in zip(eager, karma)]

for off, vals, color, label in [
    (h / 1.4, sp_compile, COMPILE, "torch.compile"),
    (-h / 1.4, sp_karma, KARMA, "KARMA (ours)"),
]:
    ys = [i + off for i in y]
    drawn = [(v if v is not None else 0) for v in vals]
    axR.barh(ys, drawn, height=h * 1.4, color=color, label=label, zorder=3)
    for yi, v in zip(ys, vals):
        if v is None:
            axR.text(1.05, yi, "n/a", va="center", ha="left", fontsize=8.5,
                     color="#8A9199", zorder=4)
        else:
            axR.text(v + 0.12, yi, f"{v:.2f}x", va="center", ha="left",
                     fontsize=9, fontweight="bold", color=TEXT, zorder=4)

axR.axvline(1.0, color="#C0392B", lw=1.3, ls="--", zorder=2)
axR.text(1.0, len(kernels) - 0.35, " eager = 1.0x", color="#C0392B",
         fontsize=8.5, va="bottom", ha="left")
axR.set_yticks([i for i in y])
axR.set_yticklabels([])
axR.invert_yaxis()
axR.set_xlim(0, max(sp_karma) * 1.22)
axR.set_xlabel("Speedup vs eager — higher is better")
axR.grid(axis="x", color="#EAEEF2", zorder=0)
axR.set_title("Speedup over PyTorch eager", fontsize=12, pad=8)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

handles, labels = axL.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.01), fontsize=10.5)

fig.tight_layout(rect=(0, 0.04, 1, 0.96))
out = Path(__file__).resolve().parents[1] / "images" / "KARMA_README_Benchmarks.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print("wrote", out)
