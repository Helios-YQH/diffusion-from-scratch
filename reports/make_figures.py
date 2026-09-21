"""Regenerate the report figures as vector PDFs (PNG copies are throwaway
previews for eyeballing in an image viewer).

Data sources (all produced by the scripts in this repo):
  * ../results/fid_<run>.json       FID / NFE / throughput sweeps  (eval/fid.py)
  * ../results/systems_<cfg>.json   step time, memory, MFU, profile (eval/systems.py)
  * ../samples/<run>_epoch<N>.png   EMA sample grids               (main.py train ...)

Run this where the data lives (sync ../results and ../samples first if you are
on another machine — figures generated against stale inputs are how the first
draft of this file shipped a CelebA grid under a caption about MNIST cells).

Sizes follow the NeurIPS layout conventions used in the CS336 reports: 5.5in
full text width, 8pt type, STIX fonts to match the Times body text.

Usage (from the repo root):
    uv run python reports/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
SRC = ROOT.parent
RESULTS = SRC / "results"
SAMPLES = SRC / "samples"

FULL_W, COL_W = 5.5, 2.65

# Okabe-Ito colour-blind-safe palette (same as the CS336 reports)
BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"
PURPLE, SKY, GREY, LIGHTGREY = "#CC79A7", "#56B4E9", "#8C8C8C", "#BFBFBF"

# One fixed colour and label per cell, shared by every figure.
CELL_STYLE = {
    "mnist_unet_eps": ("A: UNet $+$ $\\epsilon$", BLUE, "o"),
    "mnist_dit_eps": ("B: DiT $+$ $\\epsilon$", ORANGE, "s"),
    "mnist_unet_rf": ("C: UNet $+$ flow", GREEN, "^"),
    "mnist_dit_rf": ("D: DiT $+$ flow", RED, "v"),
    "ddpm_celeba": ("CelebA: UNet $+$ $\\epsilon$", PURPLE, "o"),
}
DATASET_OF = lambda run: "CelebA" if run.startswith("ddpm") else "MNIST"  # noqa: E731

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def add_panel_labels(axes, labels="ab"):
    """NeurIPS-style (a)/(b) panel labels, aligned by construction.

    Each label is anchored to its axes' top-left corner in axes fraction and
    then shifted by the *same* point offset — so labels line up across a row
    even when the panels' tick labels differ in width (the failure mode of
    hand-placed ax.text; see scipilot-figure-skill's layout_tools for the same
    approach).
    """
    for ax, lab in zip(axes, labels):
        ax.annotate(f"({lab})", xy=(0, 1), xycoords="axes fraction",
                    xytext=(-2, 3), textcoords="offset points",
                    fontsize=8.5, fontweight="bold", ha="right", va="bottom")


def save(fig, name):
    pdf = FIG / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"  wrote {pdf}")


def _load(pattern):
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(RESULTS.glob(pattern))]


def _best_points(run_json):
    """One entry per (sampler, nfe): the largest-n variant wins.

    A point can be measured at several sample counts (10k and 50k tiers);
    plotting both would zig-zag the curve with sampling noise.
    """
    best = {}
    for r in run_json.get("runs", []):
        key = (r["sampler"], r["nfe"])
        if key not in best or r["n"] > best[key]["n"]:
            best[key] = r
    return sorted(best.values(), key=lambda r: r["nfe"])


# ── Figure: the two probability paths, from the real schedule ─────────

def fig_paths():
    """Coefficients of the two paths, computed rather than sketched.

    Diffusion: sqrt(alpha_bar_t) from the linear beta schedule the code uses.
    Flow: 1 - t. Both are the actual signal coefficients on the path.
    """
    T, beta_start, beta_end = 1000, 1e-4, 0.02
    betas = np.linspace(beta_start, beta_end, T)
    alpha_bar = np.cumprod(1.0 - betas)
    t = np.linspace(0.0, 1.0, T)          # 0 = data, 1 = noise

    fig, ax = plt.subplots(figsize=(COL_W, 1.85))
    ax.plot(t, np.sqrt(alpha_bar), color=BLUE, lw=1.4,
            label=r"DDPM: $\sqrt{\bar\alpha_t}$")
    ax.plot(t, 1.0 - t, color=ORANGE, lw=1.4, ls="--",
            label=r"rectified flow: $1-t$")
    ax.annotate("data", xy=(0, 1.0), xytext=(0.01, 1.04), color=GREY, fontsize=7)
    ax.annotate("noise", xy=(1, 0.0), xytext=(0.84, -0.07), color=GREY, fontsize=7)
    ax.set_xlabel("$t$  (0 = data, 1 = noise)")
    ax.set_ylabel("signal coefficient")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 1.1)
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)
    save(fig, "paths")


def _fid_panels(key, ylabel, logy, value_key="fid"):
    """Shared two-panel layout for the FID and throughput figures."""
    data = _load("fid_*.json")
    if not data:
        print(f"  [skip] {key}: no results/fid_*.json yet")
        return None
    groups = {"MNIST": [], "CelebA": []}
    for d in data:
        groups[DATASET_OF(d["run_name"])].append(d)

    titles = {"MNIST": "MNIST $28^2$", "CelebA": "CelebA $64^2$"}
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.15))
    for ax, (key_name, runs) in zip(axes, groups.items()):
        plotted = False
        for d in sorted(runs, key=lambda x: x["run_name"]):
            if d["run_name"] not in CELL_STYLE:
                continue          # e.g. the bf16 precision run: not a cell
            label, color, marker = CELL_STYLE[d["run_name"]]
            pts = [p for p in _best_points(d) if p.get(value_key)]
            if not pts:
                continue
            swept = [(p["nfe"], p[value_key]) for p in pts if p["sampler"] != "ancestral"]
            base = [(p["nfe"], p[value_key]) for p in pts if p["sampler"] == "ancestral"]
            if swept:
                ax.plot(*zip(*swept), marker=marker, ms=3.2, lw=1.2, color=color,
                        label=label)
                plotted = True
            if base:
                ax.plot(*zip(*base), marker="*", ms=8, ls="none", color=color)
        if plotted:
            # one shared entry for the reference point instead of one per cell
            ax.plot([], [], marker="*", ls="none", ms=8, color=GREY,
                    label="ancestral 1000 NFE")
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("NFE (function evaluations)")
        ax.set_ylabel(ylabel)
        ax.set_title(titles[key_name], fontsize=8)
        ax.legend(frameon=False, loc="best", handlelength=1.5, borderpad=0.2)
    add_panel_labels(axes)
    fig.tight_layout()
    return fig


def fig_fid_vs_nfe():
    fig = _fid_panels("fid_vs_nfe", "FID", logy=True)
    if fig:
        save(fig, "fid_vs_nfe")


def fig_throughput_vs_nfe():
    fig = _fid_panels("throughput_vs_nfe", "samples / s", logy=True,
                      value_key="sampling_img_s")
    if fig:
        save(fig, "throughput_vs_nfe")


# ── Figures from eval/systems.py output ───────────────────────────────

def fig_systems_settings():
    data = _load("systems_*.json")
    if not data:
        print("  [skip] systems_settings: no results/systems_*.json yet")
        return
    rows = [(r["setting"], r["step_time_s"] * 1e3, r.get("mfu"),
             r["peak_memory_gb"]) for r in data[0]["results"]]
    single = [r for r in rows if "ddp" not in r[0] and "fsdp" not in r[0]]
    multi = [r for r in rows if "ddp" in r[0] or "fsdp" in r[0]]
    single.sort(key=lambda x: x[1])
    multi.sort(key=lambda x: x[1])

    # The 4-GPU rows do 4x the work per step, so their baseline is DDP, not the
    # single-GPU fp32 row — comparing them across groups would be meaningless.
    base_single = next((r[1] for r in single if r[0].startswith("fp32")), None)
    base_multi = next((r[1] for r in multi if r[0].endswith("ddp")), None)

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    ys = list(range(len(single))) + [len(single) + 0.6 + i for i in range(len(multi))]
    names = [r[0] + ("  (4 GPU)" if r in multi else "") for r in single + multi]
    ms = [r[1] for r in single + multi]
    colors = [ORANGE if "bf16" in n_ else BLUE if n_.startswith("fp32") else GREY
              for n_ in names]
    bars = ax.barh(ys, ms, color=colors, height=0.62)
    for bar, y, (name, val, mfu, mem) in zip(bars, ys, single + multi):
        in_multi = (name, val, mfu, mem) in multi
        base = base_multi if in_multi else base_single
        speed = f"{base / val:.2f}$\\times$ vs {'DDP' if in_multi else 'fp32'}"
        txt = f"{val:.0f} ms   {speed}" + (f"   ({mem:.1f} GB)" if in_multi else "")
        ax.text(val * 1.02, y, txt, va="center", fontsize=6, color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels(names)
    ax.set_xlabel("ms / step (CelebA DiT, 160 images/GPU)")
    ax.set_xlim(0, max(ms) * 1.62)
    ax.set_ylim(max(ys) + 0.6, -0.6)
    save(fig, "systems_settings")


def fig_profile_breakdown():
    data = [d for d in _load("systems_*.json") if d.get("profile_fractions")]
    if not data:
        print("  [skip] profile_breakdown: no --profile run recorded yet")
        return
    fracs = data[0]["profile_fractions"]
    colors = {"matmul": BLUE, "attention": SKY, "elementwise": ORANGE,
              "norm": GREEN, "conv": RED, "other": LIGHTGREY}
    labels = sorted([k for k in fracs if fracs[k] > 0], key=lambda k: fracs[k])
    vals = [fracs[k] * 100 for k in labels]

    fig, ax = plt.subplots(figsize=(COL_W, 1.65))
    bars = ax.barh(labels, vals, color=[colors.get(k, GREY) for k in labels],
                   height=0.62)
    for bar, val in zip(bars, vals):
        ax.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                va="center", fontsize=6.5, color="#333333")
    ax.set_xlabel("% of GPU time (fp32 training step)")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.tick_params(axis="y", length=0)
    save(fig, "profile_breakdown")


# ── Sample grids ──────────────────────────────────────────────────────

def fig_samples():
    """A labelled 2x2 of the four MNIST cells' final grids (A/B/C/D order)."""
    from PIL import Image

    cells = [("mnist_unet_eps", "A: UNet $+$ $\\epsilon$"),
             ("mnist_dit_eps", "B: DiT $+$ $\\epsilon$"),
             ("mnist_unet_rf", "C: UNet $+$ flow"),
             ("mnist_dit_rf", "D: DiT $+$ flow")]
    grids = {}
    for run, _ in cells:
        cands = sorted(SAMPLES.glob(f"{run}_epoch*.png"),
                       key=lambda p: int(p.stem.split("_epoch")[1]))
        if cands:
            grids[run] = cands[-1]
    if len(grids) < 4:
        print(f"  [skip] samples: found grids for {sorted(grids)} "
              f"(need all four cells)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(FULL_W * 0.62, FULL_W * 0.62))
    for ax, (run, title) in zip(axes.flat, cells):
        ax.imshow(np.asarray(Image.open(grids[run])))
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    save(fig, "samples")
    print(f"  (from {', '.join(grids[r].name for r, _ in cells)})")


def main():
    print("Regenerating report figures...")
    fig_paths()
    fig_fid_vs_nfe()
    fig_throughput_vs_nfe()
    fig_systems_settings()
    fig_profile_breakdown()
    fig_samples()
    print("Done.")


if __name__ == "__main__":
    main()
