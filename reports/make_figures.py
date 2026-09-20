"""Regenerate the report figures as vector PDFs (PNG copies are throwaway
previews for eyeballing in an image viewer).

Data sources (all produced by the scripts in this repo):
  * ../results/fid_<run>.json       FID / NFE / throughput sweeps  (eval/fid.py)
  * ../results/systems_<cfg>.json   step time, memory, MFU, profile (eval/systems.py)
  * ../samples/<run>_epoch<N>.png   EMA sample grids               (main.py train ...)

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

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def save(fig, name):
    pdf = FIG / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"  wrote {pdf}")


def _load(pattern):
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(RESULTS.glob(pattern))]


# ── Figure: the two probability paths (schematic) ─────────────────────

def fig_paths():
    t = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    # diffusion: data at t=0, noise at t=1, curved (non-constant speed)
    ax.plot(t, (1 - t) ** 0.6, color=BLUE, lw=1.4, label="DDPM (curved)")
    ax.plot(t, 1 - t, color=ORANGE, lw=1.4, ls="--", label="rectified flow (straight)")
    ax.annotate("data", xy=(0, 1.0), xytext=(0.02, 1.04), color=GREY, fontsize=7)
    ax.annotate("noise", xy=(1, 0.0), xytext=(0.86, -0.06), color=GREY, fontsize=7)
    ax.set_xlabel("t  (0 = data, 1 = noise)")
    ax.set_ylabel("signal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.1, 1.12)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "paths")


# ── Figures from eval/fid.py output ───────────────────────────────────

def fig_fid_vs_nfe():
    data = _load("fid_*.json")
    if not data:
        print("  [skip] fid_vs_nfe: no results/fid_*.json yet")
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    colors = [BLUE, ORANGE, GREEN, RED, PURPLE, SKY]
    for d, color in zip(data, colors):
        runs = sorted(d["runs"], key=lambda r: r["nfe"])
        xs = [r["nfe"] for r in runs]
        ys = [r["fid"] for r in runs]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.2, color=color, label=d["run_name"])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("NFE (function evaluations)")
    ax.set_ylabel("FID")
    ax.legend(frameon=False)
    save(fig, "fid_vs_nfe")


def fig_throughput_vs_nfe():
    data = _load("fid_*.json")
    if not data or not any(r.get("sampling_img_s") for d in data for r in d["runs"]):
        print("  [skip] throughput_vs_nfe: no sampling throughput recorded yet")
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    colors = [BLUE, ORANGE, GREEN, RED, PURPLE, SKY]
    for d, color in zip(data, colors):
        runs = [r for r in sorted(d["runs"], key=lambda r: r["nfe"])
                if r.get("sampling_img_s")]
        if not runs:
            continue
        ax.plot([r["nfe"] for r in runs], [r["sampling_img_s"] for r in runs],
                marker="o", ms=3, lw=1.2, color=color, label=d["run_name"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("NFE")
    ax.set_ylabel("samples / s")
    ax.legend(frameon=False)
    save(fig, "throughput_vs_nfe")


# ── Figures from eval/systems.py output ───────────────────────────────

def fig_systems_settings():
    data = _load("systems_*.json")
    if not data:
        print("  [skip] systems_settings: no results/systems_*.json yet")
        return
    names = [Path(d["config"]).stem for d in data]
    settings = sorted({r["setting"] for d in data for r in d["results"]})
    x = np.arange(len(names))
    width = 0.8 / max(len(settings), 1)
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    colors = [BLUE, ORANGE, GREEN, RED]
    for i, setting in enumerate(settings):
        vals = []
        for d in data:
            match = [r for r in d["results"] if r["setting"] == setting]
            vals.append(match[0]["step_time_s"] * 1e3 if match else np.nan)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.9,
               color=colors[i % len(colors)], label=setting)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("ms / step")
    ax.legend(frameon=False, ncol=2)
    save(fig, "systems_settings")


def fig_profile_breakdown():
    data = [d for d in _load("systems_*.json") if d.get("profile_fractions")]
    if not data:
        print("  [skip] profile_breakdown: no --profile run recorded yet")
        return
    labels = sorted({k for d in data for k in d["profile_fractions"]})
    colors = {"matmul": BLUE, "attention": ORANGE, "conv": GREEN, "norm": RED,
              "elementwise": PURPLE, "communication": SKY, "other": LIGHTGREY}
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    x = np.arange(len(data))
    bottom = np.zeros(len(data))
    for label in labels:
        vals = np.array([d["profile_fractions"].get(label, 0.0) for d in data]) * 100
        ax.bar(x, vals, bottom=bottom, color=colors.get(label, GREY), label=label)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([Path(d["config"]).stem for d in data], rotation=20, ha="right")
    ax.set_ylabel("% of GPU time")
    ax.legend(frameon=False, ncol=4, fontsize=6)
    save(fig, "profile_breakdown")


# ── Sample grids ──────────────────────────────────────────────────────

def fig_samples():
    """Stack the newest sample grid per run into one preview image."""
    grids = {}
    for png in sorted(SAMPLES.glob("*_epoch*.png")):
        run = png.name.split("_epoch")[0]
        epoch = int(png.name.split("_epoch")[1].split(".")[0])
        if run not in grids or epoch > grids[run][0]:
            grids[run] = (epoch, png)
    if not grids:
        print("  [skip] samples: no samples/*_epoch*.png yet")
        return
    from PIL import Image

    imgs = [(run, Image.open(p)) for run, (_, p) in sorted(grids.items())]
    w = max(im.width for _, im in imgs)
    total_h = sum(im.height for _, im in imgs)
    canvas = Image.new("RGB", (w, total_h), "white")
    y = 0
    for _, im in imgs:
        canvas.paste(im, (0, y))
        y += im.height
    out = FIG / "samples.png"
    canvas.save(out)
    print(f"  wrote {out}  ({', '.join(r for r, _ in imgs)})")


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
