"""Render the results archive (results/*.json) as markdown tables.

Usage:
  python eval/make_table.py                 # print to stdout
  python eval/make_table.py --out reports/results.md
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = "results"

# run name -> cell id in the 2x2 design
CELLS = {
    "celeba_unet_eps": "A",
    "celeba_dit_eps": "B",
    "celeba_unet_rf": "C",
    "celeba_dit_rf": "D",
}
CELL_ORDER = ["celeba_unet_eps", "celeba_dit_eps", "celeba_unet_rf", "celeba_dit_rf"]


def _cell(run_name):
    return CELLS.get(run_name, "?")


def _sort_key(run_name):
    return (CELL_ORDER.index(run_name) if run_name in CELL_ORDER else 99, run_name)


def load_training():
    out = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        base = os.path.basename(path)
        if base.startswith("fid_") or base.startswith("systems_"):
            continue  # FID sweeps and systems benchmarks have their own tables
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
    return sorted(out, key=lambda d: _sort_key(d.get("run_name", "")))


def load_eval():
    out = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "fid_*.json")):
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
    return sorted(out, key=lambda d: _sort_key(d.get("run_name", "")))


def _fmt(value, digits=2, dash="—"):
    if value is None:
        return dash
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def training_table(runs):
    if not runs:
        return "_No training runs found in results/._\n"
    lines = [
        "| Cell | Run | Backbone | Objective | Params (M) | GFLOPs/sample | "
        "s/step | img/s | Peak GB | MFU | Best loss |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in runs:
        run = d.get("run_name", "?")
        mfu = d.get("mfu")
        lines.append(
            f"| {_cell(run)} | `{run}` | {d.get('backbone', '?')} | "
            f"{d.get('objective', '?')} | {_fmt(d.get('params', 0) / 1e6, 1)} | "
            f"{_fmt((d.get('forward_flops_per_sample') or 0) / 1e9)} | "
            f"{_fmt(d.get('step_time_s'), 3)} | "
            f"{_fmt(d.get('throughput_img_s'), 0)} | "
            f"{_fmt(d.get('peak_memory_gb'), 1)} | "
            f"{('%d%%' % round(mfu * 100)) if mfu is not None else '—'} | "
            f"{_fmt(d.get('best_loss'), 4)} |")
    return "\n".join(lines) + "\n"


def fid_table(evals):
    if not evals:
        return "_No FID results found in results/._\n"
    lines = [
        "| Cell | Run | Sampler | NFE | n | Weights | FID |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in evals:
        run = d.get("run_name", "?")
        for r in sorted(d.get("runs", []), key=lambda r: r["nfe"]):
            lines.append(
                f"| {_cell(run)} | `{run}` | {r['sampler']} | {r['nfe']} | "
                f"{r['n']} | {r['weights']} | {r['fid']:.3f} |")
    return "\n".join(lines) + "\n"


def build():
    parts = ["## Training cost\n", training_table(load_training()),
             "\n## Generation quality (FID)\n", fid_table(load_eval())]
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser("results -> markdown")
    p.add_argument("--out", default=None, help="write to this file instead of stdout")
    args = p.parse_args()

    text = build()
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Written: {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
