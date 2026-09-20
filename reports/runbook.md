# Runbook — running the 2×2 study

From a fresh clone to FID numbers. Written to be **machine-agnostic**: the reference path is
a [uv](https://docs.astral.sh/uv/) environment (portable, and it pins torch via `uv.lock`);
an already-working conda/pip environment is a drop-in alternative.

## 0. What you need

- Linux box, ≥1 NVIDIA GPU (the models fit in 24GB; the protocol assumes 4–6 for DDP)
- ~30GB free disk: data ~5.5GB (zip 1.4 + extracted 1.6 + uint8 cache 2.5) + checkpoints
  ~22GB for all four cells (per cell: `best` 1.0–1.4GB, EMA snapshots 0.5–0.7GB each,
  `latest` 2.1–2.8GB)
- Python 3.10+ (uv can fetch one)

## 1. Environment

### Default: uv

```bash
git clone git@gitee.com:hy_ucas/dit-sys.git && cd dit-sys
uv sync              # torch, numpy, matplotlib, pillow, tqdm, pyyaml
uv sync --extra log  # + wandb
uv run python tests/test_sanity.py     # 14 CPU invariants, ~1 min, no GPU needed
```

- `uv run <cmd>` runs inside `.venv`; `uv run torchrun ...` works too.
- **CUDA mismatch?** `uv sync` pulls the default CUDA build from PyPI. If `nvidia-smi` shows
  an older driver, override: `uv pip install torch --index-url https://download.pytorch.org/whl/cu121`
  (pick the `cuXXX` matching the driver), then re-run the tests.
- **No uv on the box / no network?** Any env with torch ≥2.0 + numpy + matplotlib + pillow +
  tqdm + pyyaml works: activate it and run `python main.py ...` directly. `uv.lock` exists only
  to make the reference environment reproducible.

### Alternative: an existing conda/pip env

```bash
PY=/path/to/envs/torch/bin/python
$PY tests/test_sanity.py
$PY main.py train --config configs/celeba_dit_rf.yml ...
```

## 2. Data

`data/` is gitignored, so it must be copied to the machine that trains:

```bash
mkdir -p data/mnist
# from the machine that holds the dataset:
scp data/celeba.zip        <host>:dit-sys/data/
scp data/mnist/mnist.pkl.gz <host>:dit-sys/data/mnist/
```

The first CelebA run extracts the zip and builds a **uint8** cache
(`data/celeba_64_uint8.pt`, ~2.5GB) — one-time, ~20 min for 202k images.

## 3. Pre-flight (every session)

```bash
nvidia-smi --query-gpu=index,name,memory.free,utilization,gpu --format=csv,noheader
```

On a shared box: never disturb another user's process, and if nothing is free — wait.
A GPU that shows `0%` utilisation but holds memory is still someone's allocation.

## 4. Smoke tests

```bash
uv run python tests/test_sanity.py                     # CPU invariants

uv run python main.py train --config configs/mnist_unet_eps.yml \
    --max-steps 30 --no-resume --run-name smoke_mnist   # ~1 min

uv run python main.py train --config configs/celeba_dit_rf.yml \
    --max-steps 30 --no-resume --run-name smoke_dit_rf  # + cache build on first run
```

`--max-steps 30` stops after 30 optimizer steps; check the printed loss, grad norm and the
`samples/` grid before committing to a full run.

## 5. Launching a cell

Keep the **global** batch at 800 (the frozen protocol) — `steps/epoch` then stays identical
regardless of GPU count:

```bash
NGPUS=4                     # GPUs you actually got
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 \
  nohup uv run torchrun --standalone --nproc_per_node=$NGPUS main.py train \
  --config configs/celeba_dit_rf.yml --batch-size $((800 / NGPUS)) --wandb \
  > logs/dit_rf.log 2>&1 &

tail -f logs/dit_rf.log
```

- `NCCL_P2P_DISABLE=1` is required on topologies where P2P hangs (it was on the 6×A6000 box).
- Long runs must be detached (`nohup`/`tmux`) — an SSH drop would otherwise kill them.
- **Resume:** the configs set `resume: true`, so re-running the same command continues from
  `checkpoints/<run_name>_latest.pt`. `--no-resume` forces a fresh start.
- **W&B:** export `WANDB_API_KEY` (the key lives in the gitignored `.env` on the laptop) or
  run `wandb login` inside the env. Drop `--wandb` to train without it.
- **Order:** D (DiT+RF) → B (DiT+ε) → A (UNet+ε) → C (UNet+RF). D is the fastest path to a
  result; C is the most expensive cell.

### How long a cell takes

Derived from the measured FLOPs (training ≈ 3× forward), 253.7 steps/epoch, 300 epochs:

| Cell | Forward GFLOPs/sample | Train FLOPs/step @800 | @20% MFU, 4×A6000 | @35% MFU |
|---|---|---|---|---|
| DiT (B, D) | 43.6 | 104.6 TFLOPs | ~18 h | ~10 h |
| UNet (A, C) | 67.8 | 162.8 TFLOPs | ~28 h | ~16 h |
| **all four** | | | **~3.8 days** | **~2.2 days** |

Empirical anchor: the previous 300-epoch UNet run on this shared box spanned ~3 days of
wall-clock. Treat ~1–2 days (DiT) and ~1.5–3 days (UNet) per cell as the planning range, and
let the first cell calibrate — the printed `s/step` and MFU make the extrapolation explicit.

## 6. Evaluation

```bash
PY="uv run python"        # or an existing env's interpreter

# reference stats — once, ~5 min (writes eval/stats/celeba64_ref_pool3.npz)
$PY eval/fid.py --build-ref

# Tier 1 (10k samples) — sweeps while a run is still going
$PY eval/fid.py --run-name celeba_dit_rf  --n 10000 --sampler euler  --nfe 4 8 16 32 64 128
$PY eval/fid.py --run-name celeba_dit_eps --n 10000 --sampler ddim   --nfe 10 20 50
$PY eval/fid.py --run-name celeba_unet_eps --n 10000 --sampler ancestral

# Tier 2 (50k, sharded over GPUs) — final numbers only
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  eval/fid.py --run-name celeba_dit_rf --n 50000 --sampler euler --nfe 16
```

Each call appends to `results/fid_<run_name>.json` (re-running the same
sampler/NFE/weights combination overwrites that entry).

Cost accounting for a finished run:

```bash
uv run python eval/cost.py --config configs/celeba_dit_rf.yml \
  --step-time 1.23 --batch-size 800 --gpu A6000
```

## 6b. The cheap package (recommended scope, ~6–8 GPU·h)

Everything the report needs except the CelebA 2×2 trainings. Run in this order.

```bash
PY="uv run python"        # or an existing env's interpreter

# 1) MNIST 2×2 — all four cells, ~5 min each on one GPU
for c in mnist_unet_eps mnist_dit_eps mnist_unet_rf mnist_dit_rf; do
  $PY main.py train --config configs/$c.yml
done

# 2) MNIST evaluation (reference stats + sampler sweep per cell)
$PY eval/fid.py --build-ref --feature-net mnist
for r in mnist_unet_eps mnist_dit_eps; do
  $PY eval/fid.py --run-name $r --feature-net mnist --sampler ancestral --n 10000
  $PY eval/fid.py --run-name $r --feature-net mnist --sampler ddim --nfe 10 20 50
done
for r in mnist_unet_rf mnist_dit_rf; do
  $PY eval/fid.py --run-name $r --feature-net mnist --sampler euler --nfe 4 8 16 32 64
done

# 3) CelebA sampler study on the EXISTING DDPM checkpoint — no training needed.
#    Requires the old checkpoint on the box (it is not in git):
#      scp checkpoints/ddpm_celeba_best.pt <host>:dit-sys/checkpoints/
#    and the data (§2). The 1000-NFE point is the expensive one: ~1 h on 4 GPUs,
#    ~4 h on one. Every DDIM point is minutes.
$PY eval/fid.py --build-ref --feature-net inception
$PY eval/fid.py --run-name ddpm_celeba --sampler ancestral --n 10000
$PY eval/fid.py --run-name ddpm_celeba --sampler ddim --nfe 10 20 50

# 4) Systems measurements (minutes, no training)
$PY eval/systems.py --config configs/celeba_dit_rf.yml \
    --settings fp32 bf16 compile bf16+compile --profile
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
    eval/systems.py --config configs/celeba_dit_rf.yml --ddp --settings fp32

# 5) Figures and tables for the report
$PY reports/make_figures.py
$PY eval/make_table.py --out reports/results.md
```

The four CelebA cells (Section 5) are the expensive half; add them when GPUs allow and the
same figures/tables absorb them automatically.

## 6c. Building the report

```bash
cd reports && latexmk -pdf tech_report.tex     # 7 pages now, placeholders auto-fill
```

`tech_report.tex` includes any figure that exists in `reports/figures/` and renders a
labelled placeholder box for the rest, so it compiles before the results land.

## 7. Collecting results

```bash
# back on the laptop
scp -r <host>:dit-sys/results/*.json  results/
scp -r <host>:dit-sys/samples/*.png   samples/
uv run python eval/make_table.py --out reports/results.md
```

`results/*.json` is the raw archive behind every number in the report and is committed to the
repo once the runs finish.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| Hang at startup, no progress | `NCCL_P2P_DISABLE=1`; also check that all GPUs are in one NUMA group |
| `torch.cuda.OutOfMemoryError` at batch 200/GPU | drop to 160/GPU with 5 GPUs (keeps the global 800) or note the protocol deviation |
| OOM mid-run on a shared box | someone else grew their allocation — restart; `resume: true` continues |
| Loss explodes / NaNs | gradient clipping is on (1.0); lower `--lr` (5e-4 collapsed in an earlier run) |
| Sampling takes forever | `--sample-interval 0` disables the periodic grid; NFE is the real cost |
| `--save-interval 0` | saves only at the final epoch (a `latest` checkpoint always exists) |
| Everything ran but no metrics | check `results/<run>.json`; the run prints its path on exit |

## 9. Machine notes (optional)

### The shared 6×A6000 box (as of 2026-09-20)

| | |
|---|---|
| SSH | `ssh -p 30153 houyi@frp-egg.com` (passwordless; the frp tunnel drops sometimes) |
| Python | `/mnt/14T/houyi/miniconda3/envs/torch/bin/python` (torch 2.13+cu126, 6 GPUs) — **uv is not installed**, so use the conda env path above |
| Topology | GPU 0–3 = NUMA 0 (PIX); 4–5 = NUMA 1; cross-group = SYS (**slow**) → keep DDP inside one group |
| Disk | root 9.7GB / `/mnt/14T` 20GB free; `/mnt/15_14T` has 1.5TB but is not writable → hence the uint8 cache and the tight checkpoint budget |
| GPUs | shared with other users; check before every launch |
