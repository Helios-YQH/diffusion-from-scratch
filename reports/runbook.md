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
# GitHub is the primary remote; the Gitee mirror is faster from inside China
git clone git@github.com:Helios-YQH/diffusion-from-scratch.git && cd diffusion-from-scratch
# or: git clone git@gitee.com:hy_ucas/dit-sys.git && cd dit-sys
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

Measured on the 6×A6000 box, **training in fp32** (the loop has no autocast):
a MNIST cell is ~65–80 min, and the CelebA cells extrapolate from their measured FLOPs
(training ≈ 3× forward; the step runs at ≈49% of the fp32 peak, which is what the
MNIST run demonstrated):

| Cell | Forward GFLOPs/sample | Train FLOPs/step @800 | ≈ wall clock, 4 GPUs, fp32 |
|---|---|---|---|
| DiT (B, D) | 43.6 | 104.6 TFLOPs | ~36 h |
| UNet (A, C) | 67.8 | 162.8 TFLOPs | ~55 h |
| **all four** | | | **~4 days** |

**Mixed precision is the lever**: `eval/systems.py` measures bf16 against fp32 on this
hardware, and that ratio is the number to apply to this table before committing to the
CelebA 2×2. The one data point we have is that the MNIST cells hit ≈49% of fp32 peak, which
is a healthy fp32 efficiency — the time is going into a quarter-width compute pipe, not into
overhead.

## 6. Evaluation

```bash
PY="uv run python"        # or an existing env's interpreter

# reference stats — once, ~5 min (writes eval/stats/celeba64_ref_pool3.npz)
$PY eval/fid.py --build-ref

# Tier 1 (10k samples) — sweeps while a run is still going
$PY eval/fid.py --run-name celeba_dit_rf  --num-samples 10000 --sampler euler  --nfe 4 8 16 32 64 128
$PY eval/fid.py --run-name celeba_dit_eps --num-samples 10000 --sampler ddim   --nfe 10 20 50
$PY eval/fid.py --run-name celeba_unet_eps --num-samples 10000 --sampler ancestral

# Tier 2 (50k, sharded over GPUs) — final numbers only
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  eval/fid.py --run-name celeba_dit_rf --num-samples 50000 --sampler euler --nfe 16
```

Each call appends to `results/fid_<run_name>.json` (re-running the same
sampler/NFE/weights combination overwrites that entry).

Cost accounting for a finished run:

```bash
uv run python eval/cost.py --config configs/celeba_dit_rf.yml \
  --step-time 1.23 --batch-size 800 --gpu A6000
```

## 6b. The cheap package (recommended scope)

One script runs everything the report needs except the CelebA 2×2 trainings — the four MNIST
cells in parallel (one per GPU), both FID sweeps, the systems measurements, then the figures
and tables:

```bash
GPUS=0,1,2,3 PY=<interpreter> TORCHRUN=<torchrun> \
  nohup bash scripts/cheap_package.sh </dev/null >/dev/null 2>&1 &
tail -f logs/cheap_package_*.log          # progress; scripts/status.sh summarises it
```

Prerequisites: the data (§2) and, for the CelebA half, `checkpoints/ddpm_celeba_best.pt`
(not in git — copy it over). Wall clock ≈ 5.5–6 h on 4 GPUs, dominated by the single 1000-NFE
CelebA point (≈4 h in fp32).

The script is **restart-safe**: completed stages, individual MNIST cells and individual FID
sweep points are detected from `results/*.json` and skipped, so a re-run after a crash
resumes rather than starting over (`FORCE=1` redoes everything). A stage that fails is logged
and the run continues; failures are listed at the end.

```bash
# progress at any time (from your laptop):
ssh -p 30153 houyi@frp-egg.com 'bash /mnt/14T/houyi/dit-sys/scripts/status.sh'
```

The individual commands, if you want to run them by hand:

```bash
PY="uv run python"        # or an existing env's interpreter

# 1) MNIST 2x2 — four cells, ~65-80 min each; give each cell its own GPU
for c in mnist_unet_eps mnist_dit_eps mnist_unet_rf mnist_dit_rf; do
  CUDA_VISIBLE_DEVICES=$g $PY main.py train --config configs/$c.yml &
done; wait

# 2) MNIST evaluation (reference stats + sampler sweep per cell)
$PY eval/fid.py --build-ref --feature-net mnist
$PY eval/fid.py --run-name mnist_unet_eps --feature-net mnist --sampler ancestral --num-samples 10000
$PY eval/fid.py --run-name mnist_unet_eps --feature-net mnist --sampler ddim --nfe 10 20 50
# ... likewise for the other three cells

# 3) CelebA sampler study on the EXISTING DDPM checkpoint — no training needed.
#    The 1000-NFE point is the expensive one (~4 h on 4 GPUs in fp32); DDIM is minutes.
$PY eval/fid.py --build-ref --feature-net inception
$PY eval/fid.py --run-name ddpm_celeba --sampler ddim --nfe 10 20 50
$TORCHRUN --standalone --nproc_per_node=4 \
  eval/fid.py --run-name ddpm_celeba --sampler ancestral --num-samples 10000

# 4) Systems measurements (minutes, no training)
$PY eval/systems.py --config configs/celeba_dit_rf.yml \
    --settings fp32 bf16 compile bf16+compile --profile
CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --standalone --nproc_per_node=2 \
    eval/systems.py --config configs/celeba_dit_rf.yml --ddp --settings fp32

# 5) Figures and tables for the report
$PY reports/make_figures.py
$PY eval/make_table.py --out reports/results.md
```

> `eval/fid.py` deliberately has no `--n` flag: torchrun's own argument parser rejects
> `--n` as an ambiguous prefix of `--nnodes`/`--nproc-per-node`. Use `--num-samples`.

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
