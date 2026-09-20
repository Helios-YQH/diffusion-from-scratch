# Runbook — 2×2 study on the GPU server

## Server facts (measured 2026-09-20)

| | |
|---|---|
| SSH | `ssh -p 30153 houyi@frp-egg.com` (passwordless; the frp tunnel drops occasionally) |
| Python | `/mnt/14T/houyi/miniconda3/envs/torch/bin/python` — Python 3.11, torch 2.13.0+cu126, 6 GPUs |
| GPUs | 6 × RTX A6000 48GB, **shared with other users** |
| Topology | GPU 0–3 = NUMA 0 (PIX, fast); 4–5 = NUMA 1; cross-group = SYS (**slow**) → keep DDP inside one group |
| Disk | root 9.7 GB free, `/mnt/14T` 20 GB free; `/mnt/15_14T` has 1.5 TB but **not writable** |
| uv | not installed — use the conda env directly (no `uv sync` needed) |

**Disk budget** (with the uint8 cache): data ≈ 1.4 GB zip + 1.6 GB extracted + 2.5 GB cache;
checkpoints ≈ 1.0–1.4 GB per `best`, 0.5–0.7 GB per EMA snapshot, 2.1–2.8 GB per `latest`.
All four cells together ≈ 22 GB → run cells one at a time; delete a finished cell's
`latest`/snapshots if space runs short.

## 0. One-time setup

```bash
# --- local machine ---
scp -P 30153 data/celeba.zip houyi@frp-egg.com:~/dit-sys/data/
scp -P 30153 data/mnist/mnist.pkl.gz houyi@frp-egg.com:~/dit-sys/data/mnist/

# --- server ---
cd ~ && git clone git@gitee.com:hy_ucas/dit-sys.git dit-sys && cd dit-sys
PY=/mnt/14T/houyi/miniconda3/envs/torch/bin/python
$PY tests/test_sanity.py          # CPU invariants, ~1 min
```

## 1. Pre-flight — always do this first

```bash
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
```

Pick GPUs **within one NUMA group** (0–3 or 4–5), preferring idle ones.
Never touch other users' processes; if nothing is free, wait.

## 2. Smoke tests (~2 min each)

```bash
PY=/mnt/14T/houyi/miniconda3/envs/torch/bin/python
mkdir -p logs

# MNIST: exercises the whole loop without CelebA preprocessing
$PY main.py train --config configs/mnist.yml --max-steps 30 --no-resume \
     --run-name smoke_mnist

# First CelebA run also extracts the zip and builds the uint8 cache (~20 min one-time)
CUDA_VISIBLE_DEVICES=<free card> $PY main.py train \
     --config configs/celeba_dit_rf.yml --max-steps 30 --no-resume \
     --run-name smoke_dit_rf
```

## 3. Launch a full cell

Keep the **global** batch at 800 (the frozen protocol): `--batch-size $((800 / NGPUS))`
(e.g. 4 GPUs → 200). `steps/epoch` stays identical whatever the GPU count.

```bash
TORCHRUN=/mnt/14T/houyi/miniconda3/envs/torch/bin/torchrun
export WANDB_API_KEY=<key>                     # or run `torchrun` env's `wandb login`

# D: DiT + rectified flow  (start here)
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 \
  nohup $TORCHRUN --standalone --nproc_per_node=4 main.py train \
  --config configs/celeba_dit_rf.yml --batch-size 200 --wandb \
  > logs/dit_rf.log 2>&1 &

tail -f logs/dit_rf.log
```

Resume after an interruption: the configs have `resume: true`, so re-running the same
command continues from `checkpoints/<run_name>_latest.pt`. Add `--no-resume` for a fresh
start.

## 4. Evaluation

```bash
PY=/mnt/14T/houyi/miniconda3/envs/torch/bin/python

# reference stats — once, ~5 min (writes eval/stats/celeba64_ref_pool3.npz)
$PY eval/fid.py --build-ref

# Tier 1 (10k samples) — sweeps
$PY eval/fid.py --run-name celeba_dit_rf  --n 10000 --sampler euler  --nfe 4 8 16 32 64 128
$PY eval/fid.py --run-name celeba_dit_eps --n 10000 --sampler ddim   --nfe 10 20 50
$PY eval/fid.py --run-name celeba_unet_eps --n 10000 --sampler ancestral

# Tier 2 (50k, sharded over GPUs) — final numbers only
CUDA_VISIBLE_DEVICES=0,1,2,3 $TORCHRUN --standalone --nproc_per_node=4 \
  eval/fid.py --run-name celeba_dit_rf --n 50000 --sampler euler --nfe 16
```

Each call appends to `results/fid_<run_name>.json` (re-running the same
sampler/NFE/weights combination overwrites that entry).

## 5. Order of operations

1. **D — DiT + RF** (fastest to a result)
2. **B — DiT + eps**
3. **A — UNet + eps** (retrained under the frozen protocol)
4. **C — UNet + RF** (the expensive one)

## 6. Gotchas

- `NCCL_P2P_DISABLE=1` is required — the topology hangs without it
- Long jobs must run under `nohup`; the frp tunnel drops and would kill a foreground job
- Run from the repo root (`checkpoints/`, `samples/`, `results/` are relative paths)
- Copy `results/*.json` and `samples/*.png` back to the laptop when building the report
  (they are the experiment archive and get committed later)
- If a GPU is `0%` but has memory held by someone else's process, do not assume it is
  free — check `nvidia-smi --query-compute-apps`
