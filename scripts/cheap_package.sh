#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Cheap-study package: everything the report needs except the CelebA 2x2
# trainings. Wall clock ~2.5-3 h on 4 GPUs: the four MNIST cells run in
# parallel (one per GPU, ~65 min each at fp32), the two expensive ancestral
# FID sweeps are sharded over the same GPUs, and the single CelebA DDPM-1000
# point is the remaining big item.
#
# Run from the repo root:
#     bash scripts/cheap_package.sh
#
# Environment overrides:
#     GPUS=0,1,2,3   GPUs to use (pick ONE NUMA group, prefer idle cards)
#     PY=/path/to/python          (default: python3)
#     TORCHRUN=/path/to/torchrun  (default: torchrun)
#     N=10000        samples per FID evaluation
#     WANDB=1        pass --wandb to training runs (needs WANDB_API_KEY)
#     SKIP_MNIST=1 | SKIP_CELEBA=1 | SKIP_SYSTEMS=1
#     FORCE=1        redo stages whose outputs already exist
#
# The script is restart-safe: a stage (or a single MNIST cell) whose results
# JSON already exists is skipped, so re-running after a crash resumes instead
# of starting over. Completed stages are detected from results/*.json.
# ---------------------------------------------------------------------------
set -euo pipefail

GPUS=${GPUS:-0}
PY=${PY:-python3}
TORCHRUN=${TORCHRUN:-torchrun}
N=${N:-10000}
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_P2P_DISABLE=1

WANDB_FLAG=""
if [ "${WANDB:-0}" = "1" ]; then WANDB_FLAG="--wandb"; fi

# have <path>: true when the stage that writes <path> is already done
have() { [ "${FORCE:-0}" = "1" ] && return 1; [ -f "$1" ]; }

if [ ! -f main.py ]; then echo "run me from the repo root"; exit 1; fi
mkdir -p logs results samples checkpoints
LOG="logs/cheap_package_$(date +%Y%m%d_%H%M%S).log"

echo "=== environment ===" | tee -a "$LOG"
$PY -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| visible gpus', torch.cuda.device_count())" | tee -a "$LOG"
echo "GPUS=$GPUS ($NGPU visible)  N=$N  wandb=${WANDB:-0}" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader | tee -a "$LOG"

if [ "${SKIP_MNIST:-0}" != "1" ]; then
  echo | tee -a "$LOG"
  echo "=== 1/5  MNIST 2x2  (one cell per GPU, in parallel) ===" | tee -a "$LOG"
  i=0
  for c in mnist_unet_eps mnist_dit_eps mnist_unet_rf mnist_dit_rf; do
    g=$(echo "$GPUS" | cut -d',' -f$((i % NGPU + 1)))
    i=$((i + 1))
    if have "results/$c.json"; then
      echo "--- $c already trained (results/$c.json exists) — skipping" | tee -a "$LOG"
      continue
    fi
    echo "--- $c  ->  GPU $g   (per-cell log: logs/train_$c.log)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$g" $PY main.py train --config "configs/$c.yml" $WANDB_FLAG \
        > "logs/train_$c.log" 2>&1 &
  done
  wait || true
  for c in mnist_unet_eps mnist_dit_eps mnist_unet_rf mnist_dit_rf; do
    echo "--- $c finished:" | tee -a "$LOG"
    tail -3 "logs/train_$c.log" 2>/dev/null | tee -a "$LOG"
  done

  echo | tee -a "$LOG"
  echo "=== 2/5  MNIST evaluation ===" | tee -a "$LOG"
  $PY eval/fid.py --build-ref --feature-net mnist 2>&1 | tee -a "$LOG"
  for r in mnist_unet_eps mnist_dit_eps; do
    if have "results/fid_$r.json"; then
      echo "  fid_$r already done — skipping" | tee -a "$LOG"; continue
    fi
    if [ "$NGPU" -gt 1 ]; then
      $TORCHRUN --standalone --nproc_per_node="$NGPU" eval/fid.py \
          --run-name "$r" --feature-net mnist --sampler ancestral --n "$N" 2>&1 | tee -a "$LOG"
    else
      $PY eval/fid.py --run-name "$r" --feature-net mnist --sampler ancestral --n "$N" 2>&1 | tee -a "$LOG"
    fi
    $PY eval/fid.py --run-name "$r" --feature-net mnist --sampler ddim --nfe 10 20 50 2>&1 | tee -a "$LOG"
  done
  for r in mnist_unet_rf mnist_dit_rf; do
    if have "results/fid_$r.json"; then
      echo "  fid_$r already done — skipping" | tee -a "$LOG"; continue
    fi
    $PY eval/fid.py --run-name "$r" --feature-net mnist --sampler euler --nfe 4 8 16 32 64 2>&1 | tee -a "$LOG"
  done
fi

if [ "${SKIP_CELEBA:-0}" != "1" ]; then
  echo | tee -a "$LOG"
  echo "=== 3/5  CelebA sampler study (no training) ===" | tee -a "$LOG"
  if [ ! -f checkpoints/ddpm_celeba_best.pt ]; then
    echo "  [skip] checkpoints/ddpm_celeba_best.pt is missing" | tee -a "$LOG"
  elif have "results/fid_ddpm_celeba.json"; then
    echo "  fid_ddpm_celeba.json already exists — skipping (FORCE=1 to redo)" | tee -a "$LOG"
  else
    $PY eval/fid.py --build-ref --feature-net inception 2>&1 | tee -a "$LOG"
    for nfe in 10 20 50; do
      $PY eval/fid.py --run-name ddpm_celeba --sampler ddim --nfe "$nfe" 2>&1 | tee -a "$LOG"
    done
    echo "--- the expensive point: 1000 NFE x $N samples (~1 h on 4 GPUs, ~4 h on one)" | tee -a "$LOG"
    if [ "$NGPU" -gt 1 ]; then
      $TORCHRUN --standalone --nproc_per_node="$NGPU" eval/fid.py \
          --run-name ddpm_celeba --sampler ancestral --n "$N" 2>&1 | tee -a "$LOG"
    else
      $PY eval/fid.py --run-name ddpm_celeba --sampler ancestral --n "$N" 2>&1 | tee -a "$LOG"
    fi
  fi
fi

if [ "${SKIP_SYSTEMS:-0}" != "1" ]; then
  echo | tee -a "$LOG"
  echo "=== 4/5  systems measurements ===" | tee -a "$LOG"
  if have "results/systems_celeba_dit_rf.json"; then
    echo "  systems_celeba_dit_rf.json already exists — skipping (FORCE=1 to redo)" | tee -a "$LOG"
  else
    $PY eval/systems.py --config configs/celeba_dit_rf.yml \
        --settings fp32 bf16 compile bf16+compile --profile 2>&1 | tee -a "$LOG"
    if [ "$NGPU" -gt 1 ]; then
      $TORCHRUN --standalone --nproc_per_node="$NGPU" eval/systems.py \
          --config configs/celeba_dit_rf.yml --ddp --settings fp32 2>&1 | tee -a "$LOG"
    fi
  fi
fi

echo | tee -a "$LOG"
echo "=== 5/5  figures + tables ===" | tee -a "$LOG"
$PY reports/make_figures.py 2>&1 | tee -a "$LOG"
$PY eval/make_table.py --out reports/results.md 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "done. next:" | tee -a "$LOG"
echo "  cd reports && latexmk -pdf tech_report.tex      # figures now fill in" | tee -a "$LOG"
echo "  full log: $LOG" | tee -a "$LOG"
