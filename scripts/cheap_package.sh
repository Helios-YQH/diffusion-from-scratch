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
# Restart-safe: a stage (or a single MNIST cell, or a single FID sweep point)
# whose result is already recorded is skipped, so re-running after a crash
# resumes instead of starting over. A stage that fails is logged and the run
# continues; the failure list is printed at the end. Completed work is
# detected from results/*.json.
# ---------------------------------------------------------------------------
set -uo pipefail

GPUS=${GPUS:-0}
PY=${PY:-python3}
TORCHRUN=${TORCHRUN:-torchrun}
N=${N:-10000}
NGPU=$(awk -F',' '{print NF}' <<< "$GPUS")
export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_P2P_DISABLE=1

WANDB_FLAG=""
if [ "${WANDB:-0}" = "1" ]; then WANDB_FLAG="--wandb"; fi

if [ ! -f main.py ]; then echo "run me from the repo root"; exit 1; fi
mkdir -p logs results samples checkpoints
LOG="logs/cheap_package_$(date +%Y%m%d_%H%M%S).log"
FAILED=()

# have <path>: true when the stage that writes <path> is already done
have() { [ "${FORCE:-0}" = "1" ] && return 1; [ -f "$1" ]; }

# fid_has <json> <sampler> <nfe>: true when that exact sweep point is already
# recorded. FID JSONs are written incrementally, so file existence alone would
# mistake a half-finished sweep for a complete one.
fid_has() {
  [ "${FORCE:-0}" = "1" ] && return 1
  [ -f "$1" ] || return 1
  "$PY" - "$@" <<'PYEOF'
import json, sys
path, sampler, nfe = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    with open(path, encoding="utf-8") as f:
        runs = json.load(f).get("runs", [])
except Exception:
    sys.exit(1)
sys.exit(0 if any(r.get("sampler") == sampler and r.get("nfe") == nfe
                  for r in runs) else 1)
PYEOF
}

# run <label> <cmd...>: tee output into the log, record a failure but never
# abort the package — an unattended run should finish whatever it can.
run() {
  local label="$1"; shift
  echo ">>> $label" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  local st=${PIPESTATUS[0]}
  if [ "$st" -ne 0 ]; then
    echo "!!! FAILED (exit $st): $label" | tee -a "$LOG"
    FAILED+=("$label")
  fi
  return 0
}

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
    if [ -f "results/$c.json" ]; then
      echo "--- $c OK:" | tee -a "$LOG"
      tail -2 "logs/train_$c.log" 2>/dev/null | tee -a "$LOG"
    else
      echo "!!! FAILED: training $c (no results/$c.json)" | tee -a "$LOG"
      FAILED+=("train $c")
      tail -5 "logs/train_$c.log" 2>/dev/null | tee -a "$LOG"
    fi
  done

  echo | tee -a "$LOG"
  echo "=== 2/5  MNIST evaluation ===" | tee -a "$LOG"
  run "mnist reference stats" $PY eval/fid.py --build-ref --feature-net mnist
  for r in mnist_unet_eps mnist_dit_eps; do
    J="results/fid_$r.json"
    if fid_has "$J" ancestral 1000; then
      echo "  fid_$r ancestral already done — skipping" | tee -a "$LOG"
    elif [ "$NGPU" -gt 1 ]; then
      run "fid $r ancestral" $TORCHRUN --standalone --nproc_per_node="$NGPU" \
          eval/fid.py --run-name "$r" --feature-net mnist --sampler ancestral --num-samples "$N"
    else
      run "fid $r ancestral" $PY eval/fid.py \
          --run-name "$r" --feature-net mnist --sampler ancestral --num-samples "$N"
    fi
    for nfe in 10 20 50; do
      if fid_has "$J" ddim "$nfe"; then
        echo "  fid_$r DDIM nfe=$nfe already done — skipping" | tee -a "$LOG"
      else
        run "fid $r ddim nfe=$nfe" $PY eval/fid.py \
            --run-name "$r" --feature-net mnist --sampler ddim --nfe "$nfe"
      fi
    done
  done
  for r in mnist_unet_rf mnist_dit_rf; do
    J="results/fid_$r.json"
    for nfe in 4 8 16 32 64; do
      if fid_has "$J" euler "$nfe"; then
        echo "  fid_$r euler nfe=$nfe already done — skipping" | tee -a "$LOG"
      else
        run "fid $r euler nfe=$nfe" $PY eval/fid.py \
            --run-name "$r" --feature-net mnist --sampler euler --nfe "$nfe"
      fi
    done
  done
fi

if [ "${SKIP_CELEBA:-0}" != "1" ]; then
  echo | tee -a "$LOG"
  echo "=== 3/5  CelebA sampler study (no training) ===" | tee -a "$LOG"
  if [ ! -f checkpoints/ddpm_celeba_best.pt ]; then
    echo "  [skip] checkpoints/ddpm_celeba_best.pt is missing" | tee -a "$LOG"
  else
    run "celeba reference stats" $PY eval/fid.py --build-ref --feature-net inception
    J=results/fid_ddpm_celeba.json
    for nfe in 10 20 50; do
      if fid_has "$J" ddim "$nfe"; then
        echo "  DDIM nfe=$nfe already done — skipping" | tee -a "$LOG"
      else
        run "celeba ddim nfe=$nfe" $PY eval/fid.py \
            --run-name ddpm_celeba --sampler ddim --nfe "$nfe"
      fi
    done
    if fid_has "$J" ancestral 1000; then
      echo "  ancestral NFE=1000 already done — skipping" | tee -a "$LOG"
    else
      echo "--- the expensive point: 1000 NFE x $N samples (~4 h; see the runbook)" | tee -a "$LOG"
      if [ "$NGPU" -gt 1 ]; then
        run "celeba ancestral 1000" $TORCHRUN --standalone --nproc_per_node="$NGPU" \
            eval/fid.py --run-name ddpm_celeba --sampler ancestral --num-samples "$N"
      else
        run "celeba ancestral 1000" $PY eval/fid.py \
            --run-name ddpm_celeba --sampler ancestral --num-samples "$N"
      fi
    fi
  fi
fi

if [ "${SKIP_SYSTEMS:-0}" != "1" ]; then
  echo | tee -a "$LOG"
  echo "=== 4/5  systems measurements ===" | tee -a "$LOG"
  if have "results/systems_celeba_dit_rf.json"; then
    echo "  systems_celeba_dit_rf.json already exists — skipping (FORCE=1 to redo)" | tee -a "$LOG"
  else
    run "systems settings + profile" $PY eval/systems.py \
        --config configs/celeba_dit_rf.yml --settings fp32 bf16 compile bf16+compile --profile
    if [ "$NGPU" -gt 1 ]; then
      run "systems ddp" $TORCHRUN --standalone --nproc_per_node="$NGPU" eval/systems.py \
          --config configs/celeba_dit_rf.yml --ddp --settings fp32
    fi
  fi
fi

echo | tee -a "$LOG"
echo "=== 5/5  figures + tables ===" | tee -a "$LOG"
run "figures" $PY reports/make_figures.py
run "summary tables" $PY eval/make_table.py --out reports/results.md

echo | tee -a "$LOG"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "=== finished WITH FAILURES ===" | tee -a "$LOG"
  for f in "${FAILED[@]}"; do echo "  - $f" | tee -a "$LOG"; done
  echo "re-run this script to retry only the missing pieces" | tee -a "$LOG"
else
  echo "=== all stages completed ===" | tee -a "$LOG"
fi
echo "next:  cd reports && latexmk -pdf tech_report.tex" | tee -a "$LOG"
echo "full log: $LOG" | tee -a "$LOG"
