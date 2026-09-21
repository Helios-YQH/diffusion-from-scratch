#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Tier-2 protocol (50k-sample FID) + the FSDP-vs-DDP reverse benchmark.
#
# This is the "finish the report" package: it upgrades the headline tables from
# 10k to 50k samples and adds the distributed-strategy comparison. It trains
# nothing — every run samples from a checkpoint that already exists.
#
# Wall clock ~3 h on 4 GPUs: the 1000-NFE CelebA point alone is ~1.8 h (it is
# 20x the cost of every other point in the script).
#
# Restart-safe and failure-tolerant, same as cheap_package.sh: completed points
# are detected from results/*.json and skipped, a failing stage is logged and
# the run continues.
#
# Environment overrides: GPUS, PY, TORCHRUN, FORCE=1, SKIP_SYSTEMS=1
# ---------------------------------------------------------------------------
set -uo pipefail

GPUS=${GPUS:-0,1,2,3}
PY=${PY:-python3}
TORCHRUN=${TORCHRUN:-torchrun}
NGPU=$(awk -F',' '{print NF}' <<< "$GPUS")
export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_P2P_DISABLE=1

N=50000
if [ ! -f main.py ]; then echo "run me from the repo root"; exit 1; fi
mkdir -p logs results
LOG="logs/tier2_$(date +%Y%m%d_%H%M%S).log"
FAILED=()

# t2_has <json> <sampler> <nfe>: true when a 50k-tier entry for that point exists
t2_has() {
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
                  and r.get("n", 0) >= 50000 for r in runs) else 1)
PYEOF
}

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

# fid <run> <feature-net> <sampler> <nfe...>  — one 50k point per NFE, skipping
# the ones already recorded.
fid() {
  local r="$1" net="$2" sampler="$3"; shift 3
  local json="results/fid_$r.json"
  for nfe in "$@"; do
    if t2_has "$json" "$sampler" "$nfe"; then
      echo "  $r $sampler nfe=$nfe @50k already done — skipping" | tee -a "$LOG"
      continue
    fi
    if [ "$NGPU" -gt 1 ]; then
      run "$r $sampler nfe=$nfe (50k, $NGPU GPUs)" \
        $TORCHRUN --standalone --nproc_per_node="$NGPU" eval/fid.py \
        --run-name "$r" --feature-net "$net" --sampler "$sampler" \
        --nfe "$nfe" --num-samples "$N" --variability
    else
      run "$r $sampler nfe=$nfe (50k)" \
        $PY eval/fid.py --run-name "$r" --feature-net "$net" \
        --sampler "$sampler" --nfe "$nfe" --num-samples "$N" --variability
    fi
  done
}

echo "=== environment ===" | tee -a "$LOG"
$PY -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| visible gpus', torch.cuda.device_count())" | tee -a "$LOG"
echo "GPUS=$GPUS ($NGPU visible)  50k samples per point" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader | tee -a "$LOG"

# ── 1. distributed strategy: DDP vs FSDP at matched work ----------------
if [ "${SKIP_SYSTEMS:-0}" != "1" ] && [ "$NGPU" -gt 1 ]; then
  echo | tee -a "$LOG"
  echo "=== 1/4  DDP vs FSDP (reverse benchmark) ===" | tee -a "$LOG"
  run "systems ddp (fp32)" $TORCHRUN --standalone --nproc_per_node="$NGPU" \
      eval/systems.py --config configs/celeba_dit_rf.yml --ddp --settings fp32
  run "systems fsdp (fp32)" $TORCHRUN --standalone --nproc_per_node="$NGPU" \
      eval/systems.py --config configs/celeba_dit_rf.yml --fsdp --settings fp32
fi

# ── 2. CelebA: the report's headline table, at 50k ----------------------
echo | tee -a "$LOG"
echo "=== 2/4  CelebA sampler study @ 50k ===" | tee -a "$LOG"
fid ddpm_celeba inception ddim 10 20 50
echo "--- the long one: 1000 NFE x 50k (~1.8 h on $NGPU GPUs)" | tee -a "$LOG"
fid ddpm_celeba inception ancestral 1000

# ── 3. MNIST 2x2 at 50k -------------------------------------------------
echo | tee -a "$LOG"
echo "=== 3/4  MNIST cells @ 50k ===" | tee -a "$LOG"
fid mnist_unet_eps mnist ancestral 1000
fid mnist_unet_eps mnist ddim 20 50
fid mnist_dit_eps  mnist ancestral 1000
fid mnist_unet_rf  mnist euler 16 32 64
fid mnist_dit_rf   mnist euler 16 32 64

# ── 4. figures and tables ----------------------------------------------
echo | tee -a "$LOG"
echo "=== 4/4  figures + tables ===" | tee -a "$LOG"
run "figures" $PY reports/make_figures.py
run "summary tables" $PY eval/make_table.py --out reports/results.md

echo | tee -a "$LOG"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "=== finished WITH FAILURES ===" | tee -a "$LOG"
  for f in "${FAILED[@]}"; do echo "  - $f" | tee -a "$LOG"; done
else
  echo "=== all stages completed ===" | tee -a "$LOG"
fi
echo "full log: $LOG" | tee -a "$LOG"
