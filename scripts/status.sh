#!/usr/bin/env bash
# One-shot status report for a running (or finished) cheap_package run.
#
# From your laptop:
#   ssh -p 30153 houyi@frp-egg.com 'bash /mnt/14T/houyi/dit-sys/scripts/status.sh'
#
# From a shell on the server:
#   cd /mnt/14T/houyi/dit-sys && bash scripts/status.sh
cd "$(dirname "$0")/.." || exit 1

echo "=========== 进程 ==========="
if pgrep -f "[c]heap_package.sh" >/dev/null; then
  echo "cheap_package.sh : RUNNING"
else
  echo "cheap_package.sh : not running (finished, or stopped)"
fi
# NOTE: filter out this script's own pipeline (its sed/pgrep command lines
# contain the very strings we are searching for).
ps -eo args= | grep -vE "status\.sh|sed |pgrep |grep " | \
  grep -E "main\.py train|eval/fid\.py|eval/systems\.py|torchrun" | \
  sed 's|/mnt/14T/houyi/miniconda3/envs/torch/bin/||' | \
  sed 's|--config configs/|--config |;s|\.yml.*||' | \
  cut -c1-130 | sed 's/^/  /'

echo
echo "=========== 各 cell 训练进度（末行含当轮耗时）==========="
for f in logs/train_*.log; do
  [ -f "$f" ] || continue
  name=$(basename "$f" .log | sed 's/train_//')
  line=$(grep -aE "^Epoch [0-9]+/" "$f" | tail -1)
  printf "  %-18s %s\n" "$name" "${line:-（尚未完成第一个 epoch）}"
done

echo
echo "=========== 最近的主日志 ==========="
LOG=$(ls -t logs/cheap_package_*.log 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
  echo "  ($LOG)"
  grep -avE "^\s*$" "$LOG" | tail -8 | cut -c1-150 | sed 's/^/  /'
else
  echo "  (no log)"
fi

echo
echo "=========== 已有结果文件 ==========="
ls -1 results/*.json 2>/dev/null | sed 's|results/|  |' || echo "  (none yet)"
ls -1 reports/figures/*.pdf 2>/dev/null | sed 's|reports/figures/|  fig: |' || true

echo
echo "=========== GPU ==========="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
