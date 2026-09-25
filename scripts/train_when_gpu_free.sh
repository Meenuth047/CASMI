#!/usr/bin/env bash
# Starts CASMI training ONLY after the YOLO11 training process has exited and the GPU memory is released.
# Usage: setsid nohup bash scripts/train_when_gpu_free.sh [extra casmi.train args] &
cd /home/airbotix/Downloads/CASMI || exit 1
PY=/home/airbotix/casmi-gpu-venv/bin/python
LAUNCH_LOG=work/launcher.log
echo "$(date '+%F %T') launcher started; waiting for train_yolo11s.py to exit" >> "$LAUNCH_LOG"
while pgrep -f "train_yolo11s.py" > /dev/null; do sleep 60; done
echo "$(date '+%F %T') YOLO process gone; waiting for GPU memory to be released" >> "$LAUNCH_LOG"
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ -n "$used" ] && [ "$used" -lt 4000 ] && break
  sleep 30
done
# re-check: a new YOLO run may have been started in the meantime
if pgrep -f "train_yolo11s.py" > /dev/null; then exec bash "$0" "$@"; fi
echo "$(date '+%F %T') GPU free (${used} MiB used) -> starting casmi.train $*" >> "$LAUNCH_LOG"
"$PY" -u -m casmi.train "$@" >> work/train.log 2>&1
echo "$(date '+%F %T') casmi.train exited with code $?" >> "$LAUNCH_LOG"
