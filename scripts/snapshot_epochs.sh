#!/usr/bin/env bash
# Copies work/ckpt/model.pt to model_epN.pt each time the trainer finishes an epoch (model.pt is overwritten every epoch).
cd /home/airbotix/Downloads/CASMI || exit 1
last=0
while pgrep -f "casmi.train" > /dev/null; do
  n=$(grep -c "== epoch" work/train.log)
  if [ "$n" -gt "$last" ]; then
    sleep 20   # let the trainer finish writing the checkpoint
    [ -f "work/ckpt/model_ep${n}.pt" ] || cp work/ckpt/model.pt "work/ckpt/model_ep${n}.pt"
    last=$n
  fi
  sleep 30
done
n=$(grep -c "== epoch" work/train.log); [ -f "work/ckpt/model_ep${n}.pt" ] || cp work/ckpt/model.pt "work/ckpt/model_ep${n}.pt"
