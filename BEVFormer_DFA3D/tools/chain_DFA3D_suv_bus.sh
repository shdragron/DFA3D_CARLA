#!/usr/bin/env bash
# Auto-chain DFA3D-CARLA oracles: wait for the running SEDAN run to finish
# (epoch_24.pth + process exit), then train SUV, then BUS — sequentially on the
# same 2 GPUs (global batch 16, lr 4e-4, identical to sedan). The sedan run keeps
# going in its own process; this only adds suv -> bus afterwards.
#
# Usage:  cd BEVFormer_DFA3D; nohup bash tools/chain_DFA3D_suv_bus.sh > logs/chain_DFA3D.log 2>&1 &
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
GPUS=2
CFGDIR=projects/configs/bevformer

wait_done() {   # $1 = work_dir name (== config basename)
  echo "[chain] waiting for work_dirs/$1/epoch_24.pth ... ($(date))"
  until [ -f "work_dirs/$1/epoch_24.pth" ]; do sleep 300; done
  echo "[chain] $1 hit epoch_24; waiting for its training process to exit ..."
  # match the exact work-dir token so suv/bus (carla_suv / carla_bus) don't match sedan.
  # [t]ools regex trick: matches the real train.py procs but NOT pgrep's own cmdline.
  until ! pgrep -f "[t]ools/train.py.*work_dirs/$1 --deterministic" >/dev/null 2>&1; do sleep 60; done
  sleep 20
  echo "[chain] $1 finished + GPUs free ($(date))"
}

run() {         # $1 = config path, $2 = master port
  echo "[chain] ===== launching $(basename "$1") ($(date)) ====="
  CUDA_VISIBLE_DEVICES=0,1 bash tools/train_DFA3D_carla.sh "$1" "$2" "$GPUS"
}

# sedan is already training in its own process (work_dir bevformer_DFA3D_carla)
wait_done bevformer_DFA3D_carla
run "$CFGDIR/bevformer_DFA3D_carla_suv.py" 28534
wait_done bevformer_DFA3D_carla_suv
run "$CFGDIR/bevformer_DFA3D_carla_bus.py" 28535
echo "[chain] ===== ALL DONE: sedan -> suv -> bus ($(date)) ====="
