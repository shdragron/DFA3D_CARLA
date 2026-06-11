#!/usr/bin/env bash
# Single-GPU DFA3D-CARLA training (BEVFormer + depth-aware 3D deformable attention).
#
# Fair single-variable comparison vs BEVFormer-tiny CARLA: same R50/BEV-50x50/
# single-frame/6-class/no-aug protocol, global batch 16 (samples_per_gpu=16 on 1 GPU,
# frozen BN => equivalent to the baseline's 2-GPU x bs8), lr 4e-4, 24 epochs.
#
# Usage:  CUDA_VISIBLE_DEVICES=1 bash tools/train_DFA3D_carla.sh <config> [PORT]
#   e.g.  CUDA_VISIBLE_DEVICES=1 bash tools/train_DFA3D_carla.sh \
#             projects/configs/bevformer/bevformer_DFA3D_carla.py 28533
set -eu
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

CFG=${1:-projects/configs/bevformer/bevformer_DFA3D_carla.py}
PORT=${2:-28533}
GPUS=${3:-1}
PY=${PY:-/NHNHOME/WORKSPACE/0526040099_A/giyong/miniconda3/envs/bevformer-b200/bin/python}
NAME=$(basename "$CFG" .py)
WORKDIR=work_dirs/${NAME}
mkdir -p logs "$WORKDIR"

echo "[DFA3D] config=$CFG  port=$PORT  GPUS=$GPUS  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
PYTHONPATH="$HERE":${PYTHONPATH:-} \
"$PY" -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$PORT" \
    tools/train.py "$CFG" \
    --launcher pytorch \
    --work-dir "$WORKDIR" \
    --deterministic 2>&1 | tee "logs/train_${NAME}.log"
