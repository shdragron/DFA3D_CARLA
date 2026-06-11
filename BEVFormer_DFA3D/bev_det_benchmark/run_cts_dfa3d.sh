#!/usr/bin/env bash
# DFA3D cross-platform transfer (CTS): sedan model on suv/bus under
# NORMAL/EXT/IMG/CAL + per-target ORACLE (denominator). Full 3792-frame val,
# 10 dist_test subprocesses total; per-cell resume via out/cts_<TAG>/cells/.
#
# Usage: [CUDA_VISIBLE_DEVICES=0] run_cts_dfa3d.sh [TAG]
set -e -o pipefail
cd /home/hanyan_arch/viewpoint/BEVFormer/3D-deformable-attention/BEVFormer_DFA3D
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bevformer-b200

TAG=${1:-dfa3d_sedan}
mkdir -p bev_det_benchmark/out
python bev_det_benchmark/eval_cts_det.py \
    --config projects/configs/bevformer/bevformer_DFA3D_carla.py \
    --ckpt work_dirs/bevformer_DFA3D_carla/epoch_24.pth \
    --target-ckpt-tmpl 'work_dirs/bevformer_DFA3D_carla_{}/epoch_24.pth' \
    --ngpu 1 --tag "$TAG" \
    2>&1 | tee bev_det_benchmark/out/cts_driver_${TAG}.log
