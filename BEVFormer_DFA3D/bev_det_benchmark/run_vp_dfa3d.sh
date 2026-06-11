#!/usr/bin/env bash
# Launch the in-process VP robustness driver (loads DFA3D once, loops the
# 631-cell grid). Extra args are forwarded to eval_vp_robustness_det.py.
#
#   CUDA_VISIBLE_DEVICES=1 run_vp_dfa3d.sh [--frames-per-scene N --tag ... ]
#
# No `set -u`: conda's cuda-nvcc activate.d hook references unset NVCC_PREPEND_FLAGS.
set -e

DFA3D_ROOT=/home/hanyan_arch/viewpoint/BEVFormer/3D-deformable-attention/BEVFormer_DFA3D
ENV=bevformer-b200

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV"
fi
cd "$DFA3D_ROOT"

PORT=${PORT:-$((29870 + RANDOM % 200))}
PYTHONPATH="$DFA3D_ROOT:$PYTHONPATH" python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port="$PORT" \
    bev_det_benchmark/eval_vp_robustness_det.py "$@"
