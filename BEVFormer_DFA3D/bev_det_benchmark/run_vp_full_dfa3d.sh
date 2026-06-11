#!/usr/bin/env bash
# Full-data VP robustness for DFA3D on 2 GPUs (cell-sharded), NDS-exact FAST path,
# then merge. Mirrors the main repo's run_vp_full_nostage.sh: fast_decode reads
# images straight from data/nuscenes/sweeps (Lustre) via the original relative
# data_path (VP_STAGE_ROOT unset). Per-cell checkpointing makes this crash-
# resumable: rerun with the same TAG and finished cells are skipped.
#
# Usage: run_vp_full_dfa3d.sh [WORKERS] [TAG] [FPS]
set -e
cd /home/hanyan_arch/viewpoint/BEVFormer/3D-deformable-attention/BEVFormer_DFA3D
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bevformer-b200

WORKERS=${1:-6}                       # 6/shard = 12 Lustre readers (<16 thrash bound)
TAG=${2:-dfa3d_sedan_full}
FPS=${3:-79}                          # frames-per-scene (79=all 3792; 16=768 subset)
CFG=projects/configs/bevformer/bevformer_DFA3D_carla.py
CKPT=work_dirs/bevformer_DFA3D_carla/epoch_24.pth
A="--config $CFG --ckpt $CKPT --frames-per-scene $FPS --protocol both --fast --batch 1 --workers $WORKERS --tag $TAG"
OUT=bev_det_benchmark/out
mkdir -p "$OUT"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset VP_STAGE_ROOT || true
echo "NO-STAGE (Lustre reads)  workers=$WORKERS  frames-per-scene=$FPS  OMP=$OMP_NUM_THREADS"

echo "================ DFA3D VP FULL START $(date)  (2-GPU, FAST+pipeline, workers $WORKERS, NUMA-pinned) ================"
CUDA_VISIBLE_DEVICES=0 PORT=29910 numactl --cpunodebind=0 \
    bash bev_det_benchmark/run_vp_dfa3d.sh $A --shard 0/2 \
    > $OUT/vp_${TAG}_shard0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 PORT=29911 numactl --cpunodebind=1 \
    bash bev_det_benchmark/run_vp_dfa3d.sh $A --shard 1/2 \
    > $OUT/vp_${TAG}_shard1.log 2>&1 &
P1=$!
echo "shard0 pid=$P0 (GPU0)  shard1 pid=$P1 (GPU1)"
wait $P0; E0=$?
wait $P1; E1=$?
echo "shard0 exit=$E0  shard1 exit=$E1"

echo "================ MERGE $(date) ================"
python bev_det_benchmark/eval_vp_robustness_det.py --merge --tag $TAG --protocol both
echo "================ DFA3D VP FULL DONE $(date) ================"
cat $OUT/vp_${TAG}/eval_vp_summary.txt 2>/dev/null
