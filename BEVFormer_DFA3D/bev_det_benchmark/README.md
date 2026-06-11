# DFA3D robustness benchmark (VP + CTS)

Port of the main-repo `bev_det_benchmark` drivers into this fork. The drivers
MUST live inside this repo (the two repos' `projects.mmdet3d_plugin` packages
cannot be imported in one process: same module name + duplicate registry names),
and all runs require `cwd` = this repo root (relative `data/nuscenes/` paths;
the symlink points at the same carla_geobev as the main repo).

Depth note: DFA3D predicts depth at test time (`pts_dpt_head(..., return_dpt=False)`
inside `simple_test_pts`); the DPT png files are train-only supervision
(`loss_dpt`). The test pipeline is byte-identical to `bevformer_tiny_carla`
(1600x900 -> x0.5 -> pad to 480x800, same normalize), so the `--fast` decode
path applies unchanged. Do NOT add a depth-loading step to eval.

Env: `bevformer-b200` + the editable `dfa3D` ext package
(`3D-deformable-attention/DFA3D`, compiled `_ext`).

## Files
- `eval_vp_robustness_det.py` / `build_condition_pkls.py` / `fast_decode.py` /
  `eval_cts_det.py` — copies of the main-repo drivers; the ONLY edits are
  (a) `build_condition_pkls.BEVF_ROOT` -> this repo root,
  (b) CLI defaults (DFA3D config, `epoch_24.pth` ckpts, tag `dfa3d_sedan`,
  `--ngpu 1`), (c) CTS runner -> `run_dfa3d.sh`. Keep them in sync with the
  main repo when the originals change.
- Port-review fixes (2026-06-11, applied to BOTH repos where noted):
  (i) fork `tools/test.py` now accepts `--local-rank` AND `--local_rank`
  (torch>=2 launcher passes the dashed form; without this every CTS dist_test
  subprocess died at argparse);
  (ii) the fast-path eval tmp dir is now per-tag `/tmp/vpeval_<tag>_shard<i>`
  in BOTH repos' drivers (the old `/tmp/vpeval_shard<i>` was shared across
  repos -> concurrent VP runs could silently cross-score each other's cells);
  (iii) `run_dfa3d.sh` CTS port range moved to 30100-30299 (clear of the VP
  full launcher's pinned 29910/29911 and a live listener on 30001).
- `run_dfa3d.sh`        — CTS per-cell runner (tools/dist_test.sh wrapper).
- `run_vp_dfa3d.sh`     — VP driver launcher (torch.distributed.launch, 1 proc).
- `run_vp_full_dfa3d.sh`— full-val VP, 2 GPU shards + merge, crash-resumable.
- `run_cts_dfa3d.sh`    — full CTS (suv+bus oracles + 8 condition cells).

## Run
```bash
# smoke (1 frame/scene, one VR cell):
CUDA_VISIBLE_DEVICES=0 bash bev_det_benchmark/run_vp_dfa3d.sh \
    --frames-per-scene 1 --conditions VR --axes pitch --mags -8 \
    --protocol allcam --fast --workers 4 --tag dfa3d_smoke

# full VP (3792 frames, 631 cells, 2 GPUs, resumable by re-running):
bash bev_det_benchmark/run_vp_full_dfa3d.sh 6 dfa3d_sedan_full 79

# full CTS (oracle + NORMAL/EXT/IMG/CAL x suv/bus):
CUDA_VISIBLE_DEVICES=0 bash bev_det_benchmark/run_cts_dfa3d.sh
```

Outputs land in `bev_det_benchmark/out/{vp_<tag>,cts_<tag>}/` with the same
`eval_vp.json` / `eval_cts.json` schemas as every other model in the benchmark
(RRS = NDS_cell/NDS_normal; CTS = NDS_cond/NDS_target-oracle; 6-class CARLA NDS,
GT visibility>=2). Copy the merged outputs to `results/DFA3D/{vp,cts}/` in the
main repo.

Gotchas inherited from the main drivers (do not "fix" these):
- VP image swap uses carla_VR frame index `2N` for geobev frame `N` (already
  encoded in `build_condition_pkls.vr_image_path`).
- VP swaps `sensor2lidar_*` only; `sensor2ego_*` stays stale by design (the
  dataset builds lidar2img from sensor2lidar + intrinsics; sensor2ego unused).
- `--merge` must repeat the shard runs' `--protocol/--tag/--outdir`.
- Resume reuses `cells_shard*/` keyed by tag+shard; changing ckpt/config without
  changing `--tag` silently reuses stale cells.
- `--fast --batch 1` only (batch>1 is not NDS-exact).
