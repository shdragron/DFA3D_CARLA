DFA3D-CARLA
========

**DFA3D-enabled BEVFormer adapted to the CARLA / GeoBEV camera-geometry robustness benchmark.**

This is a research fork of [DFA3D (3D Deformable Attention, ICCV 2023)](https://github.com/IDEA-Research/3D-deformable-attention).
It ports the DFA3D-enabled BEVFormer to a CARLA-rendered nuScenes-format dataset so that
**BEVFormer (extrinsic-gated sampling, no depth)** and **BEVFormer + DFA3D (extrinsic-gated
sampling, *with* depth-aware 3D feature lifting)** can be compared as a clean single-variable
study. In the GeoBEV benchmark this is the 7th detector and fills the otherwise-empty
*gates-sampling × uses-depth* quadrant of the architecture matrix.

> **Attribution / License.** This repository is derived from
> [`https://github.com/IDEA-Research/3D-deformable-attention`](https://github.com/IDEA-Research/3D-deformable-attention),
> which is licensed under the **IDEA License 1.0, Copyright (c) IDEA. All Rights Reserved.**
> The same license applies to this fork (see [`LICENSE`](LICENSE)); use is limited to
> non-commercial research. The DFA3D CUDA operator and the `BEVFormer_DFA3D` model code are
> the original authors' work. BEVFormer is © its authors
> ([fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer)). The original
> project README is preserved at [`README_DFA3D.md`](README_DFA3D.md), and the upstream paper
> should be cited (see [Citation](#citation)).

---

## What this fork adds

All changes live in `BEVFormer_DFA3D/`; the upstream `DFA3D/` CUDA operator is unchanged except
for a one-line C++ standard bump needed by newer PyTorch (see Installation).

| Area | File | Change |
|---|---|---|
| Config | `projects/configs/bevformer/bevformer_DFA3D_carla.py` | CARLA single-variable config: R50 (ImageNet), BEV 50×50, single-frame, 6 CARLA classes, no aug/EMA/CBGS, fp32, single FPN level. DFA3D depth head (`DepthHead_MLVGDpt`) + `SpatialCrossAttention_DFA3D` grafted at FPN level 0. |
| Dataset | `projects/mmdet3d_plugin/datasets/carla_nuscenes_dataset.py` | `CarlaNuScenesDataset` — runs the nuScenes-devkit detection metrics on the CARLA eval DB without modifying the devkit (custom version string + scene-name + visibility-based GT validity). Subclasses the depth-aware `CustomNuScenesDataset` so the depth map is carried through the temporal queue. |
| Depth GT | `projects/mmdet3d_plugin/datasets/pipelines/loading.py::CarlaDPTMultiViewDepthDFA3D` | Loads CARLA dense **DPT** depth (RGB→DPT path, decode `(R + G·256 + B·256²)/(256³−1)·1000` planar-Z meters) into `results['dpt']`, aligned to the multi-view images. |
| Pipeline | `projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py::RandomScaleImageMultiViewImageDpt` | Used so the depth map is scaled (nearest) together with its image; `PadMultiViewImage` then pads both — keeping pixel-for-pixel depth↔image alignment into the depth head. |
| Launcher | `tools/train_DFA3D_carla.sh` | Single-GPU launch matching the BEVFormer baseline's global batch and LR. |

### Fair-comparison protocol

To keep the comparison to the no-depth BEVFormer baseline a true single-variable one (the only
addition is DFA3D's depth-aware lifting), this config matches the baseline exactly: same backbone,
BEV grid, single-frame setup, classes, augmentation policy, optimizer, schedule, and **global batch
16 at lr 4e-4** (here `samples_per_gpu=16` on one GPU; with frozen backbone BN this is equivalent
to the baseline's 2-GPU × bs8).

### Verified

Build, a forward+backward training step (detection losses **and** the depth loss), and inference
all run on the CARLA data. Coordinate system checked: GT-box projection rate via `lidar2img` = 1.000
(matches the BEVFormer baseline), and the loaded DPT depth is in the same metric units and
pixel-aligned to the scaled/padded image (median DPT-surface / box-center-Z ≈ 0.99).

---

## Installation (B200 / CUDA 12.8 notes)

Upstream targets `pytorch=1.9.1, cuda=11.1`. This fork was run on an NVIDIA B200
(`torch 2.x, cuda 12.8, sm_100`). Two adjustments:

1. **Compile the DFA3D operator with C++17** (newer PyTorch headers require it) and a CUDA-compatible
   host compiler:
   ```sh
   cd DFA3D
   sed -i 's/c++14/c++17/g' setup.py
   TORCH_CUDA_ARCH_LIST=10.0 CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 \
       pip install -e . --no-build-isolation
   cd .. && python unittest_DFA3D.py     # silent + exit 0 == pass
   ```
2. The `--local-rank` / `--local_rank` argument spelling is accepted by `tools/train.py` for
   `torch>=2` `torch.distributed.launch`.

For the base operator details and the original nuScenes model zoo, see [`README_DFA3D.md`](README_DFA3D.md).

## Data

Point `BEVFormer_DFA3D/data/nuscenes` at the CARLA GeoBEV dataset (nuScenes DB format, per-vehicle
`{vehicle}_infos_{train,val}.pkl`). Dense DPT depth images live under `sweeps/DPT-*` next to the
`sweeps/RGB-*` images and are loaded automatically by the depth pipeline (no separate `depth_gt/`
download is needed for CARLA).

## Train

```sh
cd BEVFormer_DFA3D
CUDA_VISIBLE_DEVICES=0 bash tools/train_DFA3D_carla.sh \
    projects/configs/bevformer/bevformer_DFA3D_carla.py 28533
```

## Eval

```sh
cd BEVFormer_DFA3D
bash tools/dist_test.sh \
    projects/configs/bevformer/bevformer_DFA3D_carla.py path_to_checkpoint 1 --eval bbox
```

## Citation

If you use this code, please cite the original DFA3D paper and BEVFormer.

```bibtex
@inproceedings{li2023dfa3d,
  title={DFA3D: 3D Deformable Attention For 2D-to-3D Feature Lifting},
  author={Hongyang Li and Hao Zhang and Zhaoyang Zeng and Shilong Liu and Feng Li and Tianhe Ren and Lei Zhang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
@article{li2022bevformer,
  title={BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2203.17270},
  year={2022}
}
```
