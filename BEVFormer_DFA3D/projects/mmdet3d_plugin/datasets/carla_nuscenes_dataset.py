"""CARLA-adapted NuScenes dataset for BEVFormer.

Wraps CustomNuScenesDataset and overrides _evaluate_single() so that the
standard nuscenes-devkit detection evaluator can run on the CARLA jeongtae
v1.0-carla_eval data without modifying nuScenes site-packages or the original
BEVFormer code.

Three small patches happen at evaluate() time only:

  1. eval_set_map: 'v1.0-carla_eval' -> 'val'
  2. nuscenes.utils.splits: dynamically extend splits['val'] with the CARLA
     scene names actually present in the eval DB.
  3. nuscenes-devkit load_gt(): the assertion `version.endswith('trainval')`
     is bypassed via a monkey-patch on the helper that maps eval_split ->
     version requirement, scoped to the evaluate() call only.

NDS / mAP / per-class AP / mATE / mASE / mAOE / mAVE / mAAE all come out
exactly the same as nuscenes-devkit computes them, just on our 6 CARLA classes.

Original code (projects/mmdet3d_plugin/datasets/nuscenes_dataset.py,
projects/mmdet3d_plugin/datasets/nuscnes_eval.py, and nuscenes-devkit) is
NOT modified.
"""
from os import path as osp

import mmcv
from mmdet.datasets import DATASETS

from .nuscenes_dataset import CustomNuScenesDataset
from .nuscnes_eval import NuScenesEval_custom

# CARLA GT-validity rule for eval: keep annotations with visibility_token >= this.
# Matches training (use_valid_flag=True, where valid_flag == visibility>=2) and
# BEVDepth's eval, instead of the devkit's default num_pts>0 rule.
CARLA_MIN_VISIBILITY = 2


def _patch_nuscenes_eval_for_carla(carla_val_scene_names):
    """Install monkey-patches needed for nuscenes-devkit eval on CARLA.

    Two issues to overcome:

    1. nuscnes_eval.py (BEVFormer's own copy) defines its own load_gt() at
       module level (line 244) that re-implements the devkit logic. It
       uses DetectionBox_modified and reads splits via
       create_splits_scenes(). That custom load_gt itself enforces
       `nusc.version.endswith('trainval')`, which fails for
       v1.0-carla_eval. We patch that assertion via wrapping load_gt
       to temporarily flip nusc.version while delegating to the ORIGINAL
       custom load_gt (not the devkit one — DetectionBox_modified is not
       compatible with devkit's load_gt).

    2. create_splits_scenes() returns a dict whose 'val' entry is the
       hard-coded nuScenes val scene names. We replace it with the
       CARLA val scene names so load_gt picks up the correct samples.

    Both names are bound at import time inside nuscnes_eval, so we set the
    module attributes directly on that module (in addition to the devkit
    modules, for completeness).

    Returns a callable that restores the originals when called.
    """
    from nuscenes.utils import splits as nusc_splits
    from nuscenes.eval.common import loaders as nusc_loaders
    from projects.mmdet3d_plugin.datasets import nuscnes_eval as _bev_eval_mod

    original_create = nusc_splits.create_splits_scenes
    original_devkit_load_gt = nusc_loaders.load_gt
    # IMPORTANT: keep a reference to the BEVFormer-custom load_gt, not the
    # devkit one — the custom version handles DetectionBox_modified properly.
    original_bev_load_gt = _bev_eval_mod.load_gt

    def patched_create_splits_scenes(verbose=False):
        d = original_create(verbose=verbose)
        # Replace 'val' split entirely with the CARLA val scene names.
        d = {k: list(v) for k, v in d.items()}
        d['val'] = list(carla_val_scene_names)
        return d

    def patched_load_gt(nusc, eval_split, box_cls, verbose=False):
        # Bypass the version assertion inside the custom load_gt by
        # temporarily flipping nusc.version. nusc.scene / nusc.sample are
        # unchanged so the split lookup still matches CARLA scenes.
        real_version = nusc.version
        try:
            if not real_version.endswith('trainval'):
                nusc.version = 'v1.0-trainval'
            gt = original_bev_load_gt(
                nusc, eval_split, box_cls, verbose=verbose)
        finally:
            nusc.version = real_version
        # CARLA eval validity = visibility, not num_pts: keep only GT with
        # visibility_token >= CARLA_MIN_VISIBILITY (== training's valid_flag),
        # and force num_pts=1 so filter_eval_boxes' num_pts==0 removal is a
        # no-op. This makes the eval GT match TRAINING and BEVDepth exactly.
        from nuscenes.eval.common.data_classes import EvalBoxes
        filtered = EvalBoxes()
        for st in gt.sample_tokens:
            keep = []
            for b in gt[st]:
                vis = getattr(b, 'visibility', None)
                if vis is None or int(vis) >= CARLA_MIN_VISIBILITY:
                    b.num_pts = 1
                    keep.append(b)
            filtered.add_boxes(st, keep)
        return filtered

    # Install patches.
    nusc_splits.create_splits_scenes = patched_create_splits_scenes
    nusc_loaders.load_gt = patched_load_gt
    _bev_eval_mod.create_splits_scenes = patched_create_splits_scenes
    _bev_eval_mod.load_gt = patched_load_gt

    def restore():
        nusc_splits.create_splits_scenes = original_create
        nusc_loaders.load_gt = original_devkit_load_gt
        _bev_eval_mod.create_splits_scenes = original_create
        _bev_eval_mod.load_gt = original_bev_load_gt

    return restore


@DATASETS.register_module()
class CarlaNuScenesDataset(CustomNuScenesDataset):
    """CARLA dataset using nuScenes DB format (v1.0-carla / v1.0-carla_eval).

    Inherits all loading/training behavior from CustomNuScenesDataset.  Only
    the eval helper _evaluate_single() is overridden so the nuscenes-devkit
    metric pipeline accepts our custom version string and our scene names.
    """

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        from nuscenes import NuScenes

        self.nusc = NuScenes(version=self.version,
                             dataroot=self.data_root,
                             verbose=False)
        output_dir = osp.join(*osp.split(result_path)[:-1])

        # Map our custom versions to nuscenes-devkit's expected eval_split.
        # All CARLA versions evaluate against the 'val' split: the per-vehicle
        # eval DBs (v1.0-carla_{sedan,suv,bus}_eval), the train DBs
        # (v1.0-carla_{veh}, in case someone evaluates the train set), and the
        # legacy single-DB names (v1.0-carla / v1.0-carla_eval).
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        if self.version in eval_set_map:
            eval_split = eval_set_map[self.version]
        elif 'carla' in self.version:
            eval_split = 'val'
        else:
            raise KeyError(
                f'Unknown dataset version for eval: {self.version!r}')

        # Restrict eval GT to exactly the scenes present in THIS split's info
        # pkl. The per-vehicle eval DB may contain scenes that val.txt drops
        # (e.g. scene_0260): if load_gt used all DB scenes, the GT sample set
        # would exceed the predicted set and nuscenes-devkit's
        # `pred.sample_tokens == gt.sample_tokens` assertion would fail. We map
        # each pkl sample (info['token']) back to its real scene via the DB
        # (info['scene_token'] is overwritten to the sample token for the
        # temporal-off trick, so it can't be used here).
        eval_sample_tokens = {info['token'] for info in self.data_infos}
        eval_scene_tokens = {
            self.nusc.get('sample', t)['scene_token']
            for t in eval_sample_tokens
        }
        carla_val_scene_names = [
            self.nusc.get('scene', st)['name'] for st in eval_scene_tokens
        ]

        restore = _patch_nuscenes_eval_for_carla(carla_val_scene_names)
        try:
            # Diagnostic: verify the patched splits actually reach the eval module.
            from projects.mmdet3d_plugin.datasets import (
                nuscnes_eval as _bev_eval_mod,
            )
            _splits = _bev_eval_mod.create_splits_scenes()
            print(f'[CARLA-EVAL] dataset version={self.version}  '
                  f'eval_split={eval_split}  '
                  f'splits[{eval_split}] size={len(_splits.get(eval_split, []))}  '
                  f'sample first={_splits.get(eval_split, [None])[0]}  '
                  f'carla scene count={len(carla_val_scene_names)}  '
                  f'carla first={carla_val_scene_names[0] if carla_val_scene_names else None}')

            self.nusc_eval = NuScenesEval_custom(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_split,
                output_dir=output_dir,
                verbose=True,
                overlap_test=self.overlap_test,
                data_infos=self.data_infos,
            )

            # Diagnostic just before the assertion fires.
            pred_tokens = set(self.nusc_eval.pred_boxes.sample_tokens)
            gt_tokens = set(self.nusc_eval.gt_boxes.sample_tokens)
            print(f'[CARLA-EVAL] pred_tokens={len(pred_tokens)}  '
                  f'gt_tokens={len(gt_tokens)}  '
                  f'intersection={len(pred_tokens & gt_tokens)}  '
                  f'pred_only={len(pred_tokens - gt_tokens)}  '
                  f'gt_only={len(gt_tokens - pred_tokens)}')
            if pred_tokens != gt_tokens:
                missing_from_gt = list(pred_tokens - gt_tokens)[:5]
                missing_from_pred = list(gt_tokens - pred_tokens)[:5]
                print(f'[CARLA-EVAL] sample pred-not-in-gt: {missing_from_gt}')
                print(f'[CARLA-EVAL] sample gt-not-in-pred: {missing_from_pred}')

            self.nusc_eval.main(plot_examples=0, render_curves=False)
        finally:
            restore()

        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        # nuscenes-devkit averages mAP/NDS over all 10 detection_cvpr_2019
        # classes; the 4 absent in CARLA score AP=0 and dilute the result.
        # DetectionConfig hard-asserts 10 classes, so instead we recompute
        # mAP/NDS over exactly our classes using the devkit formula
        # (NDS = (5*mAP + sum max(0,1-nanmean(TP)))/10). This is the clean
        # 6-class score and matches BEVDepth's CARLA eval.
        import numpy as np
        tp_keys = list(metrics['tp_errors'].keys())  # trans/scale/orient/vel/attr
        mean_dist_aps = [np.mean(list(metrics['label_aps'][c].values()))
                         for c in self.CLASSES]
        mAP6 = float(np.mean(mean_dist_aps))
        # 6-class TP error per component (nanmean over our classes) and its
        # score max(0, 1-err); these are the exact ingredients of the 6-class NDS.
        tp_errs6 = {m: float(np.nanmean(
            [metrics['label_tp_errors'][c][m] for c in self.CLASSES]))
            for m in tp_keys}
        tp_scores = [max(0.0, 1.0 - tp_errs6[m]) for m in tp_keys]
        nds6 = (5.0 * mAP6 + float(np.sum(tp_scores))) / (5.0 + len(tp_keys))
        detail['{}/NDS'.format(metric_prefix)] = nds6
        detail['{}/mAP'.format(metric_prefix)] = mAP6
        # 6-class TP error components (the unsuffixed mATE/.. keys set in the
        # per-class loop above are the 10-class devkit values; these match NDS6).
        for m in tp_keys:
            detail['{}/{}_6class'.format(
                metric_prefix, self.ErrNameMapping[m])] = round(tp_errs6[m], 4)
        detail['{}/NDS_10class'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP_10class'.format(metric_prefix)] = metrics['mean_ap']
        print(f'[CARLA-EVAL] {len(self.CLASSES)}-class mAP={mAP6:.4f} '
              f'NDS={nds6:.4f} | 10-class mAP={metrics["mean_ap"]:.4f} '
              f'NDS={metrics["nd_score"]:.4f}')
        # Full metric dump on one stdout line so the robustness drivers can scrape
        # every number (per-class AP at all dist thresholds, per-class/6-class/
        # 10-class TP errors, mAP, NDS) without re-running the eval.
        import json as _json
        print('[CARLA-METRICS-JSON] ' + _json.dumps(detail))
        return detail
