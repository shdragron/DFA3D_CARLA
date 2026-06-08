# ------------------------------------------------------------------------
# DFA3D
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the IDEA License, Version 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from mmcv (https://github.com/open-mmlab/mmcv)
# Copyright (c) OpenMMLab. All rights reserved
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright 2018-2019 Open-MMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------ 
# Modified from bevdepth (https://github.com/Megvii-BaseDetection/BEVDepth)
# Copyright (c) 2022 Megvii-BaseDetection
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------ 
#  Modified by Hongyang Li
# ---------------------------------------------

import os
import pdb
import torch
import numpy as np
import mmcv
from PIL import Image
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class CarlaDPTMultiViewDepthDFA3D(object):
    """Dense CARLA DPT depth GT for the DFA3D DepthHead.

    Produces ``results['dpt']`` = list of per-camera dense depth maps (meters,
    float32, one (H, W) array per view at the CURRENT image resolution), exactly
    parallel to ``results['img']``. It is inserted BEFORE
    RandomScaleImageMultiViewImageDpt + PadMultiViewImage so the depth map is
    scaled (nearest) and padded identically to its RGB image, keeping
    pixel-for-pixel alignment all the way to the DepthHead (which then
    down-samples by ``downsample_factor`` and quantizes into ``dbound`` bins).

    DPT source (same as BEVDet/BEVDepth's CarlaDPTMultiViewDepth, verified):
    an RGB-encoded PNG at the original image size storing PLANAR Z-depth
    (median DPT/z == 0.998 vs LiDAR, so NO range->Z conversion). The path is
    derived from the RGB path by RGB->DPT, .jpg->.png — which also selects the
    correct per-vehicle folder (RGB-CAM_FRONT->DPT-CAM_FRONT for sedan,
    RGB-suv-*->DPT-suv-*, RGB-bus-*->DPT-bus-*). Decode:
        depth_m = (R + G*256 + B*256^2) / (256^3 - 1) * 1000
    Sky/far ~ 1000 m falls outside dbound=[2,58] and is auto-masked by the head.
    """

    def __call__(self, results):
        img_paths = results['img_filename']
        imgs = results['img']
        map_depths = []
        dpt_paths = []
        for cid, rgb_path in enumerate(img_paths):
            dpt_path = rgb_path.replace('RGB', 'DPT').replace('.jpg', '.png')
            mmcv.check_file_exist(dpt_path)
            dpt = np.asarray(Image.open(dpt_path).convert('RGB'))
            r = dpt[..., 0].astype(np.float32)
            g = dpt[..., 1].astype(np.float32)
            b = dpt[..., 2].astype(np.float32)
            depth = (r + g * 256.0 + b * 256.0 ** 2) / (256.0 ** 3 - 1.0) * 1000.0
            # Align to the loaded image size (should already match; resize with
            # NEAREST so depth VALUES are preserved if the DPT png size differs).
            h, w = imgs[cid].shape[:2]
            if depth.shape[0] != h or depth.shape[1] != w:
                depth = mmcv.imresize(
                    depth, (w, h), interpolation='nearest')
            map_depths.append(depth.astype(np.float32))
            dpt_paths.append(dpt_path)
        results['dpt'] = map_depths
        results['filename_dpt'] = dpt_paths
        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'


@PIPELINES.register_module()
class LoadMultiViewDepthFromFiles(object):
    """Load the gound truth depth map generated from BEVDepth using lidar.
    """

    def __init__(self, is_to_depth_map=True, map_size=None):
        self.is_to_depth_map = is_to_depth_map
        self.map_size     = map_size
    def __call__(self, results):
        if self.map_size is None:
            self.map_size = results['img'][0].shape[:2]
        img_paths = results['img_filename']
        dpt_paths = []
        map_depths = []
        for img_path in img_paths:
            dpt_path = os.path.join(img_path.split("/samples/")[0], "depth_gt", img_path.split("/")[-1]+".bin")
            point_depth = np.fromfile(dpt_path, dtype=np.float32, count=-1).reshape(-1, 3)
            dpt_paths.append(dpt_path)
            if self.is_to_depth_map:
                map_depth = self.to_depth_map(point_depth)
                map_depths.append(map_depth)
        # img is of shape (h, w, c, num_views)
        results['dpt'] = map_depths
        results['filename_dpt'] = dpt_paths
        return results
    def to_depth_map(self, point_depth):
        """Transform depth based on ida augmentation configuration.

        Args:
            cam_depth (np array): Nx3, 3: x,y,d.
            resize (float): Resize factor.
            resize_dims (list): Final dimension.
            crop (list): x1, y1, x2, y2
            flip (bool): Whether to flip.
            rotate (float): Rotation value.

        Returns:
            np array: [h/down_ratio, w/down_ratio, d]
        """


        depth_coords = point_depth[:, :2].astype(np.int16)

        depth_map = np.zeros(self.map_size)
        valid_mask = ((depth_coords[:, 1] < self.map_size[0])
                    & (depth_coords[:, 0] < self.map_size[1])
                    & (depth_coords[:, 1] >= 0)
                    & (depth_coords[:, 0] >= 0))
        depth_map[depth_coords[valid_mask, 1],
                depth_coords[valid_mask, 0]] = point_depth[valid_mask, 2]

        return depth_map
