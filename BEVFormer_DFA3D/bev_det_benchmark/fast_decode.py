"""Lightweight, NDS-exact image decode for the fast VP eval path.

Replicates the mmdet test pipeline's image ops EXACTLY (verified bit-identical to
the dataloader output, NDS diff 0.0000), but with NO DataContainer/collate/DDP and
runnable in a ProcessPool so JPEG decode overlaps with the GPU forward:
    LoadMultiViewImageFromFiles(to_float32) -> mmcv.imread().astype(float32)  (BGR)
    NormalizeMultiviewImage(mean,std,to_rgb) -> mmcv.imnormalize(..., to_rgb=True)
    RandomScaleImageMultiViewImage([0.5])    -> mmcv.imresize(img, (800, 450))
    PadMultiViewImage(size_divisor=32)       -> mmcv.impad_to_multiple(img, 32)
Imported by the ProcessPool workers, so kept free of torch/mmdet (only numpy+mmcv).
"""
import os
import mmcv
import numpy as np

# Optional RAM-staging: if VP_STAGE_ROOT is set, read images from the staged copy
# (tmpfs) instead of Lustre. The staged tree preserves the 'sweeps/...' subpath
# under {STAGE}/sedan (carla_geobev baseline) and {STAGE}/vr (carla_VR variants).
# Bit-identical bytes -> NDS unchanged; only the read source (RAM vs network) differs.
_STAGE = os.environ.get('VP_STAGE_ROOT')


def _resolve(path):
    if not _STAGE:
        return path
    i = path.find('sweeps/')
    if i < 0:
        return path
    sub = path[i:]
    root = 'vr' if '/carla_VR/' in path else 'sedan'
    return os.path.join(_STAGE, root, sub)


MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
# RandomScaleImageMultiViewImage(0.5) scales the pixel axes of lidar2img by 0.5.
SCALE4 = np.eye(4, dtype=np.float64)
SCALE4[0, 0] = 0.5
SCALE4[1, 1] = 0.5


def decode_sample(info):
    """info dict (with ['cams'][cam]['data_path']) -> (6, 3, 480, 800) float32 array.

    Module-level + numpy-only return so it pickles cheaply across a ProcessPool.
    """
    imgs = []
    for cam in CAMS:
        img = mmcv.imread(_resolve(info['cams'][cam]['data_path'])).astype(np.float32)  # BGR 900x1600
        img = mmcv.imnormalize(img, MEAN, STD, to_rgb=True)                   # BGR->RGB, norm
        img = mmcv.imresize(img, (800, 450))                                  # ->450x800
        img = mmcv.impad_to_multiple(img, 32)                                 # ->480x800
        imgs.append(img.transpose(2, 0, 1))
    return np.stack(imgs)                                                     # (6,3,480,800)


# --------------------------------------------------------------------------- #
# Shared-memory decode: workers write the decoded (6,3,480,800) float32 DIRECTLY
# into a pre-allocated shared-memory slot and return only the slot index. This
# eliminates the 27MB pickle/unpickle per sample that the GIL otherwise serialises
# against the GPU forward, so decode (in the pool) overlaps the forward fully
# (infer -> forward-bound). Bytes are identical to decode_sample -> NDS unchanged.
# --------------------------------------------------------------------------- #
import multiprocessing.shared_memory as _shm_mod  # noqa: E402

SHAPE = (6, 3, 480, 800)
NBYTES = 6 * 3 * 480 * 800 * 4

_W_VIEWS = None  # per-worker list of numpy views over the shared slots


def _init_shm_worker(names):
    global _W_VIEWS
    # 1 cv2 thread per decode worker: N workers x cv2's own thread pool would
    # otherwise oversubscribe the cores and starve the concurrent eval process.
    try:
        mmcv.imread  # ensure cv2 backend loaded
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    shms = [_shm_mod.SharedMemory(name=n) for n in names]
    _W_VIEWS = [np.ndarray(SHAPE, dtype=np.float32, buffer=s.buf) for s in shms]
    _init_shm_worker._shms = shms          # keep refs alive (prevent GC/close)


def _ping(x):
    return x


def decode_to_shm(args):
    """Decode 6 cams of `info` straight into shared slot `slot`; return slot.

    Identical pixel ops to decode_sample (imread->imnormalize->imresize->impad),
    only the destination differs (shm view vs a fresh array)."""
    info, slot = args
    a = _W_VIEWS[slot]
    for ci, cam in enumerate(CAMS):
        img = mmcv.imread(_resolve(info['cams'][cam]['data_path'])).astype(np.float32)
        img = mmcv.imnormalize(img, MEAN, STD, to_rgb=True)
        img = mmcv.imresize(img, (800, 450))
        img = mmcv.impad_to_multiple(img, 32)
        a[ci] = img.transpose(2, 0, 1)
    return slot


class ShmPool:
    """A persistent ProcessPool whose workers decode into N_SLOTS shared buffers.

    Created once (before CUDA init) and reused across all cells. .submit(info, slot)
    decodes into views[slot]; the caller manages slot lifetime (a sliding window)."""

    def __init__(self, workers, n_slots=None):
        from concurrent.futures import ProcessPoolExecutor
        self.workers = workers
        self.n_slots = int(n_slots or (workers + 8))
        self.shms = [_shm_mod.SharedMemory(create=True, size=NBYTES)
                     for _ in range(self.n_slots)]
        self.views = [np.ndarray(SHAPE, dtype=np.float32, buffer=s.buf)
                      for s in self.shms]
        names = [s.name for s in self.shms]
        self.pool = ProcessPoolExecutor(max_workers=workers,
                                        initializer=_init_shm_worker,
                                        initargs=(names,))
        # Force all workers to fork+init NOW (before the parent touches CUDA) so we
        # never fork a CUDA-initialised process. chunksize=1 spreads across workers.
        list(self.pool.map(_ping, range(workers * 4), chunksize=1))

    def submit(self, info, slot):
        return self.pool.submit(decode_to_shm, (info, slot))

    def shutdown(self):
        try:
            self.pool.shutdown(wait=True)
        finally:
            for s in self.shms:
                try:
                    s.close()
                    s.unlink()
                except FileNotFoundError:
                    pass


def manual_img_metas(dataset, idx):
    """Build the img_metas the detector needs WITHOUT the dataloader/pipeline:
    get_data_info (unscaled lidar2img + can_bus + scene_token) + the x0.5 scale +
    padded img_shape + can_bus ego-shift zeroed (as forward_test does). Verified to
    give NDS identical to the dataloader-produced img_metas (diff 0.0000)."""
    di = dataset.get_data_info(idx)
    l2i = [SCALE4 @ np.asarray(x, dtype=np.float64) for x in di['lidar2img']]
    cb = np.asarray(di['can_bus'], dtype=np.float64).copy()
    cb[:3] = 0.0
    cb[-1] = 0.0
    return dict(lidar2img=l2i,
                img_shape=[(480, 800, 3)] * 6,
                ori_shape=[(480, 800, 3)] * 6,
                pad_shape=[(480, 800, 3)] * 6,
                scale_factor=1.0,
                flip=False,
                pcd_horizontal_flip=False,
                pcd_vertical_flip=False,
                box_type_3d=dataset.box_type_3d,
                box_mode_3d=dataset.box_mode_3d,
                scene_token=di['scene_token'],
                prev_bev_exists=di.get('prev_bev_exists', False),
                can_bus=cb,
                lidar2cam=di.get('lidar2cam'),
                cam_intrinsic=di.get('cam_intrinsic'),
                img_norm_cfg=dict(mean=MEAN, std=STD, to_rgb=True))
