"""VP viewpoint-robustness eval for 3D detection (NDS analogue of
bev_seg_benchmark/eval_vp_robustness_cvt.py).

A sedan-trained model is evaluated on carla_VR viewpoint perturbations under 3
conditions x 3 axes x signed magnitudes x protocols:

    Normal  (no perturbation)               <- oracle / RRS denominator
    ER      extrinsic-only swap
    VR      image-only swap                  <- primary
    CR      both image + extrinsic
  axes        : yaw / pitch / roll
  magnitudes  : +/-{4,8,12,16,20}            (10 signed, from viewpoint_metadata)
  protocols   : per-cam (6, perturb ONE cam) + all-cam (1, perturb all 6)

Full grid = 1 + 3*3*10*7 = 631 conditions. Running each as a separate process
would reload the BEVFormer weights (~100 s) 631 times, so this driver loads the
model + the NuScenes GT DB ONCE and iterates conditions in-process, swapping
only the per-condition cam fields (build_condition_pkls.make_vp_infos) and
re-running inference + the stock 6-class-NDS eval each time.

Robustness score (mirrors aggregate_rrs in the seg script, NDS replacing IoU):
    RRS(cell) = NDS_cell / NDS_Normal
    mRRS_c    = mean RRS over per-cam protocols (+ axes + mags) for condition c
    RRSALL_c  = mean RRS over the all-cam protocol (+ axes + mags)
    mVRS_c    = 0.5 * (mRRS_c + RRSALL_c)

Launch (single GPU, in-process loop -- test.py blocks the non-distributed path,
so we still go through torch.distributed.launch with one proc):

    conda activate bevformer-b200
    CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
        --nproc_per_node=1 --master_port=29590 \
        bev_det_benchmark/eval_vp_robustness_det.py \
        --config projects/configs/bevformer/bevformer_tiny_carla.py \
        --ckpt   work_dirs/bevformer_tiny_carla_sedan/latest.pth \
        --frames-per-scene 4 --tag tiny_sedan

(run_vp_bevformer.sh wraps this.)
"""
import argparse
import copy
import csv
import importlib
import json
import os
import os.path as osp
import sys
import time

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

HERE = osp.dirname(osp.abspath(__file__))
BEVF_ROOT = osp.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, BEVF_ROOT)
import build_condition_pkls as B  # noqa: E402


def single_gpu_infer(model, data_loader):
    """In-memory single-GPU inference loop (replaces custom_multi_gpu_test).

    Avoids the DDP collect_results_cpu pickle round-trip + barrier + time.sleep(2)
    per condition. Returns the same ordered list[dict] dataset.evaluate expects.
    (Verified: the forward path is byte-identical to the DDP path, so NDS matches.
    NOTE: the real bottleneck is dataloader I/O, not this; expect only ~5-12%.)
    """
    model.eval()
    results = []
    prog = mmcv.ProgressBar(len(data_loader.dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        if isinstance(result, dict):              # mirror custom_multi_gpu_test
            result = result['bbox_results']
        results.extend(result)
        for _ in range(len(result)):
            prog.update()
    return results
from projects.mmdet3d_plugin.datasets.builder import build_dataloader  # noqa: E402

P = 'pts_bbox_NuScenes/'   # detail-dict key prefix
COMP_COLS = ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE', 'nds_10class', 'map_10class']


def nds_components(metrics):
    """6-class NDS ingredients (+10-class headline) pulled from a detail dict."""
    g = (lambda k: metrics.get(P + k)) if metrics else (lambda k: None)
    return {'mATE': g('mATE_6class'), 'mASE': g('mASE_6class'),
            'mAOE': g('mAOE_6class'), 'mAVE': g('mAVE_6class'),
            'mAAE': g('mAAE_6class'), 'nds_10class': g('NDS_10class'),
            'map_10class': g('mAP_10class')}


def patch_nuscenes_cache():
    """Cache NuScenes by (version, dataroot) so the GT DB loads only once.

    _evaluate_single does `NuScenes(version=self.version, ...)` every call; for
    VP the version (sedan) and the data_infos subset are identical across all
    631 conditions, so one cached instance is correct and saves ~all the per-
    condition DB-load time.
    """
    import nuscenes
    real = nuscenes.NuScenes
    cache = {}

    def cached(version=None, dataroot=None, verbose=False, **kw):
        key = (version, dataroot)
        if key not in cache:
            cache[key] = real(version=version, dataroot=dataroot,
                              verbose=verbose, **kw)
        return cache[key]

    nuscenes.NuScenes = cached


def patch_eval_subset(allowed_tokens):
    """Restrict eval GT to exactly our subset tokens.

    The carla eval restricts GT to the *scenes* present in data_infos, then
    load_gt pulls EVERY frame of those scenes. With a per-frame subset the GT
    sample set is a superset of the predictions and nuscenes-devkit's
    `pred==gt` assertion (nuscnes_eval.py:559) fires. Filtering gt_boxes down to
    the predicted (subset) tokens makes NDS well-defined over the subset; the
    subset is identical across all VP conditions, so RRS stays a fair ratio.
    """
    from projects.mmdet3d_plugin.datasets import nuscnes_eval as ne
    real_load_gt = ne.load_gt

    def load_gt_subset(nusc, eval_split, box_cls, verbose=False):
        gt = real_load_gt(nusc, eval_split, box_cls, verbose=verbose)
        for st in list(gt.boxes.keys()):
            if st not in allowed_tokens:
                del gt.boxes[st]
        return gt

    ne.load_gt = load_gt_subset


VISIBLE_TOKENS = {'2', '3', '4'}     # visibility >= 2 (>=40%), == training gt_visibility_min=2


def patch_eval_visibility(min_vis=VISIBLE_TOKENS):
    """Filter eval GT by VISIBILITY>=2, matching training (gt_visibility_min=2 /
    the pkl valid_flag), NOT by the nuscenes-devkit default of num_pts>0.

    filter_eval_boxes drops GT with num_pts==0, but CARLA's num_pts (lidar+radar)
    is UNRELATED to visibility, so the default eval GT (num_pts>0, ~145k boxes)
    differs massively from the training GT (visibility>=2, ~99k: 50.9k extra low-vis
    boxes + 5k visible-but-0-pts boxes dropped). We overwrite each GT box's num_pts
    to 1 iff visibility>=2, so the existing point-filter keeps EXACTLY the
    visibility>=2 set -> eval GT == training GT. num_pts is used only for
    filtering/reporting, never in the AP/TP/NDS computation, so this only changes the
    (intended) GT set. Stacks on top of patch_eval_subset (wraps the current load_gt)."""
    from projects.mmdet3d_plugin.datasets import nuscnes_eval as ne
    prev_load_gt = ne.load_gt

    def load_gt_vis(*a, **k):
        gt = prev_load_gt(*a, **k)
        for st in gt.boxes:
            for box in gt.boxes[st]:
                box.num_pts = 1 if box.visibility in min_vis else 0
        return gt

    ne.load_gt = load_gt_vis


def build_subset_infos(frames_per_scene):
    """Frozen N-frames-per-scene subset of the sedan val pkl (even stride).

    test-mode temporal is OFF (scene_token==token) so per-frame subsetting is
    unbiased. Returns (metadata, infos)."""
    d = B.load_pkl(B.SEDAN_VAL)
    by_scene = {}
    for info in d['infos']:
        by_scene.setdefault(B.info_key(info)[0], []).append(info)
    sub = []
    for scene in sorted(by_scene):
        infos = sorted(by_scene[scene], key=lambda i: int(B.info_key(i)[1]))
        if frames_per_scene >= len(infos):
            pick = infos
        else:
            idx = np.linspace(0, len(infos) - 1, frames_per_scene)
            pick = [infos[j] for j in sorted(set(idx.round().astype(int)))]
        sub.extend(pick)
    return d['metadata'], sub


def build_model_once(cfg, ckpt, batch=1):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, ckpt, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', None)
    # Allow batch>1: the deformable-attn CUDA kernel needs the effective batch
    # (<= bs*num_cams=bs*6 for spatial) to be <= im2col_step (default 64 -> caps
    # bs at 10). Use max(64, bs*6): bs<=10 keeps 64 (each sample one chunk, NDS
    # identical); bs>10 raises it so any sample-batch up to `batch` is one chunk.
    if batch > 1:
        for m in model.modules():
            if hasattr(m, 'im2col_step'):
                m.im2col_step = max(64, batch * 6)
    model = MMDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
    model.eval()
    return model


def run_condition(model, cfg, metadata, infos, ann_path, tmpdir, workers,
                  timing=None, batch=1):
    """Write the swapped subset, build dataset/loader, infer, return (NDS, mAP6).

    If `timing` is a dict it gets per-stage wall-clock (build/infer/eval) so we
    can see whether the full-scale bottleneck is I/O or backbone compute.
    """
    t0 = time.perf_counter()
    B.dump_pkl({'metadata': metadata, 'infos': infos}, ann_path)
    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg['ann_file'] = ann_path
    test_cfg['test_mode'] = True
    dataset = build_dataset(test_cfg)
    loader = build_dataloader(
        dataset, samples_per_gpu=batch, workers_per_gpu=workers,
        dist=False, shuffle=False)
    t1 = time.perf_counter()
    outputs = single_gpu_infer(model, loader)
    t2 = time.perf_counter()
    if timing is not None:
        timing['build'] = t1 - t0
        timing['infer'] = t2 - t1
        timing['n'] = len(infos)
    rank, _ = get_dist_info()
    if rank != 0:
        return None, None, None
    res = dataset.evaluate(outputs, metric='bbox',
                           jsonfile_prefix=osp.join(tmpdir, 'eval'))
    if timing is not None:
        timing['eval'] = time.perf_counter() - t2
    nds = res.get('pts_bbox_NuScenes/NDS')
    m6 = res.get('pts_bbox_NuScenes/mAP')
    if nds is None:                                   # fall back: scan keys
        for k, v in res.items():
            if k.endswith('NuScenes/NDS') and '10class' not in k:
                nds = v
            if k.endswith('NuScenes/mAP') and '10class' not in k:
                m6 = v
    if nds is None:
        raise RuntimeError(f'no NDS in evaluate() result; keys={list(res)}')
    return float(nds), float(m6), dict(res)          # full detail dict per cell


def infer_fast(model, cfg, metadata, infos, ann_path, pool, batch=1, timing=None):
    """NDS-EXACT fast inference: ProcessPool JPEG decode (overlaps GPU forward) +
    manual img_metas (no dataloader/DataContainer/DDP). Returns (dataset, outputs).
    batch=1 is bit-identical to the standard path (batch>1 perturbs NDS on the
    perturbed cells, so keep 1 for exactness). The EVAL is run separately so it can
    be pipelined (CPU) against the next cell's GPU inference."""
    import fast_decode as FD
    from collections import deque
    t0 = time.perf_counter()
    B.dump_pkl({'metadata': metadata, 'infos': infos}, ann_path)
    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg['ann_file'] = ann_path
    test_cfg['test_mode'] = True
    dataset = build_dataset(test_cfg)                       # for get_data_info + evaluate
    det = model.module
    t1 = time.perf_counter()
    outputs = []

    if batch == 1:
        # SHARED-MEMORY SLIDING WINDOW (NDS-exact): workers decode directly into
        # `pool.n_slots` shared buffers and return only a slot index, so the main
        # process never unpickles the 27MB image -> the GIL no longer serialises the
        # receive against the GPU forward. A window of n_slots tasks is kept in flight
        # so decode(N+1..N+K) runs in the pool while the forward of N runs on the GPU
        # -> infer becomes forward-bound. The forward sees the same per-sample bytes in
        # the same order, so NDS is bit-identical to the serial path.
        infos_n = len(dataset.data_infos)
        free = deque(range(pool.n_slots))
        inflight = deque()                              # (idx, slot, future), in order
        it = iter(range(infos_n))

        def submit_next():
            try:
                i = next(it)
            except StopIteration:
                return
            slot = free.popleft()
            inflight.append((i, slot, pool.submit(dataset.data_infos[i], slot)))

        for _ in range(pool.n_slots):                   # prime the window
            submit_next()
        while inflight:
            i, slot, fut = inflight.popleft()
            fut.result()                                # decode into views[slot] done
            # H2D copy out of the shm slot (sync -> slot reusable once it returns)
            img = torch.from_numpy(pool.views[slot]).float().cuda()[None]
            meta = [FD.manual_img_metas(dataset, i)]
            with torch.no_grad():
                out = det.simple_test(meta, img, prev_bev=None, rescale=True)[1]
            outputs.extend(out)
            free.append(slot)                           # free AFTER H2D completed
            submit_next()                               # slide the window forward
    else:
        # batch>1: NOT NDS-exact (BEVFormer spatial cross-attn rebatch depends on
        # batch size); kept only for profiling. Uses the plain (pickling) decode.
        buf_img, buf_meta = [], []

        def flush():
            if not buf_img:
                return
            img = torch.from_numpy(np.stack(buf_img)).float().cuda(non_blocking=True)
            with torch.no_grad():
                out = det.simple_test(list(buf_meta), img, prev_bev=None, rescale=True)[1]
            outputs.extend(out)
            buf_img.clear()
            buf_meta.clear()

        for i, img_np in enumerate(pool.pool.map(FD.decode_sample, dataset.data_infos,
                                                 chunksize=2)):
            buf_img.append(img_np)
            buf_meta.append(FD.manual_img_metas(dataset, i))
            if len(buf_img) == batch:
                flush()
        flush()

    if timing is not None:
        timing['build'] = t1 - t0
        timing['infer'] = time.perf_counter() - t1
        timing['n'] = len(dataset.data_infos)
    return dataset, outputs


def eval_outputs(dataset, outputs, eval_prefix, timing=None):
    """The CPU-bound NuScenes 6-class eval (pipelined on a thread against infer)."""
    t = time.perf_counter()
    res = dataset.evaluate(outputs, metric='bbox', jsonfile_prefix=eval_prefix)
    if timing is not None:
        timing['eval'] = time.perf_counter() - t
    nds = res.get('pts_bbox_NuScenes/NDS')
    m6 = res.get('pts_bbox_NuScenes/mAP')
    if nds is None:
        for k, v in res.items():
            if k.endswith('NuScenes/NDS') and '10class' not in k:
                nds = v
            if k.endswith('NuScenes/mAP') and '10class' not in k:
                m6 = v
    return float(nds), float(m6), dict(res)


def run_condition_fast(model, cfg, metadata, infos, ann_path, tmpdir, pool,
                       timing=None, batch=1):
    """Convenience: fast infer + eval (synchronous). Used for the Normal cell and
    measure/profile. The grid loop pipelines infer_fast + eval_outputs instead."""
    dataset, outputs = infer_fast(model, cfg, metadata, infos, ann_path, pool,
                                  batch=batch, timing=timing)
    return eval_outputs(dataset, outputs, osp.join(tmpdir, 'eval'), timing=timing)


# --------------------------------------------------------------------------- #
# Persistent eval PROCESS. The NuScenes eval is CPU/Python-heavy and holds the
# GIL; running it on a thread alongside the GPU infer makes the two GIL-thrash
# (measured: 600s/cell, worse than serial). A separate process has its own GIL,
# so eval(N) truly overlaps infer(N+1). The GT + data_infos token order are
# IDENTICAL across all VP cells (perturbation changes pixels/extrinsics, not
# tokens), so one dataset built from the baseline evaluates every cell's outputs
# correctly -> NDS bit-identical to per-cell eval.
# --------------------------------------------------------------------------- #
_EVAL_DS = None


def _eval_worker_init(cfg_path, baseline_ann, subset_tokens):
    global _EVAL_DS
    import copy as _copy
    from mmcv import Config as _Config
    from mmdet3d.datasets import build_dataset as _bd
    importlib.import_module('projects.mmdet3d_plugin')
    patch_nuscenes_cache()                       # idempotent under fork; needed on spawn
    patch_eval_subset(subset_tokens)
    patch_eval_visibility()                      # GT filter = visibility>=2 (== training)
    _cfg = _Config.fromfile(cfg_path)
    tc = _copy.deepcopy(_cfg.data.test)
    tc['ann_file'] = baseline_ann
    tc['test_mode'] = True
    _EVAL_DS = _bd(tc)


def _eval_ping():
    return _EVAL_DS is not None                  # forces worker fork+init early


def _eval_proc(out_path, eval_prefix):
    """Eval one cell's outputs against the persistent baseline GT dataset.

    outputs are passed via a pickle FILE (path), not the pool pipe: sending torch
    tensors through multiprocessing triggers torch's per-tensor shared-memory FD
    reducer (~11k FDs/cell -> 'too many open files'); a plain pickle file serialises
    them as bytes, so no FDs leak. Bytes are identical -> NDS unchanged."""
    outputs = B.load_pkl(out_path)
    try:
        os.remove(out_path)
    except OSError:
        pass
    res = _EVAL_DS.evaluate(outputs, metric='bbox', jsonfile_prefix=eval_prefix)
    nds = res.get('pts_bbox_NuScenes/NDS')
    m6 = res.get('pts_bbox_NuScenes/mAP')
    if nds is None:
        for k, v in res.items():
            if k.endswith('NuScenes/NDS') and '10class' not in k:
                nds = v
            if k.endswith('NuScenes/mAP') and '10class' not in k:
                m6 = v
    return float(nds), float(m6), dict(res)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='projects/configs/bevformer/bevformer_DFA3D_carla.py')
    ap.add_argument('--ckpt', default='work_dirs/bevformer_DFA3D_carla/epoch_24.pth')
    ap.add_argument('--tag', default='dfa3d_sedan')
    ap.add_argument('--frames-per-scene', type=int, default=4)
    ap.add_argument('--conditions', nargs='+', default=['ER', 'VR', 'CR'])
    ap.add_argument('--axes', nargs='+', default=B.VP_AXES)
    ap.add_argument('--mags', nargs='+', type=int, default=B.VP_MAGNITUDES)
    ap.add_argument('--protocol', default='both', choices=['both', 'allcam', 'percam'],
                    help='both = per-cam(6)+all-cam(1); allcam = all-cam only')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--fast', action='store_true',
                    help='NDS-exact fast path: ProcessPool decode (overlaps GPU) + '
                         'manual img_metas, no dataloader/DDP (~2x infer). --workers '
                         'sets the decode-pool size.')
    ap.add_argument('--batch', type=int, default=1,
                    help='inference samples_per_gpu (raises im2col_step to bs*6)')
    ap.add_argument('--shard', default='0/1',
                    help='cell shard "i/n" for independent multi-GPU jobs '
                         '(each job runs cells[i::n]; Normal always run)')
    ap.add_argument('--outdir', default=osp.join(HERE, 'out'))
    ap.add_argument('--merge', action='store_true',
                    help='combine shard*.pkl into final outputs and exit (no GPU)')
    ap.add_argument('--batch-sweep', action='store_true',
                    help='time the Normal cell across batch sizes and exit')
    ap.add_argument('--worker-sweep', action='store_true',
                    help='warm cache, then time Normal across worker counts and exit')
    ap.add_argument('--profile-split', action='store_true',
                    help='time backbone(extract_img_feat) vs BEV+head and exit '
                         '(decides if backbone caching is worth it)')
    ap.add_argument('--measure', action='store_true',
                    help='time build/infer/eval for a few cells and exit '
                         '(to settle the full-scale bottleneck)')
    ap.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return ap.parse_args()


def set_deterministic():
    """Make fp32 inference BIT-EXACT run-to-run (kills the NDS wiggle).

    The only run-to-run variation in BEVFormer *inference* comes from
    cudnn.benchmark algo selection + TF32 (on by default on Blackwell). The
    deformable-attn atomicAdd is backward-only (never runs under no_grad), so
    disabling those + a fixed seed gives byte-identical NDS. We do NOT call
    torch.use_deterministic_algorithms(True): the ms_deform_attn CUDA op is not
    registered deterministic and would raise.
    """
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'   # before first cuda matmul
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from mmdet.apis import set_random_seed
    set_random_seed(0, deterministic=True)


def _cell_path(cell_dir, cond, axis, mag, proto):
    """Per-cell result-cache path (resume: skip already-done cells on restart)."""
    name = f"{cond}_{axis}_{mag:+d}_{proto}".replace('/', '-')
    return osp.join(cell_dir, name + '.pkl')


import pickle as _pk  # noqa: E402  (used by the crash-safe cache helpers below)


def _save_cell(path, obj):
    """Crash-safe per-cell cache WRITE: dump+flush+fsync to a .tmp, then atomic
    os.replace() into place. A SIGKILL mid-dump leaves only the .tmp (never a
    truncated `path`), so a restart never trips over a half-written .pkl."""
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        _pk.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)            # atomic on POSIX


def _load_cell(path):
    """Crash-safe per-cell cache READ: return the unpickled object, or None if the
    file is missing/corrupt/truncated. A corrupt cache is removed so the caller
    falls through and recomputes the cell instead of aborting the whole restart."""
    try:
        with open(path, 'rb') as f:
            return _pk.load(f)
    except Exception:
        try:
            os.remove(path)         # drop the corrupt/truncated cache so it recomputes
        except OSError:
            pass
        return None


def main():
    args = parse_args()
    if args.merge:                               # CPU-only finalize, no dist/model
        merge_shards(osp.join(args.outdir, f'vp_{args.tag}'), args)
        return
    cfg = Config.fromfile(args.config)
    if cfg.get('plugin', False) and hasattr(cfg, 'plugin_dir'):
        _mp = '.'.join(osp.dirname(cfg.plugin_dir).split('/'))
        importlib.import_module(_mp)

    init_dist('pytorch', **cfg.get('dist_params', dict(backend='nccl')))
    set_deterministic()                      # fp32, reproducible NDS (no fp16)
    patch_nuscenes_cache()

    outdir = osp.join(args.outdir, f'vp_{args.tag}')
    # PER-SHARD tmpdir: concurrent shards must NOT share the swapped pkl / eval
    # json / collect dir, or they clobber each other (JSONDecodeError crash).
    si0 = int(args.shard.split('/')[0])
    tmpdir = osp.join(outdir, f'tmp_shard{si0}')
    os.makedirs(tmpdir, exist_ok=True)
    # PER-CELL result cache (resume): each finished cell's row is dumped here the
    # moment it completes; on restart, done cells load from here and are SKIPPED.
    # Persistent (in outdir), unlike tmp_shard/ scratch. Per-shard to avoid clashes.
    cell_dir = osp.join(outdir, f'cells_shard{si0}')
    os.makedirs(cell_dir, exist_ok=True)
    ann_path = osp.join(tmpdir, 'cond_infos_val.pkl')

    metadata, base = build_subset_infos(args.frames_per_scene)
    patch_eval_subset({info['token'] for info in base})
    patch_eval_visibility()                  # GT filter = visibility>=2 (== training)
    rank, _ = get_dist_info()
    if rank == 0:
        print(f'[VP] subset = {len(base)} samples '
              f'({args.frames_per_scene}/scene x 48 scenes)', flush=True)

    # Create the decode pool BEFORE CUDA init (build_model_once) so the workers
    # fork from a clean, CUDA-free parent (forking a CUDA-initialised process is
    # unsafe). Persistent -> spawned once for the whole run.
    pool = None
    eval_proc_pool = None
    if args.fast:
        import fast_decode  # noqa: F401  (forked workers inherit this import)
        # ShmPool: workers decode straight into shared-memory slots (no 27MB pickle)
        # -> forward-bound infer. n_slots = workers + 8 keeps the window full.
        pool = fast_decode.ShmPool(workers=args.workers)
        # Persistent eval process: GT built once from the baseline; the CPU eval runs
        # here so it does not GIL-thrash with the GPU infer in the main process.
        from concurrent.futures import ProcessPoolExecutor
        baseline_ann = osp.join(tmpdir, 'baseline_eval.pkl')
        B.dump_pkl({'metadata': metadata, 'infos': base}, baseline_ann)
        eval_proc_pool = ProcessPoolExecutor(
            max_workers=1, initializer=_eval_worker_init,
            initargs=(args.config, baseline_ann, {info['token'] for info in base}))
        eval_proc_pool.submit(_eval_ping).result()   # fork+build NOW (before CUDA)
        eval_json_dir = f'/tmp/vpeval_{args.tag}_shard{si0}'  # tmpfs; per-tag (no cross-run sharing)
        os.makedirs(eval_json_dir, exist_ok=True)

    model = build_model_once(cfg, args.ckpt, batch=args.batch)

    if args.profile_split:
        for m in model.modules():            # batch=16 -> effective 96 needs im2col 96
            if hasattr(m, 'im2col_step'):
                m.im2col_step = 16 * 6
        t = {'backbone': 0.0, 'head': 0.0, 'get_bboxes': 0.0}
        mod = model.module
        o_ex, o_hd = mod.extract_img_feat, mod.pts_bbox_head.forward
        o_gb = mod.pts_bbox_head.get_bboxes

        def timed(fn, key):
            def w(*a, **k):
                torch.cuda.synchronize(); s = time.perf_counter()
                r = fn(*a, **k)
                torch.cuda.synchronize(); t[key] += time.perf_counter() - s
                return r
            return w
        mod.extract_img_feat = timed(o_ex, 'backbone')
        mod.pts_bbox_head.forward = timed(o_hd, 'head')
        mod.pts_bbox_head.get_bboxes = timed(o_gb, 'get_bboxes')
        tm = {}
        run_condition(model, cfg, metadata,
                      B.make_vp_infos(base, 'Normal', 'yaw', 0, 'all'),
                      ann_path, tmpdir, args.workers, timing=tm, batch=16)
        if rank == 0:
            n = tm.get('n', len(base))
            tot = tm.get('infer', 0)
            resid = tot - t['backbone'] - t['head'] - t['get_bboxes']
            pc = lambda v: f'{v:.1f}s ({v/tot*100:.0f}%, {v/n*1000:.0f}ms/s)'
            print(f'[PROFILE] n={n} infer_total={tot:.1f}s', flush=True)
            print(f'[PROFILE]   backbone   = {pc(t["backbone"])}', flush=True)
            print(f'[PROFILE]   BEV+head   = {pc(t["head"])}', flush=True)
            print(f'[PROFILE]   get_bboxes = {pc(t["get_bboxes"])}', flush=True)
            print(f'[PROFILE]   residual(dataloader I/O + transfer) = {pc(resid)}',
                  flush=True)
        return

    if args.worker_sweep:
        import concurrent.futures as cf
        norm = B.make_vp_infos(base, 'Normal', 'yaw', 0, 'all')
        paths = [c['data_path'] for info in base for c in info['cams'].values()]
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(64) as ex:
            list(ex.map(lambda p: open(p, 'rb').read() and None, paths))
        if rank == 0:
            print(f'[WSWEEP] warmed {len(paths)} imgs in '
                  f'{time.perf_counter()-t0:.1f}s', flush=True)
        for w in [4, 8, 16, 32]:
            tm = {}
            run_condition(model, cfg, metadata, norm, ann_path, tmpdir, w,
                          timing=tm, batch=args.batch)
            if rank == 0:
                n = tm.get('n', len(base))
                print(f'[WSWEEP] workers={w:3d}  infer={tm.get("infer",0):6.1f}s '
                      f'({tm.get("infer",0)/max(n,1)*1000:5.0f} ms/sample)  '
                      f'eval={tm.get("eval",0):5.1f}s', flush=True)
        return

    if args.batch_sweep:
        norm = B.make_vp_infos(base, 'Normal', 'yaw', 0, 'all')
        for bsz in [1, 4, 16, 32, 64, 128]:
            for m in model.modules():
                if hasattr(m, 'im2col_step'):
                    m.im2col_step = max(64, bsz * 6)
            tm = {}
            run_condition(model, cfg, metadata, norm, ann_path, tmpdir,
                          args.workers, timing=tm, batch=bsz)
            if rank == 0:
                n = tm.get('n', len(base))
                print(f'[SWEEP] batch={bsz:4d}  infer={tm.get("infer",0):6.1f}s '
                      f'({tm.get("infer",0)/max(n,1)*1000:5.0f} ms/sample)  '
                      f'eval={tm.get("eval",0):5.1f}s', flush=True)
        return

    if args.measure:
        # time a baseline-image cell (VR all-cam: backbone+I/O heavy) and an
        # extrinsic-only cell (ER all-cam: same images, only lidar2img differs)
        # so we can see how much is I/O/backbone vs head.
        probes = [('Normal', 'yaw', 0, 'all'),
                  ('VR', 'yaw', 20, 'all'),
                  ('ER', 'yaw', 20, 'all'),
                  ('VR', 'yaw', 20, 'CAM_FRONT')]
        for cond, axis, mag, proto in probes:
            tm = {}
            run_condition(model, cfg, metadata,
                          B.make_vp_infos(base, cond, axis, mag, proto),
                          ann_path, tmpdir, args.workers, timing=tm,
                          batch=args.batch)
            if rank == 0:
                n = tm.get('n', len(base))
                print(f'[MEASURE] {cond:6s} {axis}{mag:+d} {proto:12s} '
                      f'n={n:5d}  build={tm.get("build",0):6.1f}s  '
                      f'infer={tm.get("infer",0):7.1f}s '
                      f'({tm.get("infer",0)/max(n,1)*1000:.0f}ms/sample)  '
                      f'eval={tm.get("eval",0):5.1f}s', flush=True)
        if rank == 0:
            print('[MEASURE] done. extrapolate full-3792 cost from infer ms/sample.',
                  flush=True)
        return

    cams = B.CAM_NAMES
    if args.protocol == 'allcam':
        protocols = ['all']
    elif args.protocol == 'percam':
        protocols = list(cams)
    else:
        protocols = ['all'] + list(cams)

    def mkrow(cond, axis, mag, proto, nds, m6, metrics):
        rrs = nds / nds_norm if nds_norm else float('nan')
        return {'condition': cond, 'axis': axis, 'signed_mag': mag,
                'protocol': proto, 'nds': nds, 'map6': m6, 'rrs': rrs,
                'comp': nds_components(metrics), 'metrics': metrics}

    # fast (ProcessPool decode + manual img_metas) vs the standard dataloader path
    def rc(infos, timing=None):
        if args.fast:
            return run_condition_fast(model, cfg, metadata, infos, ann_path,
                                      tmpdir, pool, timing=timing, batch=args.batch)
        return run_condition(model, cfg, metadata, infos, ann_path, tmpdir,
                             args.workers, timing=timing, batch=args.batch)

    # ---- Normal (oracle) : every shard runs it (RRS denominator) -------------
    # resume: cache the oracle row so a restart does not recompute it.
    _normal_path = osp.join(cell_dir, 'normal.pkl')
    _tm = {}
    _d = _load_cell(_normal_path)
    if _d is not None:
        nds_norm, m6_norm, norm_metrics = _d['nds_norm'], _d['m6_norm'], _d['norm_metrics']
        if rank == 0:
            print(f'[VP] Normal (cached) NDS={nds_norm:.4f} mAP6={m6_norm:.4f}', flush=True)
    else:
        nds_norm, m6_norm, norm_metrics = rc(
            B.make_vp_infos(base, 'Normal', 'yaw', 0, 'all'), timing=_tm)
        if rank == 0:
            _save_cell(_normal_path, {'nds_norm': nds_norm, 'm6_norm': m6_norm,
                                      'norm_metrics': norm_metrics})
            n = _tm.get('n', len(base))
            print(f'[VP] Normal NDS={nds_norm:.4f} mAP6={m6_norm:.4f} | '
                  f'infer={_tm.get("infer",0):.1f}s ({_tm.get("infer",0)/max(n,1)*1000:.0f}'
                  f'ms/s) eval={_tm.get("eval",0):.1f}s build={_tm.get("build",0):.1f}s',
                  flush=True)

    # full cell list (deterministic order), then this shard's slice
    all_cells = [(c, a, mg, p) for c in args.conditions for a in args.axes
                 for mg in args.mags for p in protocols]
    si, sn = (int(x) for x in args.shard.split('/'))
    my_cells = all_cells[si::sn]

    rows = []
    if rank == 0 and si == 0:        # only shard 0 owns the Normal row
        rows.append(mkrow('Normal', '-', 0, 'all', nds_norm, m6_norm, norm_metrics))

    if args.fast:
        # PIPELINE: infer cell N on the GPU (main process) while the persistent eval
        # PROCESS evaluates cell N-1 -> per-cell wall ~= max(infer, eval). Eval runs
        # in a separate process (own GIL), so it does NOT stall the infer. Depth
        # bounded to 2 so at most ~2 outputs lists are in flight.
        from collections import deque
        pending = deque()

        def collect_one():
            d, cond, axis, mag, proto, fut = pending.popleft()
            nds, m6, metrics = fut.result()
            row = mkrow(cond, axis, mag, proto, nds, m6, metrics)
            _save_cell(_cell_path(cell_dir, cond, axis, mag, proto), row)  # resume: per-cell save (crash-safe)
            rows.append(row)
            print(f'[VP shard{si} {d}/{len(my_cells)}] {cond} {axis}{mag:+d} '
                  f'{proto:14s} NDS={nds:.4f} RRS={row["rrs"]:.4f}', flush=True)

        for done, (cond, axis, mag, proto) in enumerate(my_cells, 1):
            cp = _cell_path(cell_dir, cond, axis, mag, proto)
            _cached = _load_cell(cp)                  # resume: load-then-check (crash-safe)
            if _cached is not None:                   # skip already-done cell
                row = _cached
                rows.append(row)
                print(f'[VP shard{si} {done}/{len(my_cells)}] {cond} {axis}{mag:+d} '
                      f'{proto:14s} (cached) NDS={row["nds"]:.4f} RRS={row["rrs"]:.4f}', flush=True)
                continue
            infos = B.make_vp_infos(base, cond, axis, mag, proto)
            _, outputs = infer_fast(model, cfg, metadata, infos, ann_path,
                                    pool, batch=args.batch)
            # hand outputs to the eval process via a pickle FILE (avoids torch's
            # per-tensor shared-memory FD reducer over the pool pipe).
            out_path = osp.join(tmpdir, f'out_{done % 4}.pkl')
            B.dump_pkl(outputs, out_path)
            # fixed eval-json prefix (single serial eval worker overwrites it each
            # cell -> no per-cell JSON accumulation filling the disk).
            fut = eval_proc_pool.submit(_eval_proc, out_path,
                                        osp.join(eval_json_dir, 'e'))
            pending.append((done, cond, axis, mag, proto, fut))
            if len(pending) >= 2:           # bound depth -> overlap eval(N) w/ infer(N+1)
                collect_one()
        while pending:
            collect_one()
        eval_proc_pool.shutdown(wait=True)
    else:
        for done, (cond, axis, mag, proto) in enumerate(my_cells, 1):
            cp = _cell_path(cell_dir, cond, axis, mag, proto)
            # resume: load-then-check (crash-safe). ALL ranks load so a corrupt
            # cache is treated as 'not done' uniformly -> all ranks skip OR all
            # recompute together (collective-safe; a one-rank divergence would hang).
            _cached = _load_cell(cp)
            if _cached is not None:                  # skip done cell
                if rank == 0:
                    row = _cached
                    rows.append(row)
                    print(f'[VP shard{si} {done}/{len(my_cells)}] {cond} {axis}{mag:+d} '
                          f'{proto:14s} (cached) NDS={row["nds"]:.4f} RRS={row["rrs"]:.4f}', flush=True)
                continue
            infos = B.make_vp_infos(base, cond, axis, mag, proto)
            nds, m6, metrics = rc(infos)
            if rank == 0:
                row = mkrow(cond, axis, mag, proto, nds, m6, metrics)
                _save_cell(cp, row)                   # resume: per-cell save (crash-safe)
                rows.append(row)
                print(f'[VP shard{si} {done}/{len(my_cells)}] {cond} {axis}{mag:+d} '
                      f'{proto:14s} NDS={nds:.4f} RRS={row["rrs"]:.4f}', flush=True)

    if pool is not None and hasattr(pool, 'shutdown'):
        pool.shutdown()                              # free shared-memory slots

    if rank != 0:
        return
    if sn > 1:                                   # multi-GPU shard: dump + merge later
        with open(osp.join(outdir, f'shard{si}.pkl'), 'wb') as f:
            import pickle
            pickle.dump({'rows': rows, 'nds_norm': nds_norm, 'm6_norm': m6_norm}, f)
        print(f'[VP] shard {si}/{sn}: wrote {len(rows)} rows -> shard{si}.pkl '
              f'(run --merge to finalize)', flush=True)
    else:
        write_outputs(outdir, args, nds_norm, m6_norm, rows, cams)


def merge_shards(outdir, args):
    """Combine shard*.pkl into the final outputs (no GPU/dist needed)."""
    import glob
    import pickle
    cams = B.CAM_NAMES
    if args.protocol == 'allcam':
        protocols = ['all']
    elif args.protocol == 'percam':
        protocols = list(cams)
    else:
        protocols = ['all'] + list(cams)
    rows, nds_norm, m6_norm = [], None, None
    for p in sorted(glob.glob(osp.join(outdir, 'shard*.pkl'))):
        d = pickle.load(open(p, 'rb'))
        rows.extend(d['rows'])
        nds_norm = d['nds_norm'] if nds_norm is None else nds_norm
        m6_norm = d['m6_norm'] if m6_norm is None else m6_norm
    # canonical cell order (Normal first, then grid order) so aggregates pair up
    order = {('Normal', '-', 0, 'all'): -1}
    for i, c in enumerate([(c, a, mg, pr) for c in args.conditions for a in args.axes
                           for mg in args.mags for pr in protocols]):
        order[c] = i
    rows.sort(key=lambda r: order.get(
        (r['condition'], r['axis'], r['signed_mag'], r['protocol']), 1 << 30))
    print(f'[VP merge] {len(rows)} rows from shards', flush=True)
    write_outputs(outdir, args, nds_norm, m6_norm, rows, cams)


def write_outputs(outdir, args, nds_norm, m6_norm, rows, cams):
    fmt = lambda v: '' if v is None else f'{v:.4f}'
    # per-cell CSV with every NDS component
    csv_path = osp.join(outdir, 'eval_vp_per_config.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'axis', 'signed_mag', 'protocol',
                    'nds', 'map6'] + COMP_COLS + ['rrs'])
        for r in rows:
            w.writerow([r['condition'], r['axis'], r['signed_mag'], r['protocol'],
                        fmt(r['nds']), fmt(r['map6'])]
                       + [fmt(r['comp'][c]) for c in COMP_COLS] + [fmt(r['rrs'])])

    # aggregate per condition: mRRS (per-cam), RRSALL (all-cam), mVRS
    agg = {}
    for cond in args.conditions:
        per = [r['rrs'] for r in rows if r['condition'] == cond and r['protocol'] != 'all']
        allc = [r['rrs'] for r in rows if r['condition'] == cond and r['protocol'] == 'all']
        mrrs = float(np.mean(per)) if per else float('nan')
        rrsall = float(np.mean(allc)) if allc else float('nan')
        agg[cond] = {'mRRS_percam': mrrs, 'RRSALL_allcam': rrsall,
                     'mVRS': 0.5 * (mrrs + rrsall)}

    js = {'config': args.config, 'ckpt': args.ckpt, 'tag': args.tag,
          'frames_per_scene': args.frames_per_scene,
          'normal_nds': nds_norm, 'normal_map6': m6_norm,
          'aggregate': agg, 'rows': rows}   # rows carry full per-cell 'metrics'
    with open(osp.join(outdir, 'eval_vp.json'), 'w') as f:
        json.dump(js, f, indent=2)

    lines = [f'VP viewpoint-robustness detection eval (NDS)   tag={args.tag}',
             f'  config={args.config}  ckpt={args.ckpt}',
             f'  subset={args.frames_per_scene}/scene   '
             f'Normal NDS={nds_norm:.4f}  mAP6={m6_norm:.4f}', '',
             '  condition   mRRS(per-cam)   RRSALL(all-cam)        mVRS']
    for cond in args.conditions:
        a = agg[cond]
        star = '  <- primary' if cond == 'VR' else ''
        lines.append(f'  {cond:<10}{a["mRRS_percam"]:>13.4f}'
                     f'{a["RRSALL_allcam"]:>16.4f}{a["mVRS"]:>12.4f}{star}')
    txt = '\n'.join(lines)
    with open(osp.join(outdir, 'eval_vp_summary.txt'), 'w') as f:
        f.write(txt + '\n')
    print('\n' + txt)
    print(f'\nwrote {csv_path}\n      {osp.join(outdir, "eval_vp.json")}\n'
          f'      {osp.join(outdir, "eval_vp_summary.txt")}')


if __name__ == '__main__':
    main()
