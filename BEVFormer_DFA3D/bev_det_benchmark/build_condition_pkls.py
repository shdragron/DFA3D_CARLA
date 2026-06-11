"""Build condition-specific val .pkl files for the 3D-detection robustness
benchmark (the NDS analogue of bev_seg_benchmark's CVT IoU eval).

The detection eval is 100% pkl-driven: tools/test.py builds the val dataset from
``data.test.ann_file`` and CarlaNuScenesDataset rebuilds ``lidar2img`` purely
from each cam's ``sensor2lidar_rotation``/``sensor2lidar_translation`` /
``cam_intrinsic`` (image bytes only feed the backbone, never the GT). So we can
reproduce the seg eval's image_swap / extrinsic_swap / viewpoint-variant /
cross-platform conditions by *cloning the sedan val pkl and surgically swapping
cam fields*, then running each model's stock test entry unchanged.

GT (gt_boxes/gt_names/valid_flag/token/scene_token) and metadata['version'] are
NEVER touched, so the 6-class NDS pipeline is byte-identical across conditions
and ``metadata['version']='v1.0-carla_sedan_eval'`` keeps driving the right
NuScenes DB load + the carla scene-name patch in _evaluate_single.

Two benchmark families:

  CTS  cross-platform transfer  (sedan-trained model, suv/bus eval data)
       NORMAL = (sedan img, sedan ext)   <- sedan-inputs reference, == stock sedan pkl
       EXT    = (sedan img, target ext)
       IMG    = (target img, sedan ext)  <- primary
       CAL    = (target img, target ext)
       target/sedan cam fields are joined by parsed (scene, frame), NOT by token
       (sedan/suv/bus tokens are independent UUIDs -> 0% overlap; scene+frame
       overlap is 3792/3792 and the GT boxes are bit-identical).

  VP   viewpoint robustness  (carla_VR variant images + extrinsics)
       Normal(img=F,ext=F) ER(F,T) VR(T,F)<-primary CR(T,T) x axis x signed-mag
       image_swap -> point cam data_path at the carla_VR variant JPG, keep the
                     baseline sensor2lidar + intrinsic
       extrinsic_swap -> keep the sedan image, overwrite sensor2lidar with the
                         variant's (coupled) transform from viewpoint_metadata.json
       (carla_VR has images + metadata only, NO boxes -> GT stays the sedan pkl's)

Usage (library):
    from build_condition_pkls import make_cts_pkl, make_vp_pkl
Usage (CLI smoke test):
    python build_condition_pkls.py cts --target suv --condition IMG --out out/x.pkl
"""
import argparse
import copy
import os
import pickle
import re

import numpy as np

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
BEVF_ROOT = '/home/hanyan_arch/viewpoint/BEVFormer/3D-deformable-attention/BEVFormer_DFA3D'
DATA_ROOT = os.path.join(BEVF_ROOT, 'data', 'nuscenes')      # symlink -> carla_geobev
VR_ROOT = '/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_VR'
VR_META = os.path.join(VR_ROOT, 'viewpoint_metadata.json')

SEDAN_VAL = os.path.join(DATA_ROOT, 'sedan_infos_val.pkl')

# data_path looks like
#   data/nuscenes/sweeps/RGB-CAM_FRONT/SimBEV-scene-0241-frame-0000-RGB-subcompact__CAM_FRONT.jpg
SCENE_FRAME_RE = re.compile(r'scene-(\d+)-frame-(\d+)')

CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


# --------------------------------------------------------------------------- #
# io helpers
# --------------------------------------------------------------------------- #
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pkl(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def scene_frame(data_path):
    """('0241', '0000') from a cam data_path. Raises if the pattern is absent."""
    m = SCENE_FRAME_RE.search(data_path)
    if not m:
        raise ValueError(f'no scene-/frame- token in: {data_path}')
    return m.group(1), m.group(2)


def info_key(info):
    """(scene, frame) identifying a sample, parsed from CAM_FRONT's path.

    All 6 cams of one sample share the same scene/frame, and this key is stable
    across platforms (sedan/suv/bus) whereas info['token'] is not.
    """
    return scene_frame(info['cams']['CAM_FRONT']['data_path'])


def build_index(infos):
    """{(scene, frame): info} for cross-platform / cross-pkl joins."""
    idx = {}
    for info in infos:
        idx[info_key(info)] = info
    return idx


# --------------------------------------------------------------------------- #
# CTS  -- cross-platform transfer
# --------------------------------------------------------------------------- #
CTS_CONDITIONS = {
    # condition: (image_source, extrinsic_source)   's'=sedan, 't'=target
    'NORMAL': ('s', 's'),   # sedan-inputs reference (== stock sedan pkl); a numerator, not the denominator
    'EXT':    ('s', 't'),
    'IMG':    ('t', 's'),   # primary
    'CAL':    ('t', 't'),
}


def _copy_extrinsic(dst_cam, src_cam):
    """Overwrite the whole cam->lidar (+cam->ego) extrinsic from src to dst.

    Per the verified design we OVERWRITE the coupled sensor2lidar pair directly
    (never recompute it from sensor2ego); cam_intrinsic is left untouched because
    it is identical across platforms and viewpoint variants.
    """
    dst_cam['sensor2lidar_rotation'] = np.array(src_cam['sensor2lidar_rotation'])
    dst_cam['sensor2lidar_translation'] = np.array(src_cam['sensor2lidar_translation'])
    # sensor2ego_* do not affect lidar2img but keep the info self-consistent.
    dst_cam['sensor2ego_rotation'] = copy.deepcopy(src_cam['sensor2ego_rotation'])
    dst_cam['sensor2ego_translation'] = copy.deepcopy(src_cam['sensor2ego_translation'])


def make_cts_pkl(condition, target, out_path,
                 sedan_pkl=SEDAN_VAL, target_pkl=None):
    """Write one CTS condition pkl, based on the TARGET eval set.

    Per the seg CTS definition (eval_cts_cvt.py), the eval runs on the TARGET's
    eval set, so the GT, valid_flag (= TARGET visibility, which differs from sedan
    by mount height) and metadata.version are the TARGET's. Each condition swaps
    the cam image and/or extrinsic toward the SEDAN source:
        NORMAL = sedan img + sedan ext  (sedan-inputs reference; a numerator condition)
        EXT    = sedan img + target ext
        IMG    = target img + sedan ext   (primary)
        CAL    = target img + target ext  (== target pkl as-is, full deploy)
    Each CTS_c = NDS_c / P_TARGET is a PER-TARGET ratio (target visibility), where the
    denominator P_TARGET is the target-NATIVE oracle (a model trained on the target,
    eval'd on its own set). That oracle is run by the driver (eval_cts_det.py), not
    here; make_cts_pkl only builds the numerator condition pkls.

    target in {'suv','bus'}; condition in CTS_CONDITIONS. Returns (n, matched, missing).
    """
    assert condition in CTS_CONDITIONS, condition
    img_src, ext_src = CTS_CONDITIONS[condition]               # 's'=sedan, 't'=target

    if target_pkl is None:
        target_pkl = os.path.join(DATA_ROOT, f'{target}_infos_val.pkl')
    target_data = load_pkl(target_pkl)                         # base: target GT/vis/version
    if img_src == 't' and ext_src == 't':                      # CAL == target pkl as-is
        dump_pkl(target_data, out_path)
        n = len(target_data['infos'])
        return n, n, 0

    sidx = build_index(load_pkl(sedan_pkl)['infos'])
    new = copy.deepcopy(target_data)                           # keeps target GT/valid_flag/version
    matched = missing = 0
    for info in new['infos']:
        sinfo = sidx.get(info_key(info))
        if sinfo is None:
            missing += 1
            continue
        matched += 1
        for cam_name, cam in info['cams'].items():
            scam = sinfo['cams'][cam_name]
            if img_src == 's':                                 # swap image toward sedan
                cam['data_path'] = scam['data_path']
            if ext_src == 's':                                 # swap extrinsic toward sedan
                _copy_extrinsic(cam, scam)
    dump_pkl(new, out_path)
    return len(new['infos']), matched, missing


# --------------------------------------------------------------------------- #
# VP  -- viewpoint robustness (carla_VR)
# --------------------------------------------------------------------------- #
# carla_VR ships 31 variants = baseline + 10 each for yaw / pitch / roll, signed
# +/-{4,8,12,16,20}.  Per (scene, cam, variant) it provides the perturbed image
# and the perturbed extrinsic (sensor2lidar_rotation as a [w,x,y,z] quaternion +
# translation). It has NO boxes -> GT always comes from the sedan val pkl.
VP_AXES = ['yaw', 'pitch', 'roll']
VP_MAGNITUDES = [-20, -16, -12, -8, -4, 4, 8, 12, 16, 20]   # signed, per axis
VP_CONDITIONS = {                       # condition: (swap_image, swap_extrinsic)
    'Normal': (False, False),           # baseline == stock sedan subset (oracle)
    'ER':     (False, True),            # extrinsic-only
    'VR':     (True,  False),           # image-only  (primary)
    'CR':     (True,  True),            # both
}

_VR_META = None


def load_vr_metadata():
    """Cached viewpoint_metadata.json (scenes[scene_us][cam][variant] -> extr)."""
    global _VR_META
    if _VR_META is None:
        import json
        with open(VR_META) as f:
            _VR_META = json.load(f)
    return _VR_META


def variant_key(axis, signed_mag):
    """('yaw', 20) -> 'yaw20pitch0roll0';  ('pitch', -8) -> 'yaw0pitch-8roll0'."""
    vals = {'yaw': 0, 'pitch': 0, 'roll': 0}
    vals[axis] = signed_mag
    return f"yaw{vals['yaw']}pitch{vals['pitch']}roll{vals['roll']}"


def vr_image_path(scene, frame, cam, variant):
    """Absolute carla_VR variant-image path (filenames use the dash scene form)."""
    # carla_geobev (val GT/images) is a 1/2-rate RELABEL of the carla_VR capture:
    # geobev frame N is the SAME world-moment as carla_VR frame 2N (verified --
    # geobev 0269/0150 == VR baseline 0300, exact 2.000x across clean-motion
    # scenes). Joining by the bare frame number loads a DIFFERENT scene moment, so
    # every image-swapped VR/CR condition was fed mismatched images -> use 2N.
    return (f"{VR_ROOT}/sweeps/RGB-{cam}_{variant}/"
            f"SimBEV-scene-{scene}-frame-{int(frame) * 2:04d}-RGB-{cam}_{variant}.jpg")


def variant_extrinsic(scene, cam, variant, meta=None):
    """(sensor2lidar_rotation 3x3, sensor2lidar_translation xyz) for a variant.

    metadata scene keys use the underscore form (scene_0241); rotation is stored
    as a [w,x,y,z] quaternion which we convert to the 3x3 the pkl expects.
    """
    from pyquaternion import Quaternion
    meta = meta or load_vr_metadata()
    rec = meta['scenes'][f'scene_{scene}'][cam][variant]
    rot = Quaternion(rec['sensor2lidar_rotation']).rotation_matrix
    trans = np.array(rec['sensor2lidar_translation'], dtype=np.float64)
    return rot, trans


def make_vp_infos(base_infos, condition, axis, signed_mag, protocol):
    """Return a deep-copied infos list with VP swaps applied (in-memory).

    protocol = 'all'  -> perturb all 6 cams together
    protocol = <CAM>  -> perturb only that camera (per-cam), others baseline.
    'Normal' returns an untouched deep copy (the oracle subset).
    """
    swap_img, swap_ext = VP_CONDITIONS[condition]
    variant = variant_key(axis, signed_mag)
    target_cams = CAM_NAMES if protocol == 'all' else [protocol]
    meta = load_vr_metadata()

    new = copy.deepcopy(base_infos)
    for info in new:
        scene, frame = info_key(info)
        for cam_name in target_cams:
            cam = info['cams'][cam_name]
            if swap_img:
                cam['data_path'] = vr_image_path(scene, frame, cam_name, variant)
            if swap_ext:
                rot, trans = variant_extrinsic(scene, cam_name, variant, meta)
                cam['sensor2lidar_rotation'] = rot
                cam['sensor2lidar_translation'] = trans
    return new


# --------------------------------------------------------------------------- #
# CLI (smoke test)
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cts')
    c.add_argument('--target', required=True, choices=['suv', 'bus'])
    c.add_argument('--condition', required=True, choices=list(CTS_CONDITIONS))
    c.add_argument('--out', required=True)
    args = ap.parse_args()

    if args.cmd == 'cts':
        n, matched, missing = make_cts_pkl(args.condition, args.target, args.out)
        print(f'[cts] {args.target} {args.condition}: n={n} matched={matched} '
              f'missing={missing} -> {args.out}')


if __name__ == '__main__':
    main()
