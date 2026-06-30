"""Scan clean sparse frames for the LARGEST DFA3D CTS recovery (bus CAL-IMG
recall@2m). Picks the most dramatic IMG-collapse -> CAL-recovery sample."""
import os, sys, copy, glob, json, pickle
import numpy as np, torch
from PIL import Image
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
import importlib; importlib.import_module('projects.mmdet3d_plugin')
BENCH = '/home/hanyan_arch/viewpoint/BEVFormer/bev_det_benchmark'; sys.path.insert(0, BENCH)
import build_condition_pkls as B
import qual_conditions as Q
CFG = 'projects/configs/bevformer/bevformer_DFA3D_carla.py'
CKPT = 'work_dirs/bevformer_DFA3D_carla/epoch_24.pth'
DATA = '/home/hanyan_arch/viewpoint/BEVFormer/data/nuscenes'
GA = '/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_geobev_labels/gaussianlss/sedan_eval'
IMG = '/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_geobev'
PC = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]; THR = 0.3

sd = {B.info_key(i): i for i in pickle.load(open(f'{DATA}/sedan_infos_val.pkl', 'rb'))['infos']}
uv = {B.info_key(i): i for i in pickle.load(open(f'{DATA}/suv_infos_val.pkl', 'rb'))['infos']}
bs = {B.info_key(i): i for i in pickle.load(open(f'{DATA}/bus_infos_val.pkl', 'rb'))['infos']}
gt = {}
for jp in glob.glob(os.path.join(GA, 'scene_*.json')):
    s = os.path.basename(jp)[:-5].replace('scene_', '')
    for fi, fr in enumerate(json.load(open(jp))):
        gt[(s, f'{2*fi:04d}')] = (s, fi, fr)
# clean candidates: vis2 4-8, clear (brightness/contrast), in all platforms
cand = []
for k, si in sd.items():
    if k not in uv or k not in bs or k not in gt:
        continue
    n = int(np.array(si['valid_flag']).sum())
    if not (4 <= n <= 8):
        continue
    s, fi, fr = gt[k]
    im = np.asarray(Image.open(os.path.join(IMG, fr['images'][1])).convert('L').resize((160, 90)))
    if im.mean() > 95 and im.std() > 38:
        cand.append(k)
# cap, keep diverse + some consecutive
cand = sorted(cand)[:40]
print(f'{len(cand)} clean candidate frames', flush=True)

cfg = Config.fromfile(CFG)
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, CKPT, map_location='cpu'); model.cuda().eval()

def infer(info, ver):
    p = '/tmp/dfa3d_scan.pkl'; pickle.dump({'infos': [copy.deepcopy(info)], 'metadata': {'version': ver}}, open(p, 'wb'))
    c = copy.deepcopy(cfg); c.data.test.ann_file = p; c.data.test.test_mode = True
    ds = build_dataset(c.data.test); data = scatter(collate([ds[0]], samples_per_gpu=1), [0])[0]
    with torch.no_grad():
        r = model(return_loss=False, rescale=True, **data)
    pb = r[0]['pts_bbox']; return pb['boxes_3d'], pb['scores_3d'].numpy()

def recall(gtb, pb, ps):
    if len(gtb) == 0:
        return 0.0, 0
    c = pb.gravity_center.numpy() if len(pb) else np.zeros((0, 3))
    pm = (ps > THR) & np.array([PC[0] <= x[0] <= PC[3] and PC[1] <= x[1] <= PC[4] for x in c], bool) if len(pb) else np.zeros(0, bool)
    pc = c[pm][:, :2] if pm.any() else np.zeros((0, 2))
    g = gtb.gravity_center.numpy()[:, :2]
    if len(pc) == 0:
        return 0.0, 0
    d = np.linalg.norm(g[:, None] - pc[None], axis=2)
    return float((d.min(1) < 2.0).mean()), int(pm.sum())

res = []
for k in cand:
    tinfo = bs[k]; sinfo = sd[k]; tgt_gt = Q.boxes_from(tinfo['gt_boxes'], tinfo['valid_flag'])
    if len(tgt_gt) == 0:
        continue
    ri, ni = infer(Q.make_cts_info('IMG', tinfo, sinfo), 'v1.0-carla_bus_eval')
    rc, nc = infer(Q.make_cts_info('CAL', tinfo, sinfo), 'v1.0-carla_bus_eval')
    recI, _ = recall(tgt_gt, ri, ni); recC, _ = recall(tgt_gt, rc, nc)
    res.append((recC - recI, recI, recC, len(tgt_gt), gt[k][0], gt[k][1], k))
res.sort(reverse=True)
print('\nrecovery(busCAL-IMG)  IMG_recall  CAL_recall  GT  scene  fi  key')
for r in res[:15]:
    print(f'  {r[0]:+.2f}  {r[1]:.2f}  {r[2]:.2f}  {r[3]}  scene_{r[4]} fi{r[5]} {r[6]}')
