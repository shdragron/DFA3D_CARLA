"""High-res CAM_BACK-only DFA3D CTS panels for the paper figure.
Scene 0223 fi14 (det-key 0223-0028), target = bus. For BOTH the bus image and the
sedan image, render CAM_BACK at native 1600x900 (clean, no text overlay):
  - GT  : vis>=2 GT only (green)
  - IMG : GT (green) + CTS-IMG prediction (red)
  - CAL : GT (green) + CTS-CAL prediction (red)
Pred = sedan-trained DFA3D (CTS numerator). GT = bus-platform vis>=2 boxes.
GT and preds projected through EACH view's own CAM_BACK geometry; both GT and pred
are filtered to CAM_BACK FOV + in-range so the panel shows only what belongs to it.
Run from the DFA3D repo root in bevformer-b200."""
import os, sys, copy, pickle
import numpy as np, cv2, torch
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
CKPT = 'work_dirs/bevformer_DFA3D_carla/epoch_24.pth'   # sedan-trained base model (CTS numerator)
DATA = '/home/hanyan_arch/viewpoint/BEVFormer/data/nuscenes'
SCENE = os.environ.get('SCENE', '0223-0028'); KEY = tuple(SCENE.split('-'))
CAM = 'CAM_BACK'; THR = 0.3
OUT = f'/home/hanyan_arch/viewpoint/BEVFormer/bevformer_seg/out/qual/dfa3d_cts/{SCENE}_back_hires'
GT_GREEN = (0, 230, 0); PRED_RED = (0, 0, 255); CAL_SKY = (255, 191, 0)  # BGR: red=IMG collapse, sky-blue=CAL recovery
GT_TH = 5; PR_TH = 4
os.makedirs(OUT, exist_ok=True)


class Infer:
    def __init__(self):
        self.cfg = Config.fromfile(CFG)
        self.model = build_model(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))
        load_checkpoint(self.model, CKPT, map_location='cpu')
        self.model.cuda().eval()

    def __call__(self, info, meta):
        p = '/tmp/dfa3d_back.pkl'
        pickle.dump({'infos': [copy.deepcopy(info)], 'metadata': meta}, open(p, 'wb'))
        cfg = copy.deepcopy(self.cfg)
        cfg.data.test.ann_file = p; cfg.data.test.test_mode = True
        ds = build_dataset(cfg.data.test)
        data = scatter(collate([ds[0]], samples_per_gpu=1), [0])[0]
        with torch.no_grad():
            res = self.model(return_loss=False, rescale=True, **data)
        pb = res[0]['pts_bbox']
        return pb['boxes_3d'], pb['scores_3d'].numpy()


def pred_keep(pred, scores):
    """confident + in-pc-range preds (same selection as the grid render)."""
    if len(pred) == 0:
        return np.zeros(0, bool)
    return (scores > THR) & Q.in_range_mask(pred)


def back_geom(info):
    c = info['cams'][CAM]
    l2c = Q.s2l_inv(c); K = np.asarray(c['cam_intrinsic'], float)
    l2i = np.eye(4); l2i[:3, :3] = K; l2i = l2i @ l2c
    img = cv2.imread(Q.resolve_img(c['data_path']))
    if img is None:
        img = np.zeros((900, 1600, 3), np.uint8)
    return img, l2c, l2i


def panel(info, gt, pred, pk, draw_gt, draw_pred, pred_color):
    img, l2c, l2i = back_geom(info)
    ng = 0
    if draw_gt:                                                 # GT only on the GT panel
        mg = Q.in_range_mask(gt) & Q.visible_mask(gt, l2c)      # GT visible in CAM_BACK, vis>=2 already
        img = Q.draw_box3d_lidar(img, gt[mg], l2i, GT_GREEN, GT_TH)
        ng = int(mg.sum())
    np_ = 0
    if draw_pred and len(pred):                                 # pred only on IMG(red)/CAL(sky-blue) panels
        pm = pk & Q.visible_mask(pred, l2c)                    # pred visible in CAM_BACK
        img = Q.draw_box3d_lidar(img, pred[pm], l2i, pred_color, PR_TH)
        np_ = int(pm.sum())
    return img, ng, np_


def main():
    sedan = pickle.load(open(f'{DATA}/sedan_infos_val.pkl', 'rb'))
    sinfo = {B.info_key(i): i for i in sedan['infos']}[KEY]
    bus = pickle.load(open(f'{DATA}/bus_infos_val.pkl', 'rb'))
    binfo = {B.info_key(i): i for i in bus['infos']}[KEY]
    bus_gt = Q.boxes_from(binfo['gt_boxes'], binfo['valid_flag'])   # bus-platform vis>=2 GT
    ver = 'v1.0-carla_bus_eval'

    inf = Infer()
    img_box, img_sc = inf(Q.make_cts_info('IMG', binfo, sinfo), {'version': ver})
    cal_box, cal_sc = inf(Q.make_cts_info('CAL', binfo, sinfo), {'version': ver})
    img_pk, cal_pk = pred_keep(img_box, img_sc), pred_keep(cal_box, cal_sc)

    views = {'bus': binfo, 'sedan': sinfo}   # bus = changed/target image, sedan = baseline image
    # (cond, pred_boxes, pred_keep, draw_gt, draw_pred, pred_color): GT=green only; IMG/CAL=red pred (user: CAL red, not sky-blue)
    conds = [('GT', None, None, True, False, GT_GREEN),
             ('IMG', img_box, img_pk, False, True, PRED_RED),
             ('CAL', cal_box, cal_pk, False, True, PRED_RED)]
    for vname, vinfo in views.items():
        for cname, pbox, pk, dg, dp, pcol in conds:
            im, ng, npred = panel(vinfo, bus_gt, pbox if pbox is not None else bus_gt, pk, dg, dp, pcol)
            path = f'{OUT}/{vname}_{cname}_CAM_BACK.png'
            cv2.imwrite(path, im)
            print(f'{vname:5s} {cname:3s}  GT={ng} pred={npred}  -> {os.path.basename(path)} {im.shape[1]}x{im.shape[0]}', flush=True)


if __name__ == '__main__':
    main()
