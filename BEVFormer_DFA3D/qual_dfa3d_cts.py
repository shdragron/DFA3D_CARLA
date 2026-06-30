"""DFA3D CTS qualitative: suv/bus x IMG/CAL. For each case render GT(vis>=2,green)
+ pred(red) on (a) sedan images, (b) target images, and (c) BEV. Run from the
DFA3D repo root with bevformer-b200 env. vis>=2 enforced via valid_flag."""
import os, sys, copy, pickle
import numpy as np, cv2, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
import importlib; importlib.import_module('projects.mmdet3d_plugin')   # DFA3D plugin (cwd)
BENCH = '/home/hanyan_arch/viewpoint/BEVFormer/bev_det_benchmark'
sys.path.insert(0, BENCH)
import build_condition_pkls as B
import qual_conditions as Q

CFG = 'projects/configs/bevformer/bevformer_DFA3D_carla.py'
CKPT = 'work_dirs/bevformer_DFA3D_carla/epoch_24.pth'   # sedan-trained base model (CTS numerator)
DATA = '/home/hanyan_arch/viewpoint/BEVFormer/data/nuscenes'
SCENE = os.environ.get('SCENE', '0256-0120')            # det-key 'scene-frame'; 0256-0120 = teaser scene_0256 f60
KEY = tuple(SCENE.split('-')); THR = 0.3; PC = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
OUT = f'/home/hanyan_arch/viewpoint/BEVFormer/bevformer_seg/out/qual/dfa3d_cts/{SCENE}'
os.makedirs(OUT, exist_ok=True)


class Infer:
    def __init__(self):
        self.cfg = Config.fromfile(CFG)
        self.model = build_model(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))
        load_checkpoint(self.model, CKPT, map_location='cpu')
        self.model.cuda().eval()

    def __call__(self, info, meta):
        p = '/tmp/dfa3d_one.pkl'
        pickle.dump({'infos': [copy.deepcopy(info)], 'metadata': meta}, open(p, 'wb'))
        cfg = copy.deepcopy(self.cfg)
        cfg.data.test.ann_file = p
        cfg.data.test.test_mode = True
        ds = build_dataset(cfg.data.test)
        data = scatter(collate([ds[0]], samples_per_gpu=1), [0])[0]
        with torch.no_grad():
            res = self.model(return_loss=False, rescale=True, **data)
        pb = res[0]['pts_bbox']
        return pb['boxes_3d'], pb['scores_3d'].numpy()


def pred_mask(pred, scores):
    if len(pred) == 0:
        return np.zeros(0, bool)
    c = pred.gravity_center.numpy()
    inr = np.array([PC[0] <= x[0] <= PC[3] and PC[1] <= x[1] <= PC[4] for x in c], bool)
    return (scores > THR) & inr


def render_cam(info, gt, pred, pm, cam):
    c = info['cams'][cam]
    img = cv2.imread(Q.resolve_img(c['data_path']))
    if img is None:
        img = np.zeros((900, 1600, 3), np.uint8)
    l2c = Q.s2l_inv(c); K = np.asarray(c['cam_intrinsic'], float)
    l2i = np.eye(4); l2i[:3, :3] = K; l2i = l2i @ l2c
    mg = Q.in_range_mask(gt) & Q.visible_mask(gt, l2c)
    if mg.any():
        img = Q.draw_box3d_lidar(img, gt[mg], l2i, (0, 230, 0), 3)         # GT green
    if len(pred) and pm.any():
        img = Q.draw_box3d_lidar(img, pred[pm], l2i, (0, 0, 255), 2)        # pred red
    cv2.putText(img, f'{cam} GT={int(mg.sum())} pred={int(pm.sum()) if len(pm) else 0}',
                (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    return img


def six_view(info, gt, pred, pm, banner, path):
    panels = {cam: render_cam(info, gt, pred, pm, cam) for cam in Q.CAMS}
    grid = np.vstack([np.hstack([panels[c] for c in row]) for row in Q.GRID])
    cv2.putText(grid, banner, (12, grid.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.imwrite(path, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])


def rect(b):
    x, y, dx, dy, yaw = [float(v) for v in b]
    cs, sn = np.cos(yaw), np.sin(yaw); R = np.array([[cs, -sn], [sn, cs]])
    p = np.array([[-dx/2, -dy/2], [dx/2, -dy/2], [dx/2, dy/2], [-dx/2, dy/2]])
    return p @ R.T + [x, y]


def bev(gt, pred, pm, path, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-51.2, 51.2); ax.set_ylim(-51.2, 51.2); ax.set_aspect('equal')
    ax.invert_xaxis(); ax.plot(0, 0, 'k+', ms=12); ax.axis('off')
    for b in gt.bev.numpy():
        ax.add_patch(Polygon(rect(b)[:, [1, 0]], closed=True, fill=False, ec='#22aa22', lw=2.2))
    if len(pred) and pm.any():
        for b in pred[pm].bev.numpy():
            ax.add_patch(Polygon(rect(b)[:, [1, 0]], closed=True, fill=False, ec='#d62728', lw=2.0))
    ax.set_title(title, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=95); plt.close(fig)


def main():
    sedan = pickle.load(open(f'{DATA}/sedan_infos_val.pkl', 'rb'))
    sidx = {B.info_key(i): i for i in sedan['infos']}; sinfo = sidx[KEY]
    targets = {
        'suv': (pickle.load(open(f'{DATA}/suv_infos_val.pkl', 'rb')), 'v1.0-carla_suv_eval'),
        'bus': (pickle.load(open(f'{DATA}/bus_infos_val.pkl', 'rb')), 'v1.0-carla_bus_eval'),
    }
    inf = Infer()
    for tgt, (tdata, ver) in targets.items():
        tidx = {B.info_key(i): i for i in tdata['infos']}; tinfo = tidx[KEY]
        tgt_gt = Q.boxes_from(tinfo['gt_boxes'], tinfo['valid_flag'])     # target vis>=2 GT
        for cond in ['IMG', 'CAL']:
            cinfo = Q.make_cts_info(cond, tinfo, sinfo)
            pbox, psc = inf(cinfo, {'version': ver})
            pm = pred_mask(pbox, psc)
            tag = f'{tgt}_{cond}'
            six_view(sinfo, tgt_gt, pbox, pm, f'DFA3D CTS-{tgt} {cond}  img=SEDAN  GT=green pred=red>{THR}',
                     f'{OUT}/{tag}_on_sedan.jpg')
            six_view(tinfo, tgt_gt, pbox, pm, f'DFA3D CTS-{tgt} {cond}  img={tgt.upper()}  GT=green pred=red>{THR}',
                     f'{OUT}/{tag}_on_{tgt}.jpg')
            bev(tgt_gt, pbox, pm, f'{OUT}/{tag}_BEV.png', f'DFA3D CTS-{tgt} {cond} (BEV)  GT=green pred=red')
            print(f'{tag}: GT={len(tgt_gt)} pred={int(pm.sum())}', flush=True)


if __name__ == '__main__':
    main()
