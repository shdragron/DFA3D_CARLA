"""CTS cross-platform-transfer eval for 3D detection (NDS analogue of
bev_seg_benchmark/eval_cts_cvt.py).

A sedan-trained model is evaluated on the suv/bus eval data under 4 conditions
that independently swap the image source and the extrinsic source:

    NORMAL = (sedan img, sedan ext)   <- sedan-inputs reference (== stock sedan_infos_val.pkl)
    EXT    = (sedan img, target ext)
    IMG    = (target img, sedan ext)  <- primary cross-platform metric
    CAL    = (target img, target ext)

For each condition we build a condition pkl (build_condition_pkls.make_cts_pkl),
run the model's stock test entry (run_bevformer.sh -> tools/dist_test.sh), scrape
the deterministic ``[CARLA-EVAL] 6-class mAP=.. NDS=..`` line, and report
    CTS_c = NDS_c / P_TARGET        (paper Eq.6). The denominator P_TARGET is the
    target-NATIVE oracle: a model TRAINED ON the target platform, evaluated on its
    own eval set (PER target) -- NOT the sedan model's NORMAL. The 4 sedan-model
    conditions (incl. NORMAL) are the numerators.

Outputs (mirroring the seg script): eval_cts.csv, eval_cts.json, eval_cts_summary.txt.

Example:
    conda activate bevformer-b200
    python bev_det_benchmark/eval_cts_det.py \
        --config projects/configs/bevformer/bevformer_tiny_carla.py \
        --ckpt   work_dirs/bevformer_tiny_carla_sedan/latest.pth \
        --ngpu 2 --tag tiny_sedan
    # smoke test (one cell): --targets suv --conditions IMG
"""
import argparse
import csv
import json
import os
import pickle as _pk
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BEVF_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

CARLA_EVAL_RE = re.compile(r'\[CARLA-EVAL\]\s+\d+-class\s+mAP=([\d.]+)\s+NDS=([\d.]+)')
CARLA_METRICS_RE = re.compile(r'\[CARLA-METRICS-JSON\]\s+(\{.*\})\s*$')
P = 'pts_bbox_NuScenes/'   # detail-dict key prefix (same for bevformer/bevdepth)
CTS_COND_NAMES = ['NORMAL', 'EXT', 'IMG', 'CAL']
# The condition-pkl builder (B) and the per-framework runner are selected in
# main() from --framework, so adding bevdepth never touches the bevformer path.


def _cell_path(cell_dir, target, cond):
    """Per-cell result-cache path (resume: skip already-done cells on restart).

    Persistent (lives in the run outdir/cells), NOT tmpfs/scratch."""
    name = f"{target}_{cond}".replace('/', '-')
    return os.path.join(cell_dir, name + '.pkl')


def _save_cell(path, obj):
    """Crash-safe per-cell cache write: dump to tmp, fsync, then atomic replace.

    A SIGKILL mid-dump can only truncate the .tmp file; the real path is swapped
    in by os.replace() (atomic on POSIX) only after data hits disk."""
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        _pk.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_cell(path):
    """Safe per-cell cache read: return None (and drop the file) on any failure.

    A truncated/corrupt .pkl (e.g. from an old non-atomic SIGKILL) raises on
    load; we remove it so the cell recomputes instead of aborting the restart."""
    try:
        with open(path, 'rb') as f:
            return _pk.load(f)
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def nds_components(metrics):
    """Pull the 6-class NDS ingredients (+10-class headline) from a detail dict."""
    g = (lambda k: metrics.get(P + k)) if metrics else (lambda k: None)
    return {'mATE': g('mATE_6class'), 'mASE': g('mASE_6class'),
            'mAOE': g('mAOE_6class'), 'mAVE': g('mAVE_6class'),
            'mAAE': g('mAAE_6class'), 'nds_10class': g('NDS_10class'),
            'map_10class': g('mAP_10class')}


def run_one(run_sh, config, ckpt, ngpu, cond_pkl, log_path):
    """Shell out to the framework runner; scrape NDS/mAP + the full metric dump.

    run_sh is run_bevformer.sh or run_bevdepth.sh; both take the same 4-arg
    contract <CONFIG/EXP> <CKPT> <NGPU> <COND_PKL> and print the deterministic
    ``[CARLA-EVAL]`` / ``[CARLA-METRICS-JSON]`` lines scraped here.

    Returns {'nds','map6','metrics'} where 'metrics' is the complete detail dict
    (per-class AP at all dist thresholds, per-class/6-class/10-class TP errors,
    NDS, mAP). The whole stdout is tee'd to log_path.
    """
    cmd = ['bash', run_sh, config, ckpt, str(ngpu), cond_pkl]
    print(f'  $ {" ".join(cmd)}', flush=True)
    nds = map6 = metrics = None
    with open(log_path, 'w') as logf:
        proc = subprocess.Popen(cmd, cwd=BEVF_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            logf.write(line)
            m = CARLA_EVAL_RE.search(line)
            if m:
                map6, nds = float(m.group(1)), float(m.group(2))
                print(f'    >> {line.strip()}', flush=True)
            mj = CARLA_METRICS_RE.search(line)
            if mj:
                try:
                    metrics = json.loads(mj.group(1))
                except json.JSONDecodeError:
                    pass
        proc.wait()
    if nds is None:
        raise RuntimeError(f'no [CARLA-EVAL] line scraped; see {log_path}')
    if metrics is not None:                       # prefer the exact dumped values
        nds = metrics.get(P + 'NDS', nds)
        map6 = metrics.get(P + 'mAP', map6)
    if proc.returncode != 0:
        print(f'  WARN: dist_test returncode={proc.returncode} '
              f'(NDS scraped anyway)', flush=True)
    return {'nds': nds, 'map6': map6, 'metrics': metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='projects/configs/bevformer/bevformer_DFA3D_carla.py')
    ap.add_argument('--ckpt', default='work_dirs/bevformer_DFA3D_carla/epoch_24.pth',
                    help='SOURCE (sedan) model = the numerator (transferred model)')
    ap.add_argument('--target-ckpt-tmpl',
                    default='work_dirs/bevformer_DFA3D_carla_{}/epoch_24.pth',
                    help='TARGET model = the denominator (native upper bound); '
                         '{} filled with suv/bus')
    ap.add_argument('--ngpu', type=int, default=1)
    ap.add_argument('--tag', default='dfa3d_sedan', help='output subdir name')
    ap.add_argument('--targets', nargs='+', default=['suv', 'bus'],
                    choices=['suv', 'bus'])
    ap.add_argument('--conditions', nargs='+', default=['NORMAL', 'EXT', 'IMG', 'CAL'],
                    choices=CTS_COND_NAMES)
    ap.add_argument('--framework', default='bevformer',
                    choices=['bevformer', 'bevdepth', 'bevdet', 'cape', 'petrv2', 'detr3d'],
                    help='detector stack: selects the runner, the condition-pkl '
                         'schema, and (bevdepth) the per-target exp. cape/petrv2/'
                         'detr3d (sparse) reuse the bevformer condition pkls + a '
                         'single-GPU runner (run_<framework>.sh)')
    ap.add_argument('--exp-tmpl',
                    default='bevdepth/exps/nuscenes/carla/carla_{}.py',
                    help='bevdepth only: per-target exp ({}=suv/bus), run from '
                         'the BEVDepth root; it sets the eval DB version')
    ap.add_argument('--outdir', default=os.path.join(HERE, 'out'))
    args = ap.parse_args()

    config = os.path.join(BEVF_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    ckpt = os.path.join(BEVF_ROOT, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt

    def abspath(p):
        return p if os.path.isabs(p) else os.path.join(BEVF_ROOT, p)

    # ---- framework wiring: runner + condition-pkl builder + per-target exp ----
    # additive: bevformer (default) keeps the original behaviour exactly.
    if args.framework == 'bevdepth':
        import build_condition_pkls_bevdepth as B
        run_sh = os.path.join(HERE, 'run_bevdepth.sh')

        def cfg_for(_t):
            return args.exp_tmpl.format(_t)            # per-target exp path

        def target_val_pkl(_t):
            return os.path.join(B.BEVDEPTH_DATA, f'carla_infos_val_{_t}.pkl')
    elif args.framework == 'bevdet':
        import build_condition_pkls_bevdet as B
        run_sh = os.path.join(HERE, 'run_bevdet.sh')

        def cfg_for(_t):
            return config                              # single bevdet config; eval DB from cond-pkl version

        def target_val_pkl(_t):
            return os.path.join(B.BEVDET_DATA, f'{_t}_infos_val.pkl')
    elif args.framework in ('cape', 'petrv2', 'detr3d'):
        # Sparse detectors: SAME carla pkls + cam fields as bevformer, so reuse
        # the bevformer condition-pkl builder verbatim. Only the runner differs
        # (run_<framework>.sh = single-GPU tools/test.py in the legacy env).
        import build_condition_pkls as B
        run_sh = os.path.join(HERE, 'sparse', f'run_{args.framework}.sh')

        def cfg_for(_t):
            return config                              # single config; eval DB from cond-pkl version

        def target_val_pkl(_t):
            return os.path.join(B.DATA_ROOT, f'{_t}_infos_val.pkl')
    else:
        import build_condition_pkls as B
        run_sh = os.path.join(HERE, 'run_dfa3d.sh')

        def cfg_for(_t):
            return config                              # single bevformer config

        def target_val_pkl(_t):
            return os.path.join(B.DATA_ROOT, f'{_t}_infos_val.pkl')

    outdir = os.path.join(args.outdir, f'cts_{args.tag}')
    pkldir = os.path.join(outdir, 'pkls')
    logdir = os.path.join(outdir, 'logs')
    # PER-CELL result cache (resume): each finished cell's row is dumped here the
    # moment it completes; on restart, done cells load from here and are SKIPPED.
    # Persistent (in outdir), NOT tmpfs/scratch. No --shard here -> just cells/.
    # Each cell is its own subprocess (run_one) -> single-process, no rank gating.
    cell_dir = os.path.join(outdir, 'cells')
    for d in (pkldir, logdir, cell_dir):
        os.makedirs(d, exist_ok=True)

    # ---- CTS_c^(r) = P_c^(r) / P_TARGET^(r)  (paper Eq. 6) --------------------
    #   P_TARGET^(r) = DENOMINATOR/ORACLE = a model TRAINED ON platform r and
    #       evaluated on r (target ckpt on target's own data). PER TARGET.
    #   P_c^(r) = NUMERATOR = the SEDAN-trained model deployed on platform r
    #       (no retraining) under condition c, on the TARGET eval set:
    #       NORMAL = sedan img+ext (reference) ; EXT = sedan img+target ext ;
    #       IMG = target img+sedan ext (primary) ; CAL = target img+ext (full deploy).
    rows = []          # dicts: platform, condition, nds, map6, cts, comp, metrics
    oracles = {}       # target -> P_TARGET (oracle: target model on target)

    def add_row(platform, cond, res, cts):
        rows.append({'platform': platform, 'condition': cond, 'nds': res['nds'],
                     'map6': res['map6'], 'cts': cts,
                     'comp': nds_components(res['metrics']),
                     'metrics': res['metrics']})

    ORDER = ['NORMAL', 'EXT', 'IMG', 'CAL']
    conds = [c for c in ORDER if c in args.conditions]
    for target in args.targets:
        exp = cfg_for(target)        # bevformer: config ; bevdepth: carla_<t>.py
        # denominator P_TARGET = platform-matched oracle (target model on target)
        # resume: cache the ORACLE cell (the CTS denominator) too -> a restart
        # loads its row + p_target instead of recomputing.
        ocp = _cell_path(cell_dir, target, 'ORACLE')
        _cached = _load_cell(ocp)
        if _cached is not None:
            orow = _cached
            p_target = orow['nds']
            oracles[target] = p_target
            rows.append(orow)
            print(f'[{target}/ORACLE] (cached) P_TARGET (target-model on {target}) '
                  f'NDS={p_target:.4f}', flush=True)
        else:
            target_pkl = target_val_pkl(target)
            target_ckpt = abspath(args.target_ckpt_tmpl.format(target))
            den = run_one(run_sh, exp, target_ckpt, args.ngpu, target_pkl,
                          os.path.join(logdir, f'{target}_ORACLE.log'))
            p_target = den['nds']
            oracles[target] = p_target
            add_row(target, 'ORACLE', den, 1.0)
            _save_cell(ocp, rows[-1])                 # resume: per-cell save (atomic)
            print(f'[{target}/ORACLE] P_TARGET (target-model on {target}) '
                  f'NDS={p_target:.4f}', flush=True)
        # numerators P_c = sedan model under each condition on the target eval set
        for cond in conds:
            cp = _cell_path(cell_dir, target, cond)
            _cached = _load_cell(cp)                  # resume: skip already-done cell
            if _cached is not None:
                row = _cached
                rows.append(row)
                print(f'[{target}/{cond}] (cached) sedan NDS={row["nds"]:.4f} '
                      f'CTS={row["cts"]:.4f} (/{p_target:.4f})', flush=True)
                continue
            pkl = os.path.join(pkldir, f'{target}_{cond}_infos_val.pkl')
            n, mt, ms = B.make_cts_pkl(cond, target=target, out_path=pkl)
            res = run_one(run_sh, exp, ckpt, args.ngpu, pkl,
                          os.path.join(logdir, f'{target}_{cond}.log'))
            cts = res['nds'] / p_target if p_target else float('nan')
            add_row(target, cond, res, cts)
            _save_cell(cp, rows[-1])                  # resume: per-cell save (atomic)
            print(f'[{target}/{cond}] sedan NDS={res["nds"]:.4f} '
                  f'CTS={cts:.4f} (/{p_target:.4f})', flush=True)

    # ---- write outputs (every NDS component, not just the headline) ----------
    comp_cols = ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE', 'nds_10class', 'map_10class']
    fmt = lambda v: '' if v is None else f'{v:.4f}'
    csv_path = os.path.join(outdir, 'eval_cts.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['platform', 'condition', 'nds', 'map6'] + comp_cols
                   + ['cts_nds', 'primary'])
        for r in rows:
            w.writerow([r['platform'], r['condition'], fmt(r['nds']), fmt(r['map6'])]
                       + [fmt(r['comp'][c]) for c in comp_cols]
                       + [fmt(r['cts']), 'IMG' if r['condition'] == 'IMG' else ''])

    js = {'config': args.config, 'ckpt': args.ckpt, 'tag': args.tag,
          'oracle_nds_per_target': oracles, 'rows': rows}  # rows carry full 'metrics'
    with open(os.path.join(outdir, 'eval_cts.json'), 'w') as f:
        json.dump(js, f, indent=2)

    # by (platform, condition) for the wide table
    by = {(r['platform'], r['condition']): r for r in rows}
    nds_of = lambda p, c: by[(p, c)]['nds'] if (p, c) in by else float('nan')
    cts_of = lambda p, c: by[(p, c)]['cts'] if (p, c) in by else float('nan')

    lines = [f'CTS detection eval  (NDS)   tag={args.tag}    [paper Eq.6]',
             f'  numerator P_c = SEDAN model {args.ckpt} deployed on target',
             f'  denominator P_TARGET = ORACLE = target-trained model '
             f'({args.target_ckpt_tmpl}) on its own platform',
             '  CTS_c^(r) = P_c^(r) / P_TARGET^(r)   [IMG primary]', '']
    # wide layout:  P_SUV EXT IMG CAL  |  P_BUS EXT IMG CAL   (P = oracle P_TARGET)
    hdr = '  ' + ''.join(f'{h:>9}' for h in
                         ['P_SUV', 'EXT', 'IMG', 'CAL', 'P_BUS', 'EXT', 'IMG', 'CAL'])
    nds_row = '  NDS' + ''.join(f'{v:>9.4f}' for v in
        [nds_of('suv', 'ORACLE'), nds_of('suv', 'EXT'), nds_of('suv', 'IMG'), nds_of('suv', 'CAL'),
         nds_of('bus', 'ORACLE'), nds_of('bus', 'EXT'), nds_of('bus', 'IMG'), nds_of('bus', 'CAL')])
    cts_row = '  CTS' + ''.join(f'{v:>9.4f}' for v in
        [cts_of('suv', 'ORACLE'), cts_of('suv', 'EXT'), cts_of('suv', 'IMG'), cts_of('suv', 'CAL'),
         cts_of('bus', 'ORACLE'), cts_of('bus', 'EXT'), cts_of('bus', 'IMG'), cts_of('bus', 'CAL')])
    lines += [hdr, nds_row, cts_row, '',
              f'  {"platform":<10}{"cond":<7}{"NDS":>8}{"mAP6":>8}{"mATE":>7}'
              f'{"mASE":>7}{"mAOE":>7}{"mAVE":>7}{"mAAE":>7}{"CTS":>8}']
    for r in rows:
        c = r['comp']
        star = '  <- primary' if r['condition'] == 'IMG' else ''
        lines.append(
            f'  {r["platform"]:<10}{r["condition"]:<7}{r["nds"]:>8.4f}'
            f'{(r["map6"] or float("nan")):>8.4f}'
            + ''.join(f'{(c[k] if c[k] is not None else float("nan")):>7.3f}'
                      for k in ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE'])
            + f'{r["cts"]:>8.4f}{star}')
    txt = '\n'.join(lines)
    with open(os.path.join(outdir, 'eval_cts_summary.txt'), 'w') as f:
        f.write(txt + '\n')
    print('\n' + txt)
    print(f'\nwrote {csv_path}\n      {os.path.join(outdir, "eval_cts.json")}\n'
          f'      {os.path.join(outdir, "eval_cts_summary.txt")}')


if __name__ == '__main__':
    main()
