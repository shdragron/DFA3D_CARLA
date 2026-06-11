"""DFA3D-CARLA BUS oracle — thin override of bevformer_DFA3D_carla.py (sedan).

Only the vehicle (ann_file pkls + wandb run name) changes; backbone / BEV grid /
single-frame / depth / fair-protocol / optimizer / schedule are inherited unchanged,
so the bus oracle is trained identically to sedan (CTS P_TARGET denominator).
"""
_base_ = ['./bevformer_DFA3D_carla.py']

vehicle = 'bus'
data_root = 'data/nuscenes/'
data = dict(
    train=dict(ann_file=data_root + 'bus_infos_train.pkl'),
    val=dict(ann_file=data_root + 'bus_infos_val.pkl'),
    test=dict(ann_file=data_root + 'bus_infos_val.pkl'))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='DFA3D_CARLA', name='DFA3D_carla_bus',
                              tags=['dfa3d', 'bevformer', 'carla', 'bus', 'gates+depth']),
             interval=50, by_epoch=True, log_artifact=False)])
