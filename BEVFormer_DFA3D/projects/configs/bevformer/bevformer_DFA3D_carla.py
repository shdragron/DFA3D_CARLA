"""BEVFormer + DFA3D (3D Deformable Attention) on CARLA geobev — the 7th detector.

GOAL: a single-variable controlled comparison vs our BEVFormer-tiny CARLA model. Same
backbone / BEV grid / single-frame / classes / no-aug fair-protocol; the ONLY addition is
DFA3D's depth-aware 3D feature lifting (gates-sampling + explicit depth → fills the empty
gates×depth quadrant of the VP cross-model 2x2).

Matched to bevformer_tiny_carla.py:
  - R50 ImageNet (torchvision), out_indices=(3,), single FPN level (num_levels=1)
  - BEV 50x50, encoder 3 layers, 800x450 input (RandomScaleImage 0.5)
  - single-frame (queue_length=2 + unique scene_token -> prev_bev=None), CAN-bus off,
    rotate_prev_bev off, video_test_mode off
  - 6 CARLA classes, vis>=2, aug/EMA/CBGS/fp16 off, fp32
DFA3D grafts (single-level, from bevformer_small_DFA3D.py):
  - type=BEVFormer_DFA3D, d_bound=[2,58,0.5]; pts_dpt_head=DepthHead_MLVGDpt at FPN level 0
    (downsample 32), depth loss_weight 0.5; SpatialCrossAttention_DFA3D +
    MSDeformableAttention3D_DFA3D; BEVFormerHead_DFA3D / PerceptionTransformer_DFA3D /
    BEVFormerEncoder_DFA3D.
  - depth GT = CARLA dense DPT (CarlaDPTMultiViewDepthDFA3D: RGB->DPT png, decode
    (R+G*256+B*256^2)/(256^3-1)*1000 planar Z). Scaled/padded with the images.
"""
_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
d_bound = [2.0, 58.0, 0.5]                       # DFA3D depth bins (matches small_DFA3D)

# R50 torchvision ImageNet norm (same as our BEVFormer-tiny, NOT the R101-caffe DFA3D base).
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

class_names = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian']
num_classes = len(class_names)

input_modality = dict(use_lidar=False, use_camera=True, use_radar=False,
                      use_map=False, use_external=True)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 50
bev_w_ = 50
queue_length = 2
downsample_factors = [32]          # single level -> /32 feature grid
indice_layer_depthnet = 0

model = dict(
    type='BEVFormer_DFA3D',
    use_grid_mask=False,                          # aug off
    video_test_mode=False,                        # temporal off
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet', depth=50, num_stages=4, out_indices=(3,),
        frozen_stages=1, norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True, style='pytorch'),
    img_neck=dict(
        type='FPN', in_channels=[2048], out_channels=_dim_, start_level=0,
        add_extra_convs='on_output', num_outs=_num_levels_,
        relu_before_extra_convs=True),
    d_bound=d_bound,
    pts_dpt_head=dict(
        type='DepthHead_MLVGDpt',
        max_tol=0,
        in_channels=_dim_,
        cam_channel=0,                            # don't use cam intrinsic/extrinsic in depth net
        mid_channels=256,
        out_channels=256,
        downsample_factor=downsample_factors[indice_layer_depthnet],
        dbound=d_bound,
        loss_weight=0.5,
        indice_layer=indice_layer_depthnet,
        sfm_or_sig=True),
    pts_bbox_head=dict(
        type='BEVFormerHead_DFA3D',
        bev_h=bev_h_, bev_w=bev_w_, num_query=900, num_classes=num_classes,
        in_channels=_dim_, sync_cls_avg_factor=True, with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer_DFA3D',
            rotate_prev_bev=False, use_shift=True, use_can_bus=False,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder_DFA3D',
                d_bound=d_bound,
                num_layers=3, pc_range=point_cloud_range, num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(type='TemporalSelfAttention', embed_dims=_dim_, num_levels=1),
                        dict(
                            type='SpatialCrossAttention_DFA3D',
                            bev_h=bev_h_, bev_w=bev_w_, num_head=8, use_empty=False,
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D_DFA3D',
                                embed_dims=_dim_, num_points=8, num_levels=_num_levels_,
                                # im2col_step must divide the per-call batch (samples_per_gpu*num_cams).
                                # 2-GPU: per-GPU bs8 -> 8*6=48; 48 | 48. (Single-GPU bs16 -> 96; 48 | 96
                                # too.) Numerically-irrelevant CUDA tiling step; safe for both layouts.
                                im2col_step=48),
                            embed_dims=_dim_)
                    ],
                    feedforward_channels=_ffn_dim_, ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6, return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention', embed_dims=_dim_, num_heads=8, dropout=0.1),
                        dict(type='CustomMSDeformableAttention', embed_dims=_dim_, num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_, ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range, max_num=300, voxel_size=voxel_size,
            num_classes=num_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding', num_feats=_pos_dim_,
            row_num_embed=bev_h_, col_num_embed=bev_w_),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1], voxel_size=voxel_size,
        point_cloud_range=point_cloud_range, out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range))))

dataset_type = 'CarlaNuScenesDataset'
data_root = 'data/nuscenes/'           # symlink -> carla_geobev
vehicle = 'sedan'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # aug off (no PhotoMetricDistortion)
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='CarlaDPTMultiViewDepthDFA3D'),     # dense DPT depth map -> results['dpt']
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImageDpt', scales=[0.5]),   # 1600x900 -> 800x450 (scales 'dpt' too via nearest)
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'dpt'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D', img_scale=(1600, 900), pts_scale_ratio=1, flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='CustomDefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

# FAIR MATCH to the BEVFormer-tiny CARLA baseline: that model trained at GLOBAL
# batch 16 with lr=4e-4 (samples_per_gpu=8 x 2 B200). We run 2-GPU DDP the same way:
# samples_per_gpu=8 x 2 GPU = global batch 16, lr=4e-4 unchanged. Frozen backbone BN
# (norm_eval=True, requires_grad=False) means the batch statistics are identical, so
# this is a faithful single-variable (depth-aware lifting) comparison vs BEVFormer-tiny.
# IMPORTANT: when changing #GPUs, keep GLOBAL batch (=samples_per_gpu * #GPU) and lr
# fixed -- do NOT scale lr with GPU count here (global batch is held at 16).
data = dict(
    # workers kept modest: the dense DPT depth loader adds 6 full-res maps/sample,
    # and we share the box with the running VP jobs -> avoid the SIGKILL OOM burst.
    samples_per_gpu=8, workers_per_gpu=4,
    train=dict(type=dataset_type, data_root=data_root,
               ann_file=data_root + f'{vehicle}_infos_train.pkl',
               pipeline=train_pipeline, classes=class_names, modality=input_modality,
               test_mode=False, use_valid_flag=True, bev_size=(bev_h_, bev_w_),
               queue_length=queue_length, box_type_3d='LiDAR'),
    val=dict(type=dataset_type, data_root=data_root,
             ann_file=data_root + f'{vehicle}_infos_val.pkl',
             pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type, data_root=data_root,
              ann_file=data_root + f'{vehicle}_infos_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

optimizer = dict(
    type='AdamW', lr=4e-4,
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=500,
                 warmup_ratio=1.0 / 3, min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=2, pipeline=test_pipeline)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(interval=50, hooks=[
    dict(type='TextLoggerHook'),
    dict(type='WandbLoggerHook',
         init_kwargs=dict(project='DFA3D_CARLA',
                          name=f'DFA3D_carla_{vehicle}',
                          tags=['dfa3d', 'bevformer', 'carla', vehicle, 'gates+depth']),
         interval=50, by_epoch=True, log_artifact=False)])
checkpoint_config = dict(interval=1)
