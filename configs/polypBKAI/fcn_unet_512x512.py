_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py',
]

model = dict(
    backbone=dict(
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
    ),
    decode_head=dict(
        num_classes=3,
        loss_decode=dict(
            type='FocalLoss',
            use_sigmoid=True
        )
    ),
    auxiliary_head=dict(
        num_classes=3,
        loss_decode=dict(
            type='FocalLoss',
            use_sigmoid=True
        )
    ),
    test_cfg=dict(
        crop_size=(128, 128),
        stride=(85, 85))
)

# dataset settings
dataset_type = 'PolypDataset'
data_root = '../Dataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=crop_size, ratio_range=(1.0, 1.0), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val',
        ann_dir='val_seg_map',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val',
        ann_dir='val_seg_map',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val',
        ann_dir='val_seg_map',
        pipeline=test_pipeline
    )
)

runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])

optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=2
)