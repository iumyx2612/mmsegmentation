_base_ = [
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

# TODO: Add Rotate Augmentation                 (data)
#       No Pre-trained Backbone                 (model)
#       Add Dice Loss                           (model)
#       FPN --> CSPFPN                          (model)
#       Attention on output from FPN/CSPFPN     (model)
#       Auxiliary head                          (model)*
#       U-Net style Decoder                     (model)

custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
pretrained =\
    "https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b1_3rdparty_8xb32-aa-advprop_in1k_20220119-5715267d.pth"

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='mmcls.EfficientNet',
        arch='b1',
        out_indices=(2, 3, 4, 5),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone.'
        )
    ),
    neck=dict(
        type='FPN',
        in_channels=[24, 40, 112, 320],
        out_channels=256,
        num_outs=4
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=[256, 256, 256, 256],
        channels=128,
        num_classes=3,
        in_index=[0, 1, 2, 3],
        input_transform='resize_concat',
        concat_input=False,
        loss_decode=[
            dict(
            type='FocalLoss',
            use_sigmoid=True)
        ]
    ),
    test_cfg=dict(mode='whole')
)

# dataset settings
dataset_type = 'PolypDataset'
data_root = '../Dataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=crop_size, keep_ratio=False, ratio_range=(1, 1)),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
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
    samples_per_gpu=1,
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

'''optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=2
)'''