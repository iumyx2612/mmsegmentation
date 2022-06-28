_base_ = [
    '../_base_/schedules/schedule_40k.py',
    '../_base_/default_runtime.py'
]


custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
pretrained =\
    "https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b4_3rdparty_8xb32_in1k_20220119-81fd4077.pth"

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='mmcls.EfficientNet',
        arch='b4',
        out_indices=(2, 3, 4, 5),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone.'
        )
    ),
    neck=dict(
        type='SSFPN',
        in_channels=[24, 40, 112, 320],
        out_channels=256,
        num_outs=4,
        segmentation_channels=128
    ),
    decode_head=dict(
        type='BasicHead',
        in_channels=128,
        channels=128,
        num_classes=3,
        loss_decode=[
            dict(
                type='DiceLoss'
            ),
            dict(
                type='FocalLoss'
            )
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
    dict(
        type='RandomMosaic',
        prob=0.3,
        pad_val=0,
        seg_pad_val=0,
        img_scale=crop_size,
        center_ratio_range=(1.0, 1.0)
    ),
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

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/train',
        ann_dir='train_seg_map',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False)
        ]
    ),
    pipeline=train_pipeline
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=train_dataset,
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

checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])