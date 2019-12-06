# model settings
input_size = 300
width_mult = 0.5
model = dict(
    type='SingleStageDetector',
    #pretrained='./mv2_72.5.pth',
    backbone=dict(
        type='SSDMV2',
        input_size=input_size,
        width_mult=width_mult,
        out_feature_indices=(14,)
        ),
    neck=None,
    bbox_head=dict(
        type='SSDLiteHead',
        input_size=input_size,
        #in_channels=(576, 1280, 512, 256, 256, 128),
        in_channels=(int(width_mult*576), 1280, 512, 256, 256, 128),
        num_classes=3,
        anchor_strides=(16, 32, 64, 100, 150, 300),
        #anchor_strides=(16, 32, 64, 100, 150, 300),
        #basesize_ratio_range=(0.15, 0.9),
        basesize_ratio_range=(0.2, 0.9),
        anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
        target_means=(.0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2)))
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.6),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=100)
# model training and testing settings
# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2028/ImageSets/Main/trainval.txt',
            ],
            img_prefix=[data_root + 'VOC2028/'],
            img_scale=(300, 300),
            img_norm_cfg=img_norm_cfg,
            size_divisor=None,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=False,
            with_label=True,
            test_mode=False,
            extra_aug=dict(
                photo_metric_distortion=dict(
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                expand=dict(
                    mean=img_norm_cfg['mean'],
                    to_rgb=img_norm_cfg['to_rgb'],
                    ratio_range=(1, 2)),
                random_crop=dict(
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.1)),
            resize_keep_ratio=False)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2028/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2028/',
        img_scale=(300, 300),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2028/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2028/',
        img_scale=(300, 300),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False))
# optimizer
#optimizer = dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=4e-5)
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 20])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
evaluation = dict(interval=1)
# yapf:enable
# runtime settings
total_epochs = 24
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './models/wider_face_ssd300_mobilenet'
load_from = None
resume_from = None
workflow = [('train', 1)]
