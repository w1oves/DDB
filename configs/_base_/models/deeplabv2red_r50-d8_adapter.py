# models settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='UDAEncoderDecoder',
    backbone=dict(
        type='ResNet',
        init_cfg=dict(type='Pretrained', checkpoint='checkpoints/resnet/resnet50.pth'),
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DLV2AdapterHead',
        in_channels=2048,
        channels=256,
        in_index=3,
        dilations=(6, 12),
        num_classes=19,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        init_cfg=dict(
            type='Normal', std=0.01, override=dict(name='aspp_modules')),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # models training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))














'''###____pretty_text____###'''



'''
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ResNet',
        init_cfg=dict(
            type='Pretrained', checkpoint='checkpoints/resnet/resnet50.pth'),
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DLV2AdapterHead',
        in_channels=2048,
        channels=256,
        in_index=3,
        dilations=(6, 12),
        num_classes=19,
        align_corners=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        init_cfg=dict(
            type='Normal', std=0.01, override=dict(name='aspp_modules')),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
'''
