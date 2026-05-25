
_base_ = [
    '../_base_/models/uperent_swin.py',
    '../_base_/datasets/scapula_seg_swin.py', #self-defined dataset configuration
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py',
]

#override the model config: bin-classification（background/scapula-seg）
model = dict(
    backbone = dict(
        type='SwinTransformer',
        embed_dim = 96,
        depths = [2, 2, 6, 2],
        num_heads = [3, 6, 12, 24],
        window_size = 7,
        drop_path_rate = 0.3,
        patch_norm = True,
    ),

    decode_head = dict(
        type='UPerHead',
        in_channels = [96,192,384,768],
        in_index = [0,1,2,3],
        channels = 512,
        num_classes = 2, # 0：background 1：scapula
        loss_decode = dict(
            type='CrossEntropyLoss',
            use_sigmoid = False,
            loss_weight = 1.0,
        ),
    ),

    auxiliary_head = dict(
        type='FCNHead',
        in_channels = 384,
        in_index = 2,
        channels = 256,
        num_convs = 1,
        num_classes = 2, #0:background 1：scapula
        loss_decode = dict(
            type='CrossEntropyLoss',
            use_sigmoid = False,
            loss_weight = 0.4,
        ),
    ),
)

#the image size while prediction
test_cfg = dict(model='slide',crop_size=(512,512), stride=(341,341))

#optimizer config（used when train，and not while prediction）
optimizer = dict(type='AdamW', lr=0.00006, weight_decay=0.01)