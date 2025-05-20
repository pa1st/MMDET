model = dict(
    type='SSD3DNet',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type="PointTransformerV3",
        in_channels=4,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        #输出通道dec_channels[0]
        dec_channels=(256, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    bbox_head=dict(
        type='SSD3DHead',
        vote_module_cfg=dict(
            #backbone输出通道数，head输入通道数
            in_channels=256,
            num_points=256,
            gt_per_seed=1,
            conv_channels=(128,),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.1),
            with_res_feat=False,
            vote_xyz_range=(3.0, 3.0, 2.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModuleMSG',
            num_point=256,
            radii=(4.8, 6.4),
            sample_nums=(16, 32),
            mlp_channels=((256, 256, 256, 512), (256, 256, 512, 1024)),
            norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.1),
            use_xyz=True,
            normalize_xyz=False,
            bias=True),
        pred_layer_cfg=dict(
            in_channels=1536,
            shared_conv_channels=(512, 128),
            cls_conv_channels=(128,),
            reg_conv_channels=(128,),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.1),
            bias=True),
        objectness_loss=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        center_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0),
        dir_class_loss=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0),
        corner_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0),
        vote_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        sample_mode='spec', pos_distance_thr=10.0, expand_dims_length=0.05),
    test_cfg=dict(
        nms_cfg=dict(type='nms', iou_thr=0.1),
        sample_mode='spec',
        score_thr=0.0,
        per_class_proposal=True,
        max_output_num=100))
