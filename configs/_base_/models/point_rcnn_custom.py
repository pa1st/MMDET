model = dict(
    type='PointRCNN',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type="PointTransformerV3RCNN",
        in_channels=4,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        #输出通道dec_channels[0]
        dec_channels=(128, 192, 256, 384),
        dec_num_head=(4, 6, 16, 24),
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
    # neck=dict(
    #     type='PointNetFPNeck',
    #     fp_channels=((1536, 512, 512), (768, 512, 512), (608, 256, 256),
    #                  (257, 128, 128))),
    rpn_head=dict(
        type='PointRPNHead',
        num_classes=3,
        enlarge_width=0.1,
        pred_layer_cfg=dict(
            in_channels=128,
            cls_linear_channels=(256, 256),
            reg_linear_channels=(256, 256)),
        cls_loss=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            reduction='sum',
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        bbox_loss=dict(
            type='mmdet.SmoothL1Loss',
            beta=1.0 / 9.0,
            reduction='sum',
            loss_weight=1.0),
        bbox_coder=dict(
            type='PointXYZWHLRBBoxCoder',
            code_size=8,
            # code_size: (center residual (3), size regression (3),
            #             torch.cos(yaw) (1), torch.sin(yaw) (1)
            use_mean_size=True,
            mean_size=[[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6,
                                                            1.73]])),
    roi_head=dict(
        type='PointRCNNRoIHead',
        bbox_roi_extractor=dict(
            type='Single3DRoIPointExtractor',
            roi_layer=dict(type='RoIPointPool3d', num_sampled_points=512)),
        bbox_head=dict(
            type='PointRCNNBboxHead',
            num_classes=1,
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss',
                beta=1.0 / 9.0,
                reduction='sum',
                loss_weight=1.0),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0),
            pred_layer_cfg=dict(
                in_channels=512,
                cls_conv_channels=(256, 256),
                reg_conv_channels=(256, 256),
                bias=True),
            in_channels=5,
            # 5 = 3 (xyz) + scores + depth
            mlp_channels=[128, 128],
            num_points=(128, 32, -1),
            radius=(0.2, 0.4, 100),
            num_samples=(16, 16, 16),
            sa_channels=((128, 128, 128), (128, 128, 256), (256, 256, 512)),
            with_corner_loss=True),
        depth_normalizer=70.0),
    # model training and testing settings
    train_cfg=dict(
        pos_distance_thr=10.0,
        rpn=dict(
            rpn_proposal=dict(
                use_rotate_nms=True,
                score_thr=None,
                iou_thr=0.8,
                nms_pre=9000,
                nms_post=512)),
        rcnn=dict(
            assigner=[
                dict(  # for Pedestrian
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1,
                    match_low_quality=False),
                dict(  # for Cyclist
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1,
                    match_low_quality=False),
                dict(  # for Car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1,
                    match_low_quality=False)
            ],
            sampler=dict(
                type='IoUNegPiecewiseSampler',
                num=128,
                pos_fraction=0.5,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True),
            cls_pos_thr=0.7,
            cls_neg_thr=0.25)),
    test_cfg=dict(
        rpn=dict(
            nms_cfg=dict(
                use_rotate_nms=True,
                iou_thr=0.85,
                nms_pre=9000,
                nms_post=512,
                score_thr=None)),
        rcnn=dict(use_rotate_nms=True, nms_thr=0.1, score_thr=0.1)))
