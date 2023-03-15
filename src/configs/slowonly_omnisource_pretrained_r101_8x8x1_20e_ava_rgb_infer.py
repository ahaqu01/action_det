# model setting
model = dict(
    backbone=dict(
        depth=101,
        # pretrained=None,
        pretrained2d=False,
        lateral=False,
        num_stages=4,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        spatial_strides=(1, 2, 2, 1)),
    roi_head=dict(
        bbox_roi_extractor=dict(
            # type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            in_channels=2048,
            num_classes=81,
            multilabel=True,
            dropout_ratio=0.5)),
    test_cfg=dict(rcnn=dict(action_thr=0.002))
)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
