import torch
import torch.nn as nn
from .backbones.resnet3d_slowonly import ResNet3dSlowOnly
from .roi_extractors.single_straight3d import SingleRoIExtractor3D
from .heads.bbox_head import BBoxHeadAVA


class sta_model(nn.Module):
    """Wrapper for MMAction2 spatio-temporal action models.

    Args:
        config (object): stdet config.
    """

    def __init__(self, config):
        super().__init__()
        self.backbone_cfg = config.model["backbone"]
        self.bbox_roi_extractor_cfg = config.model["roi_head"]["bbox_roi_extractor"]
        self.bbox_head_cfg = config.model["roi_head"]["bbox_head"]

        self.backbone = ResNet3dSlowOnly(**self.backbone_cfg)
        self.bbox_roi_extractor = SingleRoIExtractor3D(**self.bbox_roi_extractor_cfg)
        self.bbox_head = BBoxHeadAVA(**self.bbox_head_cfg)

    def forward(self, img, proposals):
        feat = self.backbone(img)
        roi_feats, _ = self.bbox_roi_extractor(feat, proposals)
        logit, _ = self.bbox_head(roi_feats)
        out = torch.sigmoid(logit)
        return out
