# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:45:00 2022

@author: Gavin
"""

import torch

from torch import nn

from torchvision.models import efficientnet_v2_s

from .core import GeneralSSD, GeneralSSDFeatureExtractor
from .util import efficientnet_fmap_extractor

class SSDEfficientNetV2(GeneralSSD):
    
    def __init__(self, size, pretrained=True, **kwargs):
        backbone = efficientnet_v2_s(pretrained=pretrained)
        backbone = SSDEfficientNetV2FeatureExtractor(backbone)
        
        super().__init__(backbone, size, num_feature_maps=7, **kwargs)
        


class SSDEfficientNetV2FeatureExtractor(GeneralSSDFeatureExtractor):
    
    def __init__(self, backbone):
        super().__init__()
        
        self.scale_weight = nn.Parameter(torch.ones(1280) * 20)
        self.features, self.extra = efficientnet_fmap_extractor(backbone)