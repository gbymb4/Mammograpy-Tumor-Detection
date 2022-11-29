# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 17:47:43 2022

@author: Gavin
"""

import torch

from torch import nn

from torchvision.models import mobilenet_v3_large

from .core import GeneralSSD, GeneralSSDFeatureExtractor
from .util import mobilenet_fmap_extractor

class SSDMobileNetV3(GeneralSSD):
    
    def __init__(self, size, pretrained=True, **kwargs):
        backbone = mobilenet_v3_large(pretrained=pretrained)
        backbone = SSDMobileNetV3FeatureExtractor(backbone)
        
        super().__init__(backbone, size, num_feature_maps=7, **kwargs)
        


class SSDMobileNetV3FeatureExtractor(GeneralSSDFeatureExtractor):
    
    def __init__(self, backbone):
        super().__init__()
        
        self.scale_weight = nn.Parameter(torch.ones(960) * 20)
        self.features, self.extra = mobilenet_fmap_extractor(backbone)