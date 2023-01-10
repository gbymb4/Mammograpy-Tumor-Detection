# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:42:33 2022

@author: Gavin
"""

import torch

from torch import nn

from torchvision.models import resnet50

from .core import GeneralSSD, GeneralSSDFeatureExtractor
from .util import resnet_fmap_extractor

class SSDResNet50(GeneralSSD):
    
    def __init__(
            self,
            size,
            pretrained=True,
            trainable_backbone=False,
            **kwargs
        ):
        
        backbone = resnet50(pretrained=pretrained)
        backbone = SSDResNet50FeatureExtractor(backbone, trainable_backbone)
        
        super().__init__(backbone, size, num_feature_maps=7, **kwargs)
        


class SSDResNet50FeatureExtractor(GeneralSSDFeatureExtractor):
    
    def __init__(self, backbone, trainable_backbone=False):
        super().__init__()
        
        self.scale_weight = nn.Parameter(torch.ones(2048) * 20)
        self.features, self.extra = resnet_fmap_extractor(
            backbone,
            trainable_backbone
        )