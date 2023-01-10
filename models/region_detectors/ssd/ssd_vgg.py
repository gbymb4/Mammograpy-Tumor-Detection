# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:42:58 2022

@author: Gavin
"""

import torch

from torch import nn

from torchvision.models import vgg16

from .core import GeneralSSD, GeneralSSDFeatureExtractor
from .util import vgg_fmap_extractor

class SSDVGG16(GeneralSSD):
    
    def __init__(
            self,
            size,
            pretrained=True,
            trainable_backbone=False,
            **kwargs
        ):
        
        backbone = vgg16(pretrained=pretrained)
        backbone = SSDVGG16FeatureExtractor(backbone, trainable_backbone)
        
        super().__init__(backbone, size, num_feature_maps=7, **kwargs)
        


class SSDVGG16FeatureExtractor(GeneralSSDFeatureExtractor):
    
    def __init__(self, backbone, trainable_backbone=False):
        super().__init__()
        
        self.scale_weight = nn.Parameter(torch.ones(512) * 20)
        self.features, self.extra = vgg_fmap_extractor(
            backbone, 
            trainable_backbone
        )