# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:39:24 2022

@author: Gavin
"""

from abc import ABC, abstractmethod
from collections import OrderedDict

from torch import nn

import torch.nn.functional as F

from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator

def bounding_box_generator(num_feature_maps, aspect_ratios=None, **kwargs):
    if aspect_ratios is None:
        aspect_ratios = [[2]] * num_feature_maps
    
    boxes = DefaultBoxGenerator(aspect_ratios, **kwargs)
    
    return boxes



class GeneralSSD(SSD, ABC):
    
    def __init__(
            self, backbone, size, 
            anchor_generator=None,
            num_feature_maps=None,
            **kwargs
        ):
        if num_feature_maps is None:
            raise ValueError(f'num_feature_maps not assigned value in {type(self)} implementation')
        
        self.num_feature_maps = num_feature_maps
        
        if anchor_generator is None:
            anchor_generator = bounding_box_generator(self.num_feature_maps)
        
        super().__init__(backbone, anchor_generator, size, 2, **kwargs) 
    
    
    
class GeneralSSDFeatureExtractor(nn.Module, ABC):
    
    def forward(self, x):
        x = self.features(x)
        rescaled = self.scale_weight.view(1, -1, 1, 1) * F.normalize(x)
        output = [rescaled]

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])