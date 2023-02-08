# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:34:05 2023

@author: Gavin
"""

from torch import nn

from torchvision.models import (
    resnet50,
    vgg16,
    efficientnet_v2_s, 
    mobilenet_v3_large
)



class ResNet50Classifier(nn.Module):
    
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        
        cn = nn.Conv2d(in_channels, 3, (1, 1))
        
        fc1 = nn.Linear(1000, 1000)
        a1 = nn.ReLU()
        do1 = nn.Dropout()
        
        fc2 = nn.Linear(1000, 1000)
        a2 = nn.ReLU()
        do2 = nn.Dropout()
        
        fcout = nn.Linear(1000, 1)
        aout = nn.Sigmoid()
        
        self.features = nn.Sequential(
            cn,
            resnet50(**kwargs),
            fc1, a1, do1,
            fc2, a2, do2,
            fcout, aout
        )
        
    def forward(self, x):
        return self.features(x)
    
    

class VGG16Classifier(nn.Module):
    
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        
        cn = nn.Conv2d(in_channels, 3, (1, 1))
        
        fcout = nn.Linear(1000, 1)
        aout = nn.Sigmoid()
        
        self.features = nn.Sequential(
            cn,
            vgg16(**kwargs),
            fcout, aout
        )
        
    def forward(self, x):
        return self.features(x)
    
    

class EfficientNetV2Classifier(nn.Module):
    
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        
        cn = nn.Conv2d(in_channels, 3, (1, 1))
        
        fc1 = nn.Linear(1000, 1000)
        a1 = nn.ReLU()
        do1 = nn.Dropout()
        
        fc2 = nn.Linear(1000, 1000)
        a2 = nn.ReLU()
        do2 = nn.Dropout()
        
        fcout = nn.Linear(1000, 1)
        aout = nn.Sigmoid()
        
        self.features = nn.Sequential(
            cn,
            efficientnet_v2_s(**kwargs),
            fc1, a1, do1,
            fc2, a2, do2,
            fcout, aout
        )
        
    def forward(self, x):
        return self.features(x)
    
    
    
class MobileNetV3Classifier(nn.Module):
    
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        
        cn = nn.Conv2d(in_channels, 3, (1, 1))
        
        fc1 = nn.Linear(1000, 1000)
        a1 = nn.ReLU()
        do1 = nn.Dropout()
        
        fcout = nn.Linear(1000, 1)
        aout = nn.Sigmoid()
        
        self.features = nn.Sequential(
            cn,
            mobilenet_v3_large(**kwargs),
            fc1, a1, do1,
            fcout, aout
        )
        
    def forward(self, x):
        return self.features(x)
