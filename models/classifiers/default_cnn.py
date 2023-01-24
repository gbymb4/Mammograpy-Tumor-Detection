# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 19:47:57 2023

@author: Gavin
"""

from torch import nn

class CNNClassifier(nn.Module):
    
    # in_channels is a parameter in-case the mask is concatenated to the input tensor
    def __init__(self, in_channels):
        super().__init__()
        
        cn1 = nn.Conv2d(in_channels, 64, (9, 9), (1, 1), (4, 4))
        p1 = nn.MaxPool2d((4, 4), (4, 4))
        
        cn2 = nn.Conv2d(64, 128, (5, 5), (1, 1), (2, 2))
        p2 = nn.MaxPool2d((4, 4), (4, 4))
        
        cn3 = nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1))
        p3 = nn.MaxPool2d((4, 4), (4, 4))
        
        cn4 = nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1))
        p4 = nn.MaxPool2d((4, 4), (4, 4))
        
        fl = nn.Flatten()
        
        fc3 = nn.Linear(512, 256)
        
        fc4 = nn.Linear(256, 128)
        
        fc5 = nn.Linear(128, 128)
        
        fcout = nn.Linear(128, 1)
        
        sig = nn.Sigmoid()
        
        self.features = nn.Sequential(
            cn1, p1,
            cn2, p2,
            cn3, p3,
            cn4, p4,
            fl,
            fc3,
            fc4,
            fc5,
            fcout,
            sig
        )
        
    
    
    def forward(self, x):
        return self.features(x)
    