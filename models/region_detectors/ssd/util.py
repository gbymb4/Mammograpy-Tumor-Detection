# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:39:29 2022

@author: Gavin
"""

import torch

from torch import nn

def _xavier_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)
           
                
           
def _fmap_base(highres=True):
    extra = nn.ModuleList(
        [
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),  
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),  
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3),  
                nn.ReLU(inplace=True),
            )
        ]
    )
    
    if highres:
        extra.insert(
            len(extra),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=4),  
                nn.ReLU(inplace=True),
            )
        )
    
    _xavier_init(extra)
    
    return extra



def vgg_fmap_extractor(backbone):
    backbone = backbone.features
    
    # below implementation is from torchvision.models.detection source
    # the structure has simply been adapted to be consistent with other implemented SDD backbones
    
    _, _, maxpool3_pos, maxpool4_pos, _ = (i for i, layer in enumerate(backbone) if isinstance(layer, nn.MaxPool2d))

    backbone[maxpool3_pos].ceil_mode = True

    features = nn.Sequential(*backbone[:maxpool4_pos])  

    for module in features:
        for param in module.parameters():
            param.requires_grad = False

    extra = _fmap_base()

    fc = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),  
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6),  
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
        nn.ReLU(inplace=True),
    )
    
    _xavier_init(fc)
    
    extra.insert(
        0,
        nn.Sequential(
            *backbone[maxpool4_pos:-1],  
            fc,
        ),
    )
    
    return features, extra
    


def resnet_fmap_extractor(backbone):
    backbone = backbone._modules
    
    feature_keys = list(backbone.keys())
    cutoff = feature_keys.index('avgpool')
    feature_keys = feature_keys[:cutoff]
    
    features = nn.Sequential(*[backbone[key] for key in feature_keys])
    
    final_bottleneck = features[-1][0]
    final_bottleneck.conv2.stride = (1, 1)
    final_bottleneck.downsample[0].stride = (1, 1)
    
    for module in features:
        for param in module.parameters():
            param.requires_grad = False
    
    extra = _fmap_base()
    
    fc = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),  
        nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1),  
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    
    _xavier_init(fc)
    
    extra.insert(0, fc)
    
    return features, extra
    


def efficientnet_fmap_extractor(backbone):
    features = backbone._modules['features']
    
    for module in features:
        for param in module.parameters():
            param.requires_grad = False
            
    extra = _fmap_base()
    
    fc = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),  
        nn.Conv2d(in_channels=1280, out_channels=512, kernel_size=1),  
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    
    _xavier_init(fc)
    
    extra.insert(0, fc)
    
    extra[1][2].stride = (1, 1)
    
    return features, extra



def mobilenet_fmap_extractor(backbone):
    features = backbone._modules['features']
    
    for module in features:
        for param in module.parameters():
            param.requires_grad = False
            
    extra = _fmap_base()
    
    fc = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),  
        nn.Conv2d(in_channels=960, out_channels=512, kernel_size=1),  
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    
    _xavier_init(fc)
    
    extra.insert(0, fc)
    
    extra[1][2].stride = (1, 1)
    
    return features, extra
    