# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:19:33 2023

@author: Gavin
"""

import torch.nn.functional as F

def batch_mask_ious(real, fake):
    real = real.squeeze(1)
    fake = fake.squeeze(1)
    
    intersection = F.intersection(real, fake)
    
    union = F.union(real, fake)
    
    iou = intersection / union
    iou = iou.detach().cpu().numpy()
    
    return iou