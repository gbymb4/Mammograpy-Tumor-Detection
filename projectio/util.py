# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:57:38 2023

@author: Gavin
"""

import torch

def swap_elements(x, device='cpu'):
    i = torch.tensor([0, 2]).repeat(x.shape[0], 1).to(device)
    j = torch.tensor([1, 3]).repeat(x.shape[0], 1).to(device)
    
    i = i.view(-1)
    j = j.view(-1)
    
    result = x.clone()
    
    result.index_copy_(1, j, x[:, i])
    result.index_copy_(1, i, x[:, j])
    
    return result



def scale_bboxes(bboxes, scale=0.5, limit=512):
    scale = torch.ones(len(bboxes)).to(bboxes.device) * scale
    
    bboxes = bboxes.clone()
    
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    
    width, height = x_max - x_min, y_max - y_min
    
    width = width * scale
    height = height * scale
    
    x_min = x_min - width / 2
    y_min = y_min - height / 2
    x_max = x_max + width / 2
    y_max = y_max + height / 2

    x_min[x_min < 0] = 0
    y_min[y_min < 0] = 0
    x_max[x_max > limit] = limit
    y_max[y_max > limit] = limit

    scale_x = (x_max - x_min) / (x_max.clone() - x_min.clone())
    scale_y = (y_max - y_min) / (y_max.clone() - y_min.clone())
    scale = torch.min(scale, torch.min(scale_x, scale_y))
    
    width *= scale
    height *= scale
    
    x_min = x_min - width / 2
    y_min = y_min - height / 2
    x_max = x_max + width / 2
    y_max = y_max + height / 2
    
    ans = torch.stack((x_min, y_min, x_max, y_max), dim=1)
    
    return ans