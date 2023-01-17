# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:19:33 2023

@author: Gavin
"""

import numpy as np

def batch_mask_ious(batch1, batch2):
    batch1 = batch1.detach().cpu().numpy()
    batch2 = batch2.detach().cpu().numpy()
    
    """
    Computes the IoU scores between same-index pairs of binary mask images for two batches.
    Args:
        batch1 (torch.Tensor): A Tensor of shape (batch_size, 1, H, W) representing the first batch of binary mask images.
        batch2 (torch.Tensor): A Tensor of shape (batch_size, 1, H, W) representing the second batch of binary mask images.
    Returns:
        scores (np.ndarray): A numpy array of shape (batch_size,) containing the IoU scores between the same-index pairs of binary mask images.
    """
    intersection = np.sum(np.logical_and(batch1, batch2), axis=(1,2,3))
    union = np.sum(np.logical_or(batch1, batch2), axis=(1,2,3))
    
    iou = intersection / union
    
    return iou