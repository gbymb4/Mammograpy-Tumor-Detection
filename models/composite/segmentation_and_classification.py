# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:30:42 2023

@author: Gavin
"""

import torch

import numpy as np

from torch import nn

from skimage.measure import label, regionprops
from scipy.ndimage.morphology import binary_fill_holes

class CompositeClassifier(nn.Module):
    
    def __init__(self, segmenter, classifier):
        super().__init__()
        
        self.segmenter = segmenter
        self.classifier = classifier
        
        
        
    def forward(self, x):
        raw_masks = self.segmenter(x)
        
        masks = self.__preprocess_masks(raw_masks)
        
        classifier_x = torch.cat((x, masks), 1)
        
        out = self.classifier(classifier_x)
        
        return out
    
    
        
    def __preprocess_masks(self, raw):
        device = raw.device
        
        masks = raw.cpu().detach().numpy() > 0.5
        
        preprocessed = self.__filter_and_fill(masks)
        preprocessed = torch.from_numpy(preprocessed).float().to(device)
        
        return preprocessed
    
    
        
    def __filter_and_fill(self, masks):
        labels = label(masks.squeeze(), background=0)
        
        regions = regionprops(labels)
        max_area = 0
        max_region = None
        for region in regions:
            
            if region.area > max_area and region.label != 0:
                max_area = region.area
                max_region = region
                
        if max_region is None:
            return np.zeros_like(masks)
        
        filled_mask = binary_fill_holes(labels == max_region.label)
        
        output_mask = np.expand_dims((filled_mask > 0).astype(float), axis=1)
        
        return output_mask