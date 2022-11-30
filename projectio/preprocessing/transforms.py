#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:34:08 2022

@author: taskergavi
"""

import cv2

from plantcv import plantcv as pcv

class ResizeImage:
    
    def __init__(self, dims):
        self.dims = dims
        
    def __call__(self, img):
        return cv2.resize(img, self.dims)
    


class CLAHE:
    
    def __init__(self, *args, **kwargs):
        self.clahe = cv2.createCLAHE(*args, **kwargs)
        
    def __call__(self, img):
        return self.clahe.apply(img)
    
    
    
class CannyEdges:
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
    def __call__(self, img):
        img = img * 255
        
        return pcv.canny_edge_detect(img, *self.args, **self.kwargs)
        
        