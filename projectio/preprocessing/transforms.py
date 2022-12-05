#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:34:08 2022

@author: taskergavi
"""

import cv2

import numpy as np

from plantcv import plantcv as pcv
from scipy.ndimage import center_of_mass

class MakeSquare:
    
    def __init__(self, transform_tracker=None):
        self.transform_tracker = transform_tracker
    
    def __call__(self, img):
        h, w = img.shape
        
        if w != h:
            if h > w:
                temp = np.zeros((h, h)).astype(img.dtype) 
                
                if self.__is_left(img):
                    temp[:, :w] = img
                    img = temp
                    
                    if self.transform_tracker is not None:
                        self.transform_tracker.append(('vert_left', 0))
        
                else:
                    delta = h - w 
                    
                    temp[:, delta:] = img
                    img = temp
                    
                    if self.transform_tracker is not None:
                        self.transform_tracker.append(('vert_right', delta))
                
            elif self.__is_left(img):
                img = img[:, :h]
                
                if self.transform_tracker is not None:
                    self.transform_tracker.append(('horiz_left', 0))
                
            else:
                delta = w - h
                
                img = img[h, delta:]
                
                if self.transform_tracker is not None:
                    self.transform_tracker.append(('horiz_right', delta))
                
        return img
                
    def __is_left(self, img):
        com = center_of_mass(img > 0)
        
        _, w = img.shape
        
        horiz_center = w // 2
        
        return com[1] < horiz_center
    
            

class ResizeImage:
    
    def __init__(self, dims, transform_tracker=None):
        self.dims = np.array(dims)
        self.transform_tracker = transform_tracker
        
    def __call__(self, img):
        if self.transform_tracker is not None:
            self.transform_tracker.append(('scale', (self.dims / img.shape)))
        
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
        
        