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
from scipy.ndimage import label
from scipy.ndimage import binary_erosion, binary_dilation

class MakeSquare:
    
    def __init__(self, transform_tracker=None, _padding_override=None):
        self.transform_tracker = transform_tracker
        
        self._padding_override = _padding_override
        
        if _padding_override is not None:
            self.calls_count = 0
    
    def __call__(self, img):
        if self._padding_override is None:
            img = self.__normal_execute(img)
            
        if self._padding_override is not None:
            img = self.__override_execute(img)
            
            self.calls_count += 1
                
        return img
        
    
    
    def __is_left(self, img):
        com = center_of_mass(img > 0)
        
        _, w = img.shape
        
        horiz_center = w // 2
        
        return com[1] < horiz_center
    
    
    
    def __normal_execute(self, img):
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
                
                img = img[:, delta:]
                
                if self.transform_tracker is not None:
                    self.transform_tracker.append(('horiz_right', delta))
        
        else:
            if self.transform_tracker is not None:
                self.transform_tracker.append(('none', 0))
                
        return img
        
    
    
    def __override_execute(self, img):
        h, w = img.shape
        
        padding_direction = self._padding_override[self.calls_count]
        
        if w != h:
            if padding_direction == 'vert_left':
                temp = np.zeros((h, h)).astype(img.dtype) 
                
                temp[:, :w] = img
                img = temp
                    
                if self.transform_tracker is not None:
                    self.transform_tracker.append(('vert_left', 0))
        
            elif padding_direction == 'vert_right':
                delta = h - w 
                
                temp = np.zeros((h, h)).astype(img.dtype) 
                    
                temp[:, delta:] = img
                img = temp
                    
                if self.transform_tracker is not None:
                    self.transform_tracker.append(('vert_right', delta))
                
            elif padding_direction == 'horiz_left':
                img = img[:, :h]
                
                if self.transform_tracker is not None:
                    self.transform_tracker.append(('horiz_left', 0))
                
            elif padding_direction == 'horiz_right':
                delta = w - h
                
                img = img[:, delta:]
                
                if self.transform_tracker is not None:
                    self.transform_tracker.append(('horiz_right', delta))
        
        else:
            if self.transform_tracker is not None:
                self.transform_tracker.append(('none', 0))
                
        return img
    
    
            
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
        return pcv.canny_edge_detect(img, *self.args, **self.kwargs)
    
    

class BreastMasking:
    
    def __init__(self, threshold=10, blur_scaling=10):
        self.threshold = threshold
        self.blur_scaling = blur_scaling
    
    def __call__(self, img):
        ksize = tuple((np.array(img.shape) / self.blur_scaling).astype(int))
        
        img_temp = cv2.blur(img, ksize)
        img_temp = cv2.normalize(img_temp, None, 0, 255, cv2.NORM_MINMAX)
        
        _, mask = cv2.threshold(img_temp, self.threshold, 255, cv2.THRESH_BINARY)
        
        img[mask == 0] = 0
        
        return img
    
    

class LargestRegionMasking:
    
    def __init__(self, blur_scaling=50, iterations=50):
        self.blur_scaling = blur_scaling
        self.iterations = iterations
    
    def __call__(self, img):
        ksize = tuple((np.array(img.shape) / self.blur_scaling).astype(int))
        
        img_temp = cv2.blur(img, ksize)
        img_temp = cv2.normalize(img_temp, None, 0, 255, cv2.NORM_MINMAX)
        
        thresholded = img_temp > 0
        
        eroded_mask = binary_erosion(
            thresholded,
            structure=np.ones((3,3)),
            iterations=self.iterations
        )
        
        labeled, num_components = label(eroded_mask)
        
        component_sizes = np.bincount(labeled.ravel())
        
        max_label = np.argmax(component_sizes)
        
        if max_label == 0:
            component_sizes[max_label] = -1
            max_label = np.argmax(component_sizes)
            
            mask = (labeled == max_label)
        
        else:
            mask = (labeled == max_label)
            
        mask = binary_dilation(
            mask,
            structure=np.ones((3,3)),
            iterations=self.iterations
        )
        
        img[mask == 0] = 0
        
        return img
    


    