# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 13:42:04 2023

@author: Gavin
"""

import torch, random

import numpy as np

from shapely.geometry import Polygon
from scipy.ndimage import zoom
from PIL import Image, ImageDraw

def format_segmentation_rois(rois, fuzzy_bbox_func=None):
    rois = np.array(list(zip(*rois)))
    
    formatted = np.apply_along_axis(
        format_segmentation_roi,
        1, 
        rois, 
        fuzzy_bbox_func
    )
    
    formatted = torch.from_numpy(formatted).float()
    
    return formatted
    


def format_segmentation_roi(roi, fuzzy_bbox_func, device='cpu'):
    bbox, coords, img = roi
    
    img = img.detach().cpu().numpy()
    
    if fuzzy_bbox_func is not None:
        bbox = fuzzy_bbox_func(bbox)
    
    shape = img.shape
    
    poly = Polygon(coords)
    
    temp_img = Image.new("1", shape[1:], 0)
    ImageDraw.Draw(temp_img).polygon(poly.exterior.coords, outline=1, fill=1)
    
    mask = np.array(temp_img).T[np.newaxis, :, :]
    
    x_min, y_min, x_max, y_max = [int(i) for i in bbox]
    
    img_roi = img[:, x_min:x_max, y_min:y_max]
    mask_roi = mask[:, x_min:x_max, y_min:y_max]
    
    new_shape = (shape[0], 256, 256)
    zoom_factor = np.divide(new_shape, img_roi.shape)
    
    img_roi = zoom(img_roi, zoom=zoom_factor, order=1)
    mask_roi = zoom(mask_roi, zoom=zoom_factor, order=1)
    
    return img_roi, mask_roi



class FuzzyBoundingBoxes:
    
    def __init__(
            self,
            scale_tolerance=0.3,
            x_tolerance=10,
            y_tolerance=10
        ):
        
        self.scale_tolerance = scale_tolerance
        self.x_tolerance = x_tolerance
        self.y_tolerance = y_tolerance
    
    def __call__(self, bbox):
        new_scale = 1 + random.uniform(0, self.scale_tolerance)
        
        bbox = self.__scale_bbox(bbox, new_scale)
        
        x_shift = random.uniform(-self.x_tolerance, self.x_tolerance)
        y_shift = random.uniform(-self.y_tolerance, self.y_tolerance)
        
        bbox = self.__translate_bbox(bbox, x_shift, y_shift)
        
        return bbox
        


    def __scale_bbox(self, bbox, scale):
        x_min, y_min, x_max, y_max = bbox
        
        width = x_max - x_min
        height = y_max - y_min
        
        center_x, center_y = x_min + width / 2, y_min + height / 2
        
        width *= scale
        height *= scale
        
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = x_min + width
        y_max = y_min + height
        
        return x_min, y_min, x_max, y_max
    
    
    
    def __translate_bbox(self, bbox, x_shift, y_shift):
        x_min, y_min, x_max, y_max = bbox
        
        x_min += x_shift
        x_max += x_shift
        
        y_min += y_shift
        y_max += y_shift
        
        return x_min, y_min, x_max, y_max