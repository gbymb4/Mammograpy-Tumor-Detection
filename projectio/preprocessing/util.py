# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 21:40:49 2022

@author: Gavin
"""

import numpy as np

def adapt_transformed_coords(
        img_names, 
        coord_names,
        coords,
        recenter_tracker=None,
        rescale_tracker=None,
    ):
    img_order = np.where(np.in1d(img_names, coord_names))
    
    coord_names = coord_names[img_order]
    coords = coords[img_order]
    
    if recenter_tracker is not None:
        for roi_coords, recenter_record in zip(coords, recenter_tracker):
            action, recenter_coef = recenter_record
            
            for tup in roi_coords:
                _, points = tup
                
                if action == 'vert_left':
                    continue
                    
                elif action == 'vert_right':
                    points[:, 1] = points[:, 1] + recenter_coef
                    
                elif action == 'horiz_left': 
                    continue
                
                elif action == 'horiz_right':
                    points[:, 1] = points[:, 1] - recenter_coef
        
    if rescale_tracker is not None:
        for roi_coords, rescale_record in zip(coords, rescale_tracker):
            _, rescale_coefs = rescale_record
            
            for tup in roi_coords:
                _, points = tup
                
                points[:, 0] = points[:, 0] * rescale_coefs[0]
                points[:, 1] = points[:, 1] * rescale_coefs[1]
            
    return coord_names, coords
