# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 21:40:49 2022

@author: Gavin
"""

import numpy as np

from skimage.measure import label, regionprops

def adapt_transformed_coords(
        rois,
        recenter_tracker=None,
        rescale_tracker=None,
    ):

    if recenter_tracker is not None:
        for roi_data, recenter_record in zip(rois.values(), recenter_tracker):
            action, recenter_coef = recenter_record
            
            new_roi_points = []
            for roi_points in roi_data['coords']:    
                
                if action == 'vert_left':
                    new_roi_points.append(roi_points)
                    
                elif action == 'vert_right':
                    recentered = list(
                        map(
                            lambda x: (
                                x[0] + recenter_coef,
                                x[1] 
                            ), 
                            roi_points
                        )
                    )
                    
                    new_roi_points.append(recentered)
                    
                elif action == 'horiz_left': 
                    new_roi_points.append(roi_points)
                
                elif action == 'horiz_right':
                    recentered = list(
                        map(
                            lambda x: (
                                x[0] - recenter_coef,
                                x[1]
                            ), 
                            roi_points
                        )
                    )
                    
                    new_roi_points.append(recentered)
                    
            roi_data['coords'] = new_roi_points
            
    if rescale_tracker is not None:
        for roi_data, rescale_record in zip(rois.values(), rescale_tracker):
            _, rescale_coefs = rescale_record
            
            new_roi_points = []
            for roi_points in roi_data['coords']:
                
                rescaled = list(
                    map(
                        lambda x: (
                            x[0] * rescale_coefs[0],
                            x[1] * rescale_coefs[1]
                        ),
                        roi_points
                    )
                )
                
                new_roi_points.append(rescaled)
        
            roi_data['coords'] = new_roi_points
            
    return rois



def adapt_transformed_bboxes(
        rois,
        recenter_tracker=None,
        rescale_tracker=None,
    ):
    
    if recenter_tracker is not None:
        
        for roi_data, recenter_record in zip(rois.values(), recenter_tracker):
            action, recenter_coef = recenter_record
            
            new_roi_points = []
            for roi_points in roi_data['bboxes']:    
                
                if action == 'vert_left':
                    new_roi_points.append(roi_points)
                    
                elif action == 'vert_right':
                    recentered = [
                        roi_points[0] + recenter_coef,
                        roi_points[1],
                        roi_points[2] + recenter_coef,
                        roi_points[3]
                    ]
                    
                    new_roi_points.append(recentered)
                    
                elif action == 'horiz_left': 
                    new_roi_points.append(roi_points)
                
                elif action == 'horiz_right':
                    recentered = [
                        roi_points[0] - recenter_coef,
                        roi_points[1],
                        roi_points[2] - recenter_coef,
                        roi_points[3]
                    ]
                    
                    new_roi_points.append(recentered)
                    
                elif action == 'none':
                    new_roi_points.append(roi_points)
                    
            roi_data['bboxes'] = new_roi_points
            
    if rescale_tracker is not None:
        for roi_data, rescale_record in zip(rois.values(), rescale_tracker):
            _, rescale_coefs = rescale_record
            
            new_roi_points = []
            for roi_points in roi_data['bboxes']:
                
                rescaled = [
                    roi_points[0] * rescale_coefs[0],
                    roi_points[1] * rescale_coefs[1],
                    roi_points[2] * rescale_coefs[0],
                    roi_points[3] * rescale_coefs[1]
                ]
                
                new_roi_points.append(rescaled)
        
            roi_data['bboxes'] = new_roi_points
            
    return rois



def compute_bounding_boxes(rois):
    for roi_data in rois.values():
        
        bboxes = []
        for roi_points in roi_data['coords']:
            roi_points = np.array(roi_points)
            
            h_max, h_min = np.max(roi_points[:, 0]), np.min(roi_points[:, 0])
            w_max, w_min = np.max(roi_points[:, 1]), np.min(roi_points[:, 1])
        
            delta_h = h_max - h_min
            delta_w = w_max - w_min
            
            if delta_h > delta_w:
                widen = (delta_h - delta_w) / 2
                
                x1, x2 = w_min - widen, w_max + widen
                y1, y2 = h_min, h_max
                
            elif delta_h < delta_w:
                heighten = (delta_w - delta_h) / 2
                
                x1, x2 = w_min, w_max
                y1, y2 = h_min - heighten, h_max + heighten
                
            elif delta_h == delta_w:
                x1, x2 = w_min, w_max
                y1, y2 = h_min, h_max
                
            # x and y are backwards - the easiest fix is just flipping them
            bboxes.append([y1, x1, y2, x2])
            
        roi_data['bboxes'] = bboxes
        
    return rois
        
        
    
def infer_bounding_boxes(names, centers, radii, rotate_tracker):
    ans = {name: {'bboxes': []} for name in names}
    
    for name, center, radius in zip(names, centers, radii):
        y, x = center
        
        rotate_record = rotate_tracker[name]
        
        if rotate_record[0] == 'left':
            y = rotate_record[1][1] - y
            x = rotate_record[1][0] - x

        x_min, x_max = x - radius, x + radius
        y_min, y_max = y - radius, y + radius
        
        bbox = [x_min, y_min, x_max, y_max]
        
        ans[name]['bboxes'].append(bbox)
        
    return ans



def compute_coords_of_mask(mask):
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    
    largest_region = max(regions, key=lambda x: x.area)
    
    coords = np.column_stack([largest_region.coords[:,1], largest_region.coords[:,0]])
    return coords