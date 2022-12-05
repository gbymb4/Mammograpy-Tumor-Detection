# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 21:40:49 2022

@author: Gavin
"""

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
                                x[0],
                                x[1] + recenter_coef
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
                                x[0],
                                x[1] - recenter_coef
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