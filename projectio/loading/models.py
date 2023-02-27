# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:55:31 2023

@author: Gavin
"""

import torch

def load_classifier(
        dataset,
        name, 
        classifier_type,
        map_location='cuda:0',
        *args,
        **kwargs
    ):
    
    model_dir = f'out/{dataset}/classifiers/{name}/model'
    
    model = classifier_type(*args, **kwargs)
    model.load_state_dict(torch.load(model_dir, map_location=map_location))
    model.eval()
    
    return model



def load_segmenter(
        dataset,
        name, 
        segmenter_type, 
        map_location='cuda:0',
        *args,
        **kwargs
    ):
    
    model_dir = f'out/{dataset}/segmenters/{name}/generator'
    
    model = segmenter_type(*args, **kwargs)
    model.load_state_dict(torch.load(model_dir, map_location=map_location))
    model.eval()
    
    return model