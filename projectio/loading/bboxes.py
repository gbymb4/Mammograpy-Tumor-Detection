# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 18:30:23 2022

@author: Gavin
"""

import json

import pconfig as c

def load_bboxes(dataset, fname_suffix=None, load_limit=None):
    data_dir = None
    
    if dataset.lower() == 'inbreast':
        data_dir = f'{c.PREPROCESSED_INBREAST_DIR}/rois'
    
    if fname_suffix is None:   
        fname = f'{data_dir}/rois.json'
    else:
        fname = f'{data_dir}/rois{fname_suffix}.json'
        
    with open(fname, 'r') as file:
        loaded = json.load(file)
        
        file.close()
      
    if load_limit is not None:
        keys = loaded.keys()
        keys = list(keys)[:load_limit]
        
        loaded = {key: loaded[key] for key in keys}
        
    return loaded