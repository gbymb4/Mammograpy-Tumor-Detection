# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 18:30:23 2022

@author: Gavin
"""

import json

import pconfig as c

def load_bboxes(dataset):
    data_dir = None
    
    if dataset.lower() == 'inbreast':
        data_dir = f'{c.PREPROCESSED_INBREAST_DIR}/rois'
    
    fname = f'{data_dir}/rois.json'
        
    with open(fname, 'r') as file:
        loaded = json.load(file)
        
        file.close()
        
    return loaded