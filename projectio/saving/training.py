# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 22:02:12 2022

@author: Gavin
"""

import os, torch, time

import numpy as np

def save_model(model, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    save_file = f'{save_dir}/model'
        
    torch.save(model.state_dict(), save_file)
    


def save_roi_training_hist(hist_dict, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    for key, value in hist_dict.items():    
        np.save(f'{save_dir}/{key}.npy', value)
    
    
    
    
def save_roi_results(model, hist_dict, dataset='inbreast', model_type='region_detectors'):
    save_root_dir = f'out/{dataset.lower()}'
    
    if not os.path.isdir(save_root_dir):
        os.mkdir(save_root_dir)
        
    save_parent_dir = f'{save_root_dir}/{model_type}'
    
    if not os.path.isdir(save_parent_dir):
        os.mkdir(save_parent_dir)
        
    save_dir = f'{save_parent_dir}/{time.time()}'
    
    save_model(model, save_dir)
    save_roi_training_hist(hist_dict, save_dir)