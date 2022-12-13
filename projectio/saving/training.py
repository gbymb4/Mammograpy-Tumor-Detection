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
    


def save_roi_training_hist(hist_tuple, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    train_losses, test_losses, train_detect_fracs, test_detect_fracs = hist_tuple
    
    np.save(f'{save_dir}/train_losses', train_losses)
    np.save(f'{save_dir}/test_losses', test_losses)
    np.save(f'{save_dir}/train_detect_fracs', train_detect_fracs)
    np.save(f'{save_dir}/test_detect_fracs', test_detect_fracs)
    
    
    
def save_roi_results(model, hist_tuple, dataset='inbreast', model_type='region_detectors'):
    save_root_dir = 'out/{dataset.lower()}'
    
    if not os.path.isdir(save_root_dir):
        os.mkdir(save_root_dir)
        
    save_parent_dir = f'{save_root_dir}/{model_type}'
    
    if not os.path.isdir(save_parent_dir):
        os.mkdir(save_parent_dir)
        
    save_dir = f'{save_parent_dir}/{time.time()}'
    
    save_model(model, save_dir)
    save_roi_training_hist(hist_tuple, save_dir)