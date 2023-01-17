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
        
    __add_non_learnable_params_to_state_dict(model)
    
    torch.save(model.state_dict(), save_file)
    


def save_generator(gen, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    save_file = f'{save_dir}/generator'
        
    __add_non_learnable_params_to_state_dict(gen)
    
    torch.save(gen.state_dict(), save_file)
    


def save_discriminator(disc, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    save_file = f'{save_dir}/discriminator'
        
    __add_non_learnable_params_to_state_dict(disc)
    
    torch.save(disc.state_dict(), save_file)



def save_training_hist(hist_dict, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    for key, value in hist_dict.items():    
        np.save(f'{save_dir}/{key}.npy', value)
    
    
    
    
def save_roi_results(
        model, 
        hist_dict, 
        dataset='inbreast', 
        model_type='region_detectors'
    ):
    
    save_root_dir = f'out/{dataset.lower()}'
    
    if not os.path.isdir(save_root_dir):
        os.mkdir(save_root_dir)
        
    save_parent_dir = f'{save_root_dir}/{model_type}'
    
    if not os.path.isdir(save_parent_dir):
        os.mkdir(save_parent_dir)
        
    save_dir = f'{save_parent_dir}/{int(time.time())}'
    
    save_model(model, save_dir)
    save_training_hist(hist_dict, save_dir)
    


def save_segmentation_results(
        gen, 
        disc, 
        hist_dict, 
        dataset='inbreast', 
        model_type='segmenters'
    ):
    
    save_root_dir = f'out/{dataset.lower()}'

    if not os.path.isdir(save_root_dir):
        os.mkdir(save_root_dir)
        
    save_parent_dir = f'{save_root_dir}/{model_type}'
    
    if not os.path.isdir(save_parent_dir):
        os.mkdir(save_parent_dir)
    
    save_dir = f'{save_parent_dir}/{int(time.time())}'
    
    save_generator(gen, save_dir)
    save_discriminator(disc, save_dir)
    save_training_hist(hist_dict, save_dir)
    
    
    
def __add_non_learnable_params_to_state_dict(model):
    non_learnable_params = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            non_learnable_params[name] = param.data
            
    model.state_dict().update(non_learnable_params)
    
    return model.state_dict()
