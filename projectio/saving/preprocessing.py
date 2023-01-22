# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 22:00:39 2022

@author: Gavin
"""

import os, json

import pconfig as c
import numpy as np

def save_preprocessed_images(imgs, names, dataset, path_suffix=None):
    data_dir = None
    
    if dataset.lower() == 'inbreast':
        data_dir = c.PREPROCESSED_INBREAST_DIR
        
    elif dataset.lower() == 'mias':
        data_dir = c.PREPROCESSED_MIAS_DIR
        
    elif dataset.lower() == 'ddsm':
        data_dir = c.PREPROCESSED_DDSM_DIR
       
    if path_suffix is None:
        save_imgs_dir = f'{data_dir}/imgs' 
    else:
        save_imgs_dir = f'{data_dir}/imgs{path_suffix}' 
        
    if not os.path.exists(save_imgs_dir):
        os.mkdir(save_imgs_dir)
        
    for img, name in zip(imgs, names):
        np.save(f'{save_imgs_dir}/{name}', img)



def save_rois(coords, dataset, fname_suffix=None):
    data_dir = None
    
    if dataset.lower() == 'inbreast':
        data_dir = c.PREPROCESSED_INBREAST_DIR
        
    elif dataset.lower() == 'mias':
        data_dir = c.PREPROCESSED_MIAS_DIR
        
    elif dataset.lower() == 'ddsm':
        data_dir = c.PREPROCESSED_DDSM_DIR
        
    save_rois_dir = f'{data_dir}/rois' 
    
    if not os.path.exists(save_rois_dir):
        os.mkdir(save_rois_dir)
    
    if fname_suffix is None:
        fname = f'{save_rois_dir}/rois.json'
    else:
        fname = f'{save_rois_dir}/rois{fname_suffix}.json'
        
    with open(fname, 'w') as file:
        json.dump(coords, file)
        
        file.close()
        


def save_bboxes(bboxes, dataset, fname_suffix=None):
    data_dir = None
    
    if dataset.lower() == 'inbreast':
        data_dir = c.PREPROCESSED_INBREAST_DIR
        
    elif dataset.lower() == 'mias':
        data_dir = c.PREPROCESSED_MIAS_DIR
        
    elif dataset.lower() == 'ddsm':
        data_dir = c.PREPROCESSED_DDSM_DIR
        
    save_rois_dir = f'{data_dir}/bboxes' 
    
    if not os.path.exists(save_rois_dir):
        os.mkdir(save_rois_dir)
        
    if fname_suffix is None:   
        fname = f'{save_rois_dir}/bboxes.json'
    else:
        fname = f'{save_rois_dir}/bboxes{fname_suffix}.json'
        
    with open(fname, 'w') as file:
        json.dump(bboxes, file)
        
        file.close()