# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:53:54 2022

@author: Gavin
"""

import os

import pconfig as c
import numpy as np

def save_preprocessed_images(imgs, names, dataset):
    data_dir = None
    
    if dataset.lower() == 'inbreast':
        data_dir = c.PREPROCESSED_INBREAST_DIR
       
    save_imgs_dir = f'{data_dir}/imgs' 
       
    if not os.path.exists(save_imgs_dir):
        os.mkdir(save_imgs_dir)
        
    for img, name in zip(imgs, names):
        np.save(f'{save_imgs_dir}/{name}', img)
    