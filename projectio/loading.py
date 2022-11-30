# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:53:48 2022

@author: Gavin
"""

import os

import pydicom as dicom
import numpy as np

import pconfig as c

def load_dicom_mammogram(fname, transforms=None, stack_transforms=None):
    img = dicom.dcmread(fname).pixel_array
    
    if transforms is not None:
        for transform in transforms:
            img = transform(img)
    
    if stack_transforms is not None:
        channels = [img]
        for transform in stack_transforms:
            channels.append(transform(img))

        img = np.array(channels)

    return img


def load_dicom_mammograms(dataset, load_limit=None, **kwargs):
    if dataset.lower() == 'inbreast':
        data_dir = f'{c.INBREAST_DIR}/AllDICOMs'
        
    img_fnames = os.listdir(data_dir)
    img_fnames = list(filter(lambda x: '.dcm' in x, img_fnames))
    img_fnames = list(map(lambda x: f'{data_dir}/' + x, img_fnames))
    
    if load_limit is not None:
        img_fnames = img_fnames[:load_limit]
        
    return np.array([load_dicom_mammogram(fname, **kwargs) for fname in img_fnames])