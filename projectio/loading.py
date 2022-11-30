# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:53:48 2022

@author: Gavin
"""

import os

from pathlib import Path

import pydicom as dicom
import numpy as np

import pconfig as c

def load_dicom_mammogram(fname, transforms=None, stack_transforms=None):
    img = dicom.dcmread(fname).pixel_array
    
    name = Path(fname).name.split('_')[0]
    
    if transforms is not None:
        for transform in transforms:
            img = transform(img)
    
    if stack_transforms is not None:
        channels = []
        for transform in stack_transforms:
            channels.append(transform(img))

        img = np.array(channels)

    return img, name


def load_dicom_mammograms(dataset, load_limit=None, **kwargs):
    data_dir = None
    
    if dataset.lower() == 'inbreast':
        data_dir = f'{c.INBREAST_DIR}/AllDICOMs'
        
    img_fnames = os.listdir(data_dir)
    img_fnames = list(filter(lambda x: '.dcm' in x, img_fnames))
    img_fnames = list(map(lambda x: f'{data_dir}/' + x, img_fnames))
    
    if load_limit is not None:
        img_fnames = img_fnames[:load_limit]
    
    loaded = [load_dicom_mammogram(fname, **kwargs) for fname in img_fnames]
    
    imgs = np.array([l[0] for l in loaded])
    names = np.array([l[1] for l in loaded])
    
    return imgs, names