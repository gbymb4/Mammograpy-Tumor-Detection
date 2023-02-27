# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:53:48 2022

@author: Gavin
"""

import os, cv2

from pathlib import Path

import pydicom as dicom
import numpy as np

import pconfig as c

def load_dicom_mammogram(fname, transforms=None, stack_transforms=None):
    img = dicom.dcmread(fname).pixel_array
    img = cv2.normalize(img,  None, 0, 255, cv2.NORM_MINMAX)
    
    name = Path(fname).name.split('_')[0]
    
    if transforms is not None:
        for transform in transforms:
            img = transform(img)
    
    if stack_transforms is not None:
        channels = [img]
        
        for transform in stack_transforms:
            transformed = transform(img)
            transformed = cv2.normalize(transformed,  None, 0, 255, cv2.NORM_MINMAX)
            
            channels.append(transformed)

        img = np.array(channels)
        img = np.swapaxes(img, 1, 2)
    
    return img, name



def load_dicom_mammograms(dataset, load_limit=None, **kwargs):
    data_dir = None
    
    if dataset.lower() == 'inbreast':
        data_dir = f'{c.INBREAST_DIR}/AllDICOMs'
        
    img_fnames = os.listdir(data_dir)
    img_fnames = list(filter(lambda x: '.dcm' in x, img_fnames))
    img_fnames = list(map(lambda x: f'{data_dir}/' + x, img_fnames))
    img_fnames = sorted(img_fnames, key=lambda x: int(Path(x).name.split('_')[0]))
    
    if load_limit is not None:
        img_fnames = img_fnames[load_limit[0]:load_limit[1]]
    
    loaded = [load_dicom_mammogram(fname, **kwargs) for fname in img_fnames]
    
    imgs = np.array([l[0] for l in loaded])
    names = np.array([l[1] for l in loaded])
    
    return imgs, names



def load_pgm_mammogram(
        fname,
        is_mias=False,
        transforms=None,
        stack_transforms=None,
        rotate_tracker=None
    ):
    img = cv2.imread(fname, -1)
    img = cv2.normalize(img,  None, 0, 255, cv2.NORM_MINMAX)
    
    if is_mias:
        left = fname[-6] == 'l'
        
        key = os.path.basename(fname)[:-4]
        
        img_shape = img.shape
        if left:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            if rotate_tracker is not None:
                rotate_tracker[key] = ('left', img_shape)
        
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            
            if rotate_tracker is not None:
                rotate_tracker[key] = ('right', img_shape)
    
    name = Path(fname).name.split('_')[0]
    
    if transforms is not None:
        for transform in transforms:
            img = transform(img)
    
    if stack_transforms is not None:
        channels = [img]
        
        for transform in stack_transforms:
            transformed = transform(img)
            transformed = cv2.normalize(transformed,  None, 0, 255, cv2.NORM_MINMAX)
            
            channels.append(transformed)

        img = np.array(channels)
        img = np.swapaxes(img, 1, 2)
    
    return img, name



def load_pgm_mammograms(dataset, load_limit=None, load_order=None, **kwargs):
    data_dir = None
    is_mias = False
    
    if dataset.lower() == 'mias':
        data_dir = f'{c.MIAS_DIR}'
        is_mias = True
    
    if load_order is not None:
        img_fnames = [f'{data_dir}/{refnum}.pgm' for refnum in load_order]
        
    else:
        img_fnames = os.listdir(data_dir)
        img_fnames = list(filter(lambda x: '.pgm' in x, img_fnames))
        img_fnames = list(map(lambda x: f'{data_dir}/' + x, img_fnames))
        img_fnames = sorted(img_fnames, key=lambda x: int(Path(x).name.split('_')[0]))
    
    if load_limit is not None:
        img_fnames = img_fnames[load_limit[0]:load_limit[1]]
    
    loaded = [load_pgm_mammogram(fname, is_mias=is_mias, **kwargs) for fname in img_fnames]
    
    imgs = np.array([l[0] for l in loaded])
    names = np.array([l[1] for l in loaded])
    
    return imgs, names


def load_jpg_mammogram(
        fname,
        transforms=None,
        stack_transforms=None
    ):
    img = cv2.imread(fname, -1)
    img = cv2.normalize(img,  None, 0, 255, cv2.NORM_MINMAX)
    
    name = str(Path(fname).parent).split('.')[-1] + '_' + str(Path(fname).name)
    
    if transforms is not None:
        for transform in transforms:
            img = transform(img)
    
    if stack_transforms is not None:
        channels = [img]
        
        for transform in stack_transforms:
            transformed = transform(img)
            transformed = cv2.normalize(transformed,  None, 0, 255, cv2.NORM_MINMAX)
            
            channels.append(transformed)

        img = np.array(channels)
        img = np.swapaxes(img, 1, 2)
    
    return img, name



def load_jpg_mammograms(dataset, load_limit=None, load_order=None, **kwargs):
    data_dir = None
    
    if dataset.lower() == 'ddsm':
        data_dir = f'{c.DDSM_DIR}'
    
    if load_order is not None:
        img_fnames = load_order
        
        if load_limit is not None:
            img_fnames = img_fnames[load_limit[0]:load_limit[1]]
    
    loaded = [load_jpg_mammogram(fname, **kwargs) for fname in img_fnames]
    
    imgs = np.array(loaded)
    names = np.array([Path(fname).name.split('.')[0] for fname in img_fnames])
    
    return imgs, names



def load_jpg_masks(dataset, load_limit=None, load_order=None, **kwargs):
    data_dir = None
    
    if dataset.lower() == 'ddsm':
        data_dir = f'{c.DDSM_DIR}'
    
    if load_order is not None:
        img_fnames = load_order
        
        if load_limit is not None:
            img_fnames = img_fnames[load_limit[0]:load_limit[1]]
    
    loaded = [load_jpg_mammogram(fname, **kwargs) for fname in img_fnames]
    
    imgs = np.array(loaded)
    
    return imgs



def load_preprocessed_images(
        dataset,
        load_order=None,
        path_suffix=None, 
        load_limit=None,
        verbose=False
    ):
    
    data_dir = None
    
    if dataset.lower() == 'inbreast':
        data_dir = f'{c.PREPROCESSED_INBREAST_DIR}/imgs'
    elif dataset.lower() == 'mias':
        data_dir = f'{c.PREPROCESSED_MIAS_DIR}/imgs'
    elif dataset.lower() == 'ddsm':
        data_dir = f'{c.PREPROCESSED_DDSM_DIR}/imgs'
        
    if path_suffix is not None:
        data_dir = f'{data_dir}{path_suffix}'
    
    if load_order is not None:
        img_fnames = [f'{data_dir}/{name}.npy' for name in load_order]
    
    else:
        img_fnames = os.listdir(data_dir)
        img_fnames = list(filter(lambda x: '.npy' in x, img_fnames))
        img_fnames = list(map(lambda x: f'{data_dir}/' + x, img_fnames))
        
    if load_limit is not None:
        img_fnames = img_fnames[load_limit[0]:load_limit[1]]
    
    def load_img(img_fname):
        img = np.load(img_fname).astype(np.uint8)
        
        if verbose:
            print(f'loaded {img_fname}')
            
        return img
    
    loaded = [load_img(fname) for fname in img_fnames]
    
    imgs = np.array(loaded)
    names = np.array([Path(fname).name.split('.')[0] for fname in img_fnames])
    
    return imgs, names