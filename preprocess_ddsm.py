# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 13:55:50 2023

@author: Gavin
"""

import numpy as np

from scipy.ndimage import center_of_mass

from projectio.preprocessing import parse_ddsm_info
from projectio.preprocessing import (
    BreastMasking,
    MakeSquare,
    ResizeImage,
    LargestRegionMasking,
    CLAHE
)
from projectio.preprocessing import compute_coords_of_mask, compute_bounding_boxes

from projectio.loading import load_jpg_mammograms, load_jpg_masks

from projectio.saving import save_rois, save_preprocessed_images

dataset = 'ddsm'

def main():
    __roi_dataset()
    __segmentation_classification_dataset()
    
    
    
def __roi_dataset():
    dims = (512, 512)
    
    load_limit = None
    
    mammograms_info, masks_info = parse_ddsm_info()
    
    mammograms_info = mammograms_info[:load_limit]
    masks_info = masks_info[:load_limit]
    
    mammogram_img_paths = mammograms_info['image_path']
    
    imgs, _ = load_jpg_mammograms(
        dataset,
        load_order=mammogram_img_paths, 
        transforms=[
            BreastMasking(),
            MakeSquare(),
            ResizeImage(dims),
            LargestRegionMasking()
        ],
        stack_transforms=[
            CLAHE()
        ],
        load_limit=load_limit
    )
    
    imgs = np.array([img for img, _ in imgs])[:, 1:]
    
    img_names = mammograms_info['mammogram_names']
    
    mask_img_paths = masks_info['image_path']
    mask_img_names = masks_info['mammogram_names']
    
    padding_override = []
    for img_name in mask_img_names:
        idx = list(img_names).index(img_name)
        
        img = np.swapaxes(imgs[idx][0], 0, 1)
        
        img_copy = img.copy()
        img_copy[(img_copy > 41) == False] = 0
        
        com = center_of_mass(img_copy > 0)
        _, w = img.shape
        horiz_center = w // 2
        is_left = com[1] < horiz_center
        
        direction = 'vert_left' if is_left else 'vert_right'
        
        padding_override.append(direction)

    mask_imgs = load_jpg_masks(
        dataset,
        load_order=mask_img_paths, 
        transforms=[
            MakeSquare(_padding_override=padding_override),
            ResizeImage(dims)
        ],
        load_limit=load_limit
    )
    
    mask_imgs = np.array([img for img, _ in mask_imgs])
    
    mammograms_to_masks = {img_name: {
        'classes': [],
        'pathologies': [],
        'coords': [],
        'bboxes': []
    } for img_name in img_names}
    
    for (index, row), mask in zip(masks_info.iterrows(), mask_imgs):
        name = row['mammogram_names']
        pathology = row['pathology']
        label = row['abnormality type']
        
        if name not in mammograms_to_masks.keys(): break
        
        coords = compute_coords_of_mask(mask)
        
        mammograms_to_masks[name]['pathologies'].append(pathology)
        mammograms_to_masks[name]['classes'].append(label)
        mammograms_to_masks[name]['coords'].append(coords)
    
    mammograms_to_masks = compute_bounding_boxes(mammograms_to_masks)
    
    save_rois(mammograms_to_masks, dataset)
    save_preprocessed_images(imgs, img_names, dataset)
        


def __segmentation_classification_dataset():
    dims = (2048, 2048)
    
    load_limit = None
    
    mammograms_info, masks_info = parse_ddsm_info()
    
    mammograms_info = mammograms_info[:load_limit]
    masks_info = masks_info[:load_limit]
    
    mammogram_img_paths = mammograms_info['image_path']
    
    imgs, _ = load_jpg_mammograms(
        dataset,
        load_order=mammogram_img_paths, 
        transforms=[
            BreastMasking(),
            MakeSquare(),
            ResizeImage(dims),
            LargestRegionMasking()
        ],
        stack_transforms=[
            CLAHE()
        ],
        load_limit=load_limit
    )
    
    imgs = np.array([img for img, _ in imgs])[:, 1:]
    
    img_names = mammograms_info['mammogram_names']
    
    mask_img_paths = masks_info['image_path']
    mask_img_names = masks_info['mammogram_names']
    
    padding_override = []
    for img_name in mask_img_names:
        idx = list(img_names).index(img_name)
        
        img = np.swapaxes(imgs[idx][0], 0, 1)
        
        img_copy = img.copy()
        img_copy[(img_copy > 41) == False] = 0
        
        com = center_of_mass(img_copy > 0)
        _, w = img.shape
        horiz_center = w // 2
        is_left = com[1] < horiz_center
        
        direction = 'vert_left' if is_left else 'vert_right'
        
        padding_override.append(direction)

    mask_imgs = load_jpg_masks(
        dataset,
        load_order=mask_img_paths, 
        transforms=[
            MakeSquare(_padding_override=padding_override),
            ResizeImage(dims)
        ],
        load_limit=load_limit
    )
    
    mask_imgs = np.array([img for img, _ in mask_imgs])
    
    mammograms_to_masks = {img_name: {
        'classes': [],
        'pathologies': [],
        'coords': [],
        'bboxes': []
    } for img_name in img_names}
    
    for (index, row), mask in zip(masks_info.iterrows(), mask_imgs):
        name = row['mammogram_names']
        pathology = row['pathology']
        label = row['abnormality type']
        
        if name not in mammograms_to_masks.keys(): break
        
        coords = compute_coords_of_mask(mask)
        
        mammograms_to_masks[name]['pathologies'].append(pathology)
        mammograms_to_masks[name]['classes'].append(label)
        mammograms_to_masks[name]['coords'].append(coords)
    
    mammograms_to_masks = compute_bounding_boxes(mammograms_to_masks)
    
    save_rois(mammograms_to_masks, dataset, path_suffix='_highres')
    save_preprocessed_images(imgs, img_names, dataset, path_suffix='_highres')
    
    
    
if __name__ == '__main__':
    main()
    