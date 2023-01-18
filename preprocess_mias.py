# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 15:50:52 2023

@author: Gavin
"""

from projectio.preprocessing import parse_mias_info, infer_bounding_boxes
from projectio.preprocessing import (
    BreastMasking,
    MakeSquare,
    ResizeImage,
    LargestRegionMasking,
    CLAHE
)
from projectio.preprocessing import adapt_transformed_bboxes

from projectio.loading import load_pgm_mammograms, assign_bbox_labels

from projectio.saving import save_rois, save_preprocessed_images

dataset = 'mias'

def main():
    __roi_dataset()
    __segmentation_classification_dataset()
    
    
    
def __roi_dataset():
    dims = (512, 512)

    rotate_tracker = {}
    
    recenter_tracker = []
    rescale_tracker = []
    
    mias_info = parse_mias_info()
    
    load_order = mias_info['REFNUM'].drop_duplicates()
    
    imgs, img_names = load_pgm_mammograms(
        dataset,
        load_order=load_order, 
        transforms=[
            BreastMasking(),
            MakeSquare(recenter_tracker),
            ResizeImage(dims, rescale_tracker),
            LargestRegionMasking()
        ],
        stack_transforms=[
            CLAHE()
        ],
        rotate_tracker=rotate_tracker
    )
    
    img_names = [name[:-4] for name in img_names]

    imgs = imgs[:, 1:]

    xs = mias_info['X']
    xs = list(map(float, xs))

    ys = mias_info['Y']
    ys = list(map(float, ys))

    centers = list(zip(xs, ys))

    radii = list(mias_info['RADIUS'])
    radii = list(map(float, radii))
    
    bboxes = infer_bounding_boxes(
        mias_info['REFNUM'],
        centers, 
        radii, 
        rotate_tracker
    )
    
    bboxes = assign_bbox_labels(
        bboxes, 
        mias_info['REFNUM'], 
        mias_info['CLASS']
    )
    
    bboxes = adapt_transformed_bboxes(
        bboxes,
        recenter_tracker=recenter_tracker,
        rescale_tracker=rescale_tracker
    )

    save_preprocessed_images(imgs, img_names, dataset)
    save_rois(bboxes, dataset)
    
    
    
def __segmentation_classification_dataset():
    dims = (2048, 2048)

    rotate_tracker = {}
    
    recenter_tracker = []
    rescale_tracker = []
    
    mias_info = parse_mias_info()
    
    load_order = mias_info['REFNUM'].drop_duplicates()
    
    imgs, img_names = load_pgm_mammograms(
        dataset,
        load_order=load_order, 
        transforms=[
            BreastMasking(),
            MakeSquare(recenter_tracker),
            ResizeImage(dims, rescale_tracker),
            LargestRegionMasking()
        ],
        stack_transforms=[
            CLAHE()
        ],
        rotate_tracker=rotate_tracker
    )
    
    img_names = [name[:-4] for name in img_names]

    imgs = imgs[:, 1:]

    xs = mias_info['X']
    xs = list(map(float, xs))

    ys = mias_info['Y']
    ys = list(map(float, ys))

    centers = list(zip(xs, ys))

    radii = list(mias_info['RADIUS'])
    radii = list(map(float, radii))
    
    bboxes = infer_bounding_boxes(
        mias_info['REFNUM'],
        centers, 
        radii, 
        rotate_tracker
    )
    
    bboxes = assign_bbox_labels(
        bboxes, 
        mias_info['REFNUM'], 
        mias_info['CLASS']
    )
    
    bboxes = adapt_transformed_bboxes(
        bboxes,
        recenter_tracker=recenter_tracker,
        rescale_tracker=rescale_tracker
    )

    save_preprocessed_images(imgs, img_names, dataset, path_suffix='_highres')
    save_rois(bboxes, dataset, fname_suffix='_highres')
    
    
    
if __name__ == '__main__':
    main()
    