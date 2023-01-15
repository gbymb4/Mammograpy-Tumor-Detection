# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 17:34:00 2022

@author: Gavin
"""

from projectio.loading import load_dicom_mammograms

from projectio.preprocessing import (
    MakeSquare, ResizeImage, CLAHE,
    parse_inbreast_xmls, adapt_transformed_coords, compute_bounding_boxes
)

from projectio.saving import save_preprocessed_images, save_rois

dataset = 'inbreast'

def main():
    __roi_dataset()
    __segmentation_classification_dataset()
    
    
    
def __roi_dataset():
    dims = (512, 512)
    
    recenter_tracker = []
    rescale_tracker = []

    imgs, img_names = load_dicom_mammograms(
        dataset,
        transforms=[MakeSquare(recenter_tracker), ResizeImage(dims, rescale_tracker)],
        stack_transforms=[CLAHE()]
    )

    imgs = imgs[:, 1:]

    rois = parse_inbreast_xmls(img_names)

    save_preprocessed_images(imgs, img_names, dataset)

    rois = adapt_transformed_coords(rois, recenter_tracker, rescale_tracker)
    rois = compute_bounding_boxes(rois)

    save_rois(rois, dataset)
    
    

def __segmentation_classification_dataset():
    dims = (2048, 2048)
    
    recenter_tracker = []
    rescale_tracker = []

    imgs, img_names = load_dicom_mammograms(
        dataset,
        transforms=[MakeSquare(recenter_tracker), ResizeImage(dims, rescale_tracker)],
        stack_transforms=[CLAHE()]
    )
    
    imgs = imgs[:, 1:]

    rois = parse_inbreast_xmls(img_names)

    save_preprocessed_images(imgs, img_names, dataset, path_suffix='_highres')

    rois = adapt_transformed_coords(rois, recenter_tracker, rescale_tracker)
    rois = compute_bounding_boxes(rois)

    save_rois(rois, dataset, fname_suffix='_highres')
    
    
    
if __name__ == '__main__':
    main()