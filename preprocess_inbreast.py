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

def main():
    dims = (512, 512)
    
    recenter_tracker = []
    rescale_tracker = []

    load_lim = None

    dataset = 'inbreast'

    imgs, img_names = load_dicom_mammograms(
        dataset,
        load_limit=load_lim,
        transforms=[MakeSquare(recenter_tracker), ResizeImage(dims, rescale_tracker)],
        stack_transforms=[CLAHE()]
    )

    imgs = imgs[:, 1:]

    rois = parse_inbreast_xmls(img_names, load_limit=load_lim)

    save_preprocessed_images(imgs, img_names, dataset)

    rois = adapt_transformed_coords(rois, recenter_tracker, rescale_tracker)
    rois = compute_bounding_boxes(rois)

    save_rois(rois, dataset)
    

    
if __name__ == '__main__':
    main()