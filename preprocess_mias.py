# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 15:50:52 2023

@author: Gavin
"""

from projectio.preprocessing import parse_mias_info, infer_bounding_boxes
from projectio.preprocessing import BreastMasking, MakeSquare, ResizeImage, CLAHE
from projectio.preprocessing import adapt_transformed_bboxes

from projectio.loading import load_pgm_mammograms

from projectio.saving import save_bboxes, save_preprocessed_images

def main():
    dims = (512, 512)

    dataset = 'mias'

    recenter_tracker = []
    rescale_tracker = []

    mias_info = parse_mias_info()

    load_order = mias_info['REFNUM']
    
    imgs, img_names = load_pgm_mammograms(
        dataset,
        load_order=load_order, 
        transforms=[BreastMasking(), MakeSquare(recenter_tracker), ResizeImage(dims, rescale_tracker)],
        stack_transforms=[CLAHE()]
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
    
    bboxes = infer_bounding_boxes(load_order, centers, radii)
    bboxes = adapt_transformed_bboxes(bboxes, recenter_tracker, rescale_tracker)

    save_preprocessed_images(imgs, img_names, dataset)
    save_bboxes(bboxes, dataset)
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    for img, bboxes_dict in zip(imgs, bboxes.values()):
        fig, ax = plt.subplots()
        
        ax.imshow(np.swapaxes(img, 0, 2), cmap='gray')
        
        x_min, y_min, x_max, y_max = bboxes_dict['bboxes'][0]

        # Create the rectangle patch
        rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, fill=False, color='red')
        
        ax.add_patch(rect)
        
        plt.show()
    
    
if __name__ == '__main__':
    main()
    