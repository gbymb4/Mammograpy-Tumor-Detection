# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 18:31:14 2022

@author: Gavin
"""

import torch

import numpy as np
import pconfig as c

from .images import load_preprocessed_images
from .bboxes import load_bboxes

from torch.utils.data import DataLoader, Dataset

class ROIDataset(Dataset):
    
    def __init__(self, dataset, device='cpu'):
        imgs = load_preprocessed_images(dataset) / 255.0
        imgs = np.swapaxes(imgs, 2, 3)
        imgs = torch.from_numpy(imgs).float()
        
        self.imgs = imgs.to(device)
        
        bboxes = load_bboxes(dataset)
        
        encoding = c.INBREAST_LABEL_ENCODING_SCHEME
        
        all_boxes, all_labels = [], []
        for bbox_data in bboxes.values():
            elem_boxes = bbox_data['bboxes']
            elem_labels = bbox_data['classes']
            
            boxes, labels = [], []
            for box, label in zip(elem_boxes, elem_labels):
                label = label.lower()
                
                if label not in encoding.keys():
                    continue
                
                label = encoding[label]
                
                boxes.append(box)
                labels.append(label)
                
            all_boxes.append(boxes)
            all_labels.append(labels)
            
        all_boxes, all_labels = np.array(all_boxes), np.array(all_labels)
        
        all_boxes = torch.from_numpy(all_boxes).float()
        all_labels = torch.from_numpy(imgs).type(torch.int64)
        
        self.boxes = all_boxes
        self.labels = all_labels
            
        

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        imgs = self.imgs[idx]
        boxes = self.boxes[idx]
        labels = self.labels[idx]

        return imgs, boxes, labels

        

def load_train_test_data(
        dataset_name,
        seed=0,
        device='cpu',
        batch_size=None,
        test_size=None
    ):
    
    generator = torch.Generator().manual_seed(seed)
    
    train_dataset, test_dataset = load_train_test_datasets_only(
        dataset_name,
        seed,
        device,
        batch_size,
        test_size
    )
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    
    return train_dataloader, test_dataloader



def load_train_test_datasets_only(
        dataset_name,
        seed=0,
        device='cpu',
        batch_size=None,
        test_size=None
    ):
    
    if batch_size is None: batch_size = c.BATCH_SIZE
    if test_size is None:test_size = c.TEST_SIZE
    
    dataset = ROIDataset(dataset_name, device=device)
    
    generator = torch.Generator().manual_seed(seed)
    
    num_instances = len(dataset)
    
    train_samples = int((1 - test_size) * num_instances)
    test_samples = num_instances - train_samples
    
    train_dataset, test_dataset = torch.\
        utils.\
        data.\
        random_split(dataset,
                     [train_samples, test_samples],
                     generator=generator)
        
    
    return train_dataset, test_dataset
