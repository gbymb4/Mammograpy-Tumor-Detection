# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 18:31:14 2022

@author: Gavin
"""

import torch

import numpy as np
import pconfig as c

from collections import Counter

from .images import load_preprocessed_images
from .bboxes import load_bboxes

from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, ConcatDataset
from sklearn.model_selection import KFold

class ROIDataset(Dataset):
    
    def __init__(
            self,
            dataset,
            device='cpu', 
            filter_empty_imgs=True,
            bbox_min_size=7,
            load_limit=None
        ):

        super().__init__()
        
        bboxes = load_bboxes(
            dataset,
            load_limit=load_limit
        )
        
        imgs, _ = load_preprocessed_images(
            dataset,
            load_order=list(bboxes.keys()),
            load_limit=load_limit
        )
        
        imgs = imgs.astype(float) / 255.0
        imgs = torch.from_numpy(imgs).float()
        imgs = imgs.to(device)
        
        if dataset.lower() == 'inbreast':
            encoding = c.INBREAST_LABEL_ENCODING_SCHEME
        elif dataset.lower() == 'mias':
            encoding = c.MIAS_LABEL_ENCODING_SCEHEME
        elif dataset.lower() == 'ddsm':
            encoding = c.DDSM_LABEL_ENCODING_SCEHEME
        
        all_boxes, all_labels = [], []
        for bbox_data in bboxes.values():
            elem_boxes = bbox_data['bboxes']
            elem_labels = bbox_data['classes']
            
            boxes, labels = [], []
            for box, label in zip(elem_boxes, elem_labels):
                label_str = label.lower()
                
                if label_str not in encoding.keys():
                    continue
                
                if bbox_lt_min(box, bbox_min_size):
                    continue
                    
                label = encoding[label_str]
                
                boxes.append(box)
                labels.append(label)
            
            boxes = np.array(boxes)
            boxes = torch.from_numpy(boxes).float()
            boxes = boxes.to(device)
            
            all_boxes.append(boxes)
            
            labels = np.array(labels)
            labels = torch.from_numpy(labels).type(torch.int64)
            labels = labels.to(device)
            
            all_labels.append(labels)
        
        if filter_empty_imgs:
            mask = []
            
            for boxes in all_boxes:
                
                if len(boxes) == 0: 
                    mask.append(False)
                else:
                    mask.append(True)
            
            imgs = imgs[mask]
            
            new_boxes, new_labels = [], []
            for boxes, labels, mask_elem in zip(all_boxes, all_labels, mask):
                if mask_elem:
                    new_boxes.append(boxes)
                    new_labels.append(labels)
                
            all_boxes = new_boxes
            all_labels = new_labels
        
        self.imgs = imgs
        
        self.boxes = all_boxes
        self.labels = all_labels
        
        

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        imgs = self.imgs[idx]
        boxes = self.boxes[idx]
        labels = self.labels[idx]
        
        return imgs, boxes, labels



class SegmentationDataset(Dataset):
    
    def __init__(
            self,
            dataset,
            device='cpu', 
            filter_empty_imgs=True,
            bbox_min_size=7,
            load_limit=None
        ):

        super().__init__()
        
        bboxes = load_bboxes(
            dataset,
            load_limit=load_limit
        )
        
        imgs, img_names = load_preprocessed_images(
            dataset,
            load_order=list(bboxes.keys()),
            path_suffix='_highres',
            load_limit=load_limit
        )
        
        imgs = imgs.astype(float) / 255.0
        imgs = torch.from_numpy(imgs).float()
        imgs = imgs.to(device)
        
        if dataset.lower() == 'inbreast':
            encoding = c.INBREAST_LABEL_ENCODING_SCHEME
        elif dataset.lower() == 'ddsm':
            encoding = c.DDSM_LABEL_ENCODING_SCEHEME
        
        all_boxes, all_labels, all_coords = [], [], []
        for bbox_data in bboxes.values():
            elem_boxes = bbox_data['bboxes']
            elem_labels = bbox_data['classes']
            elem_coords = bbox_data['coords']
            
            boxes, labels, coords = [], [], []
            for box, label, coord in zip(elem_boxes, elem_labels, elem_coords):
                label_str = label.lower()
                
                if label_str not in encoding.keys():
                    continue
                
                if bbox_lt_min(box, bbox_min_size):
                    continue
                    
                label = encoding[label_str]
                
                boxes.append(box)
                labels.append(label)
                coords.append(coord)
            
            boxes = np.array(boxes)
            boxes = torch.from_numpy(boxes).float()
            boxes = boxes.to(device)
            
            all_boxes.append(boxes)
            
            labels = np.array(labels)
            labels = torch.from_numpy(labels).type(torch.int64)
            labels = labels.to(device)
            
            all_labels.append(labels)
            
            all_coords.append(coords)
            
        if filter_empty_imgs:
            mask = []
            
            for boxes in all_boxes:
                if len(boxes) == 0: 
                    mask.append(False)
                else:
                    mask.append(True)
        
            imgs = imgs[mask]
            
            new_boxes, new_labels, new_coords = [], [], []
            for boxes, labels, coords, mask_elem in zip(all_boxes, all_labels, all_coords, mask):
                if mask_elem:
                    new_boxes.append(boxes)
                    new_labels.append(labels)
                    new_coords.append(coords)
                
            all_boxes = new_boxes
            all_labels = new_labels
            all_coords = new_coords
        
        rois_cropping = []
        for i, (boxes, coords, labels) in enumerate(zip(all_boxes, all_coords, all_labels)):
            for box, coord, label in zip(boxes, coords, labels):
                rois_cropping.append((box, coord, label, imgs[i]))
                
        self.rois = np.array(rois_cropping)
        
        

    def __len__(self):
        return len(self.rois)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        rois = self.rois[idx]
        
        return rois
    


class ClassificationDataset(Dataset):
    
    def __init__(
            self,
            dataset,
            device='cpu', 
            filter_empty_imgs=True,
            bbox_min_size=7,
            load_limit=None
        ):

        super().__init__()
        
        bboxes = load_bboxes(
            dataset,
            load_limit=load_limit
        )
        
        imgs, img_names = load_preprocessed_images(
            dataset,
            load_order=list(bboxes.keys()),
            path_suffix='_highres',
            load_limit=load_limit
        )
        
        imgs = imgs.astype(float) / 255.0
        imgs = torch.from_numpy(imgs).float()
        imgs = imgs.to(device)
        
        bboxes = load_bboxes(
            dataset,
            fname_suffix='_highres',
            load_limit=load_limit
        )
        
        encoding = c.DDSM_LABEL_ENCODING_SCHEME
        
        all_boxes, all_labels, all_coords, all_pathologies = [], [], [], []
        for bbox_data in bboxes.values():
            elem_boxes = bbox_data['bboxes']
            elem_labels = bbox_data['classes']
            elem_coords = bbox_data['coords']
            elem_pathologies = bbox_data['pathologies']
            
            boxes, labels, coords, pathologies = [], [], [], []
            for box, label, coord, pathology in zip(elem_boxes, elem_labels, elem_coords, elem_pathologies):
                label_str = label.lower()
                
                if label_str not in encoding.keys():
                    continue
                
                if bbox_lt_min(box, bbox_min_size):
                    continue
                    
                label = encoding[label_str]
                
                pathology = c.DDSM_PATHOLOGY_ENCODING_SCHEME[pathology]
                
                boxes.append(box)
                labels.append(label)
                coords.append(coord)
                pathologies.append(pathology)
            
            boxes = np.array(boxes)
            boxes = torch.from_numpy(boxes).float()
            boxes = boxes.to(device)
            
            all_boxes.append(boxes)
            
            labels = np.array(labels)
            labels = torch.from_numpy(labels).type(torch.int64)
            labels = labels.to(device)
            
            all_labels.append(labels)
            
            pathologies = np.array(pathologies)
            
            all_pathologies.append(pathologies)
            
            all_coords.append(coords)
            
            
        if filter_empty_imgs:
            mask = []
            
            for boxes in all_boxes:
                if len(boxes) == 0: 
                    mask.append(False)
                else:
                    mask.append(True)
        
            imgs = imgs[mask]
            
            new_boxes, new_labels, new_coords, new_pathologies = [], [], [], []
            for boxes, labels, coords, pathologies, mask_elem in zip(all_boxes, all_labels, all_coords, all_pathologies, mask):
                if mask_elem:
                    new_boxes.append(boxes)
                    new_labels.append(labels)
                    new_coords.append(coords)
                    new_pathologies.append(pathologies)
                
            all_boxes = new_boxes
            all_labels = new_labels
            all_coords = new_coords
            all_pathologies = new_pathologies
        
        rois_cropping = []
        for i, (boxes, coords, labels, pathologies) in enumerate(zip(all_boxes, all_coords, all_labels, all_pathologies)):
            for box, coord, label, pathology in zip(boxes, coords, labels, pathologies):
                rois_cropping.append((box, coord, label, pathology, imgs[i]))
                
        self.rois = np.array(rois_cropping)
        
        

    def __len__(self):
        return len(self.rois)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        rois = self.rois[idx]
        
        return rois



def load_cross_validation_sets(
        dataset_name,
        seed=0,
        device='cpu',
        batch_size=1,
        folds=5,
        **kwargs
    ):
    
    generator = torch.Generator().manual_seed(seed)
    
    splits = KFold(n_splits=folds, shuffle=True, random_state=seed)
    
    dataset = ROIDataset(dataset_name, device=device, **kwargs)

    all_folds = []
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            generator=generator,
            collate_fn=__collate
        )
        
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            generator=generator,
            collate_fn=__collate
        )
        
        all_folds.append((train_loader, test_loader))
        
    return all_folds

        

def load_train_test_data(
        dataset_names,
        dataset_type,
        seed=0,
        device='cpu',
        batch_size=1,
        test_size=0.1,
        **kwargs
    ):
    
    generator = torch.Generator().manual_seed(seed)
    
    train_dataset, test_dataset = load_train_test_datasets_only(
        dataset_names,
        dataset_type,
        seed,
        device,
        batch_size,
        test_size,
        **kwargs
    )
    
    if dataset_type is ClassificationDataset:
        def init_dataloader(
                dataset,
                batch_size,
                shuffle,
                generator,
                collate_fn
            ):
                data = dataset[0].rois
            
                pathology_count = Counter([d[3] for d in data])
                
                weights = [1/pathology_count[d[3]] for d in data]
                weights = torch.DoubleTensor(weights)
                
                sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
                
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    generator=generator,
                    collate_fn=collate_fn,
                    sampler=sampler
                )
    
    else:
        def init_dataloader(
                dataset,
                batch_size,
                shuffle,
                generator,
                collate_fn
            ):
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    generator=generator,
                    collate_fn=collate_fn
                )
        
    train_dataloader = init_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=__collate
    )
    
    test_dataloader = init_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=__collate
    )
    
    return train_dataloader, test_dataloader



def load_train_test_datasets_only(
        dataset_names,
        dataset_type,
        seed=0,
        device='cpu',
        batch_size=1,
        test_size=0.1,
        **kwargs
    ):
    
    generator = torch.Generator().manual_seed(seed)
    
    datasets = [dataset_type(dataset_name, device=device, **kwargs) for dataset_name in dataset_names]
    
    dataset = ConcatDataset(datasets)
    
    num_instances = len(dataset)
    
    train_samples = int((1 - test_size) * num_instances)
    test_samples = num_instances - train_samples
    
    train_dataset, test_dataset = torch.\
        utils.\
        data.\
        random_split(
            dataset,
            [train_samples, test_samples],
            generator=generator
        )
        
    
    return train_dataset, test_dataset



def to_ssd_targets(boxes, labels, ignore_labels=True, device='cpu'):
    if ignore_labels:
        targets = [{
            'boxes': bs,
            'labels': torch.ones((len(ls),), dtype=torch.long).to(device)
        } for bs, ls in zip(boxes, labels)]
    else:
        targets = [{
            'boxes': bs,
            'labels': ls,
        } for bs, ls in zip(boxes, labels)]
        
    targets = np.array(targets, dtype=dict)
    
    return targets



def bbox_lt_min(bbox, bbox_min):
    min_dim = min([bbox[2] - bbox[0], bbox[3] - bbox[1]])
    
    return min_dim < bbox_min



def __collate(batch):
    return list(zip(*batch))

