# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:58:17 2022

@author: Gavin
"""

import torch

import numpy as np

from torch.optim import SGD

from projectio.loading import to_ssd_targets
from optim.metrics import compute_batch_metric, compute_fpr, compute_tpr

class SSDOptimiser:
    
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        
    
    
    def execute(
            self,
            epochs=1000,
            verbose=True,
            augmentation_callbacks=None,
            **kwargs
        ):
        
        if len(kwargs) == 0:
            optim = SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        else:
            optim = SGD(self.model.parameters(), **kwargs)
        
        train_losses, test_losses = [], []
        train_detect_fracs, test_detect_fracs = [], []
        train_tprs, test_tprs = [], []
        train_fprs, test_fprs = [], []
        
        if verbose:
            print('#'*32)
            print('beginning SSD optimisation loop')
            print('#'*32)
        
        for epoch in range(1, epochs + 1):
            
            if verbose:
                print(f'--executing epoch {epoch}...', end='')
            
            train_loss, test_loss = [], []
            train_detect_frac, test_detect_frac = [], []
            train_tpr, test_tpr = [], []
            train_fpr, test_fpr = [], []
            
            self.model.train()
            
            for i, batch in enumerate(self.train_loader):
                imgs, boxes, labels = batch
                
                if augmentation_callbacks is not None:
                    imgs, boxes = self.__augment(
                        imgs,
                        boxes,
                        augmentation_callbacks
                    )
                
                optim.zero_grad()
                
                targets = to_ssd_targets(boxes, labels, device=imgs[0].device)
                
                losses, detections = self.model(imgs, targets)
                
                avg_detect_frac = self.__compute_avg_detect_frac(
                    detections,
                    boxes,
                )
                '''
                if i == 0 and epoch % 5 == 0:
                    import matplotlib.pyplot as plt
                    from matplotlib.patches import Rectangle
                    
                    fig, axs = plt.subplots(len(batch[0]), 1, figsize=(10, len(batch[0]) * 10))
                    
                    for ax, img, pboxes, det, label in zip(axs, imgs, targets, detections, labels):
                        img = np.swapaxes(img.cpu().detach().numpy(), 0, 2)
                        ax.imshow(img, cmap='gray')
                        fig.suptitle('training', fontsize=40)
                        for bbox in pboxes['boxes']:
                            x_min, y_min, x_max, y_max = bbox.cpu().detach().numpy()
                            
                            rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, fill=False, color='red', linewidth=2)
                            
                            ax.add_patch(rect)
                            
                        for bbox in det['boxes']:
                            x_min, y_min, x_max, y_max = bbox.cpu().detach().numpy()
                            
                            rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, fill=False, color='lightblue', linewidth=2)
                            
                            ax.add_patch(rect)
                        
                        ax.axis('off')
                    
                    fig.tight_layout()
                    
                    plt.show()
                '''
                train_detect_frac.append(avg_detect_frac)
                
                total_loss = losses['bbox_regression'] + losses['classification']
                total_loss.backward()
                
                train_loss.append((total_loss.item(), len(imgs)))
                
                detects_and_boxes = list(zip(detections, boxes))
                
                train_tpr.extend(compute_batch_metric(compute_tpr, detects_and_boxes))
                train_fpr.extend(compute_batch_metric(compute_fpr, detects_and_boxes))
                
                optim.step()
                
            total_train_loss = sum([e[0] for e in train_loss])
            total_instances = sum([e[1] for e in train_loss])
            
            avg_train_loss = total_train_loss / total_instances
            
            train_losses.append(avg_train_loss)
            
            avg_train_detect_frac = sum(train_detect_frac) / len(train_detect_frac)
            train_detect_fracs.append(avg_train_detect_frac)
            
            train_tprs.append(sum(train_tpr) / len(train_tpr))
            train_fprs.append(sum(train_fpr) / len(train_fpr))
                
            self.model.eval()
                
            for i, batch in enumerate(self.test_loader):
                imgs, boxes, labels = batch
                    
                targets = to_ssd_targets(boxes, labels, device=imgs[0].device)
                
                losses, detections = self.model(imgs, targets)
                
                avg_detect_frac = self.__compute_avg_detect_frac(
                    detections,
                    boxes,
                )
                '''
                if i == 0 and epoch % 5 == 0:
                    import matplotlib.pyplot as plt
                    from matplotlib.patches import Rectangle
                    
                    fig, axs = plt.subplots(len(batch[0]), 1, figsize=(10, len(batch[0]) * 10))
                    
                    for ax, img, pboxes, det, label in zip(axs, imgs, targets, detections, labels):
                        img = np.swapaxes(img.cpu().detach().numpy(), 0, 2)
                        ax.imshow(img, cmap='gray')
                        fig.suptitle('testing', fontsize=40)
                        for bbox in pboxes['boxes']:
                            x_min, y_min, x_max, y_max = bbox.cpu().detach().numpy()
                            
                            rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, fill=False, color='red', linewidth=2)
                            
                            ax.add_patch(rect)
                            
                        for bbox in det['boxes']:
                            x_min, y_min, x_max, y_max = bbox.cpu().detach().numpy()
                            
                            rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, fill=False, color='lightblue', linewidth=2)
                            
                            ax.add_patch(rect)
                        
                        ax.axis('off')
                    
                    fig.tight_layout()
                    
                    plt.show()
                '''
                test_detect_frac.append(avg_detect_frac)
                    
                total_loss = losses['bbox_regression'] + losses['classification']
                
                test_loss.append((total_loss.item(), len(imgs)))
                
                detects_and_boxes = list(zip(detections, boxes))
                
                test_tpr.extend(compute_batch_metric(compute_tpr, detects_and_boxes))
                test_fpr.extend(compute_batch_metric(compute_fpr, detects_and_boxes))
                
            total_test_loss = sum([e[0] for e in test_loss])
            total_instances = sum([e[1] for e in test_loss])
            
            avg_test_loss = total_test_loss / total_instances
                
            test_losses.append(avg_test_loss)
            
            avg_test_detect_frac = sum(test_detect_frac) / len(test_detect_frac)
            test_detect_fracs.append(avg_test_detect_frac)
            
            test_tprs.append(sum(test_tpr) / len(test_tpr))
            test_fprs.append(sum(test_fpr) / len(test_fpr))
            
            if verbose:
                print(f'evaluated {len(train_loss)} train batches with avg_loss {avg_train_loss:.4f}')
                print(f'->TPR: train = {train_tprs[-1]:.4f}, test = {test_tprs[-1]:.4f}')
                print(f'->FPR: train = {train_fprs[-1]:.4f}, test = {test_fprs[-1]:.4f}')
                print(f'->Detection Fraction: train = {train_detect_fracs[-1]:.4f}, test = {test_detect_fracs[-1]:.4f}')
                print('-'*32)
            
        train_losses, test_losses = np.array(train_losses), np.array(test_losses)
        
        train_detect_fracs = np.array(train_detect_fracs)
        test_detect_fracs = np.array(test_detect_fracs)
        
        train_tprs, test_tprs = np.array(train_tprs), np.array(test_tprs)
        train_fprs, test_fprs = np.array(train_fprs), np.array(test_fprs)
        
        results = {
            'train_loss': train_losses,
            'test_loss': test_losses,
            'train_detect_frac': train_detect_fracs,
            'test_detect_frac': test_detect_fracs,
            'train_tpr': train_tprs,
            'test_tpr': test_tprs,
            'train_fpr': train_fprs,
            'test_fpr': test_fprs,
        }
        
        return results
    
    
    
    def __compute_avg_detect_frac(self, detections, boxes):
        batch_fracs = []
        
        for detection, true_boxes in zip(detections, boxes):
            pred_boxes = detection['boxes']
            
            detect_frac = len(pred_boxes) / len(true_boxes)
            
            batch_fracs.append(detect_frac)
            
        avg_frac = sum(batch_fracs) / len(batch_fracs)
        
        return avg_frac
    
    
    
    def __augment(self, imgs, boxes, callbacks):
        new_imgs, new_boxes = [], []
        
        device = imgs[0].device
        
        img_shape = imgs[0].shape
        
        for img, img_boxes in zip(imgs, boxes):
            temp_img = np.swapaxes(img.cpu().numpy(), 0, 2)
            temp_boxes = img_boxes.cpu().numpy()
            
            for callback in callbacks:
                temp_img, temp_boxes = callback(temp_img, temp_boxes)
                
                if len(temp_img.shape) != 3:
                    temp_img = temp_img[:img_shape[2], :img_shape[1], np.newaxis]
                
            temp_img = np.swapaxes(temp_img, 0, 2) 
            temp_img = torch.from_numpy(temp_img.copy()).float().to(device)
            
            temp_boxes = torch.from_numpy(temp_boxes.copy()).float().to(device)
                
            new_imgs.append(temp_img)
            new_boxes.append(temp_boxes)
            
        return new_imgs, new_boxes
            
            