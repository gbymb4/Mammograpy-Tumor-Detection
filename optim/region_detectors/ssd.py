# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:58:17 2022

@author: Gavin
"""

import numpy as np

from torch.optim import SGD

from projectio.loading import to_ssd_targets

class SSDOptimiser:
    
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        
    
    
    def execute(self, epochs=1000, verbose=True, **kwargs):
        if len(kwargs) == 0:
            optim = SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        else:
            optim = SGD(self.model.parameters(), **kwargs)
        
        train_losses, test_losses = [], []
        train_detect_fracs, test_detect_fracs = [], []
        
        if verbose:
            print('#'*32)
            print('beginning SSD optimisation loop')
            print('#'*32)
        
        for epoch in range(1, epochs + 1):
            
            if verbose:
                print(f'--executing epoch {epoch}...', end='')
            
            train_loss, test_loss = [], []
            train_detect_frac, test_detect_frac = [], []
            
            self.model.train()
            
            for batch in self.train_loader:
                imgs, boxes, labels = batch
                
                optim.zero_grad()
                
                targets = to_ssd_targets(boxes, labels, device=imgs[0].device)
                
                losses, detections = self.model(imgs, targets)
                
                avg_detect_frac = self.__compute_avg_detect_frac(
                    detections,
                    boxes,
                )
                
                train_detect_frac.append(avg_detect_frac)
                
                total_loss = losses['bbox_regression'] + losses['classification']
                total_loss.backward()
                
                train_loss.append((total_loss.item(), len(imgs)))
                
                optim.step()
                
            total_train_loss = sum([e[0] for e in train_loss])
            total_instances = sum([e[1] for e in train_loss])
            
            avg_train_loss = total_train_loss / total_instances
            
            train_losses.append(avg_train_loss)
            
            avg_train_detect_frac = sum(train_detect_frac) / len(train_detect_frac)
            train_detect_fracs.append(avg_train_detect_frac)
            
            if verbose:
                print(f'evaluated {len(train_loss)} train batches with avg_loss {avg_train_loss:.4f}')
                print('-'*32)
                
            self.model.eval()
                
            for batch in self.test_loader:
                imgs, boxes, labels = batch
                    
                targets = to_ssd_targets(boxes, labels, device=imgs[0].device)
                    
                losses, detections = self.model(imgs, targets)
                
                avg_detect_frac = self.__compute_avg_detect_frac(
                    detections,
                    boxes,
                )
                
                test_detect_frac.append(avg_detect_frac)
                    
                total_loss = losses['bbox_regression'] + losses['classification']
                
                test_loss.append((total_loss.item(), len(imgs)))
                
            total_test_loss = sum([e[0] for e in test_loss])
            total_instances = sum([e[1] for e in test_loss])
            
            avg_test_loss = total_test_loss / total_instances
                
            test_losses.append(avg_test_loss)
            
            avg_test_detect_frac = sum(test_detect_frac) / len(test_detect_frac)
            test_detect_fracs.append(avg_test_detect_frac)
            
        train_losses, test_losses = np.array(train_losses), np.array(test_losses)
        
        train_detect_fracs = np.array(train_detect_fracs)
        test_detect_fracs = np.array(test_detect_fracs)
        
        return train_losses, test_losses, train_detect_fracs, test_detect_fracs
    
    
    
    def __compute_avg_detect_frac(self, detections, boxes):
        batch_fracs = []
        
        for detection, true_boxes in zip(detections, boxes):
            pred_boxes = detection['boxes']
            
            detect_frac = len(pred_boxes) / len(true_boxes)
            
            batch_fracs.append(detect_frac)
            
        avg_frac = sum(batch_fracs) / len(batch_fracs)
        
        return avg_frac
            
            