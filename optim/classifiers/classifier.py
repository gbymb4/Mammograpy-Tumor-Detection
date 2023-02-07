# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:52:12 2023

@author: Gavin
"""

import torch

import numpy as np

from torch.optim import RMSprop
from torch.nn import BCELoss

from projectio.loading import format_segmentation_rois, FuzzyBoundingBoxes

class ClassifierOptimiser:
    
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        
    
    
    def execute(
            self,
            epochs=250,
            verbose=True,
            augmentation_callbacks=None,
            fuzzy_bboxes_func=FuzzyBoundingBoxes(
                x_tolerance=0.05,
                y_tolerance=0.05
            ),
            _data_device_override='cuda',
            **kwargs
        ):
        
        criterion = BCELoss()
        
        if len(kwargs) == 0:
            optim = RMSprop(self.model.parameters(), momentum=0.9, lr=1e-3)
        else:
            optim = RMSprop(self.model.parameters(), **kwargs)
            
        train_losses, test_losses = [], []
        train_accs, test_accs = [], []
        
        if verbose:
            print('#'*32)
            print('beginning classifier optimisation loop')
            print('#'*32)
            
        for epoch in range(1, epochs + 1):
            if verbose:
                print(f'--executing epoch {epoch}...')
                
            train_batches = 0
                
            train_loss, test_loss = 0, 0
            train_correct, test_correct = 0, 0
            train_instances, test_instances = 0, 0
            
            self.model.train()
            
            for batch in self.train_loader:
                train_batches += 1
                
                imgs = format_segmentation_rois(batch, fuzzy_bboxes_func)
                imgs = imgs.squeeze(dim=2)
                
                batch = np.array([np.array(elem, dtype=object) for elem in list(zip(*batch))])
                #batch = np.swapaxes(np.array(batch), 0, 1)

                pathologies = torch.from_numpy(batch[:, -2].astype(float)).float()
                
                if _data_device_override is not None:
                    pathologies = pathologies.to(_data_device_override)

                if augmentation_callbacks is not None:
                    imgs = self.__augment(
                        imgs,
                        augmentation_callbacks
                    )
                
                if _data_device_override is not None:
                    imgs = imgs.to(_data_device_override)
                    
                optim.zero_grad()
                
                preds = self.model(imgs)
                
                if len(preds.shape) != 1:
                    preds = preds.squeeze()
                    
                if len(preds.shape) == 0:
                    continue
                
                loss = criterion(preds, pathologies)
                
                loss.backward()
                optim.step()
                
                correct = torch.round(preds) == pathologies
                
                train_instances += len(batch)
                
                train_loss += loss.item()
                
                train_correct += correct.sum().item()
            
            self.model.eval()
            
            train_losses.append(train_loss / train_instances)
            train_accs.append(train_correct / train_instances)
            
            for batch in self.test_loader:
                imgs = format_segmentation_rois(batch, fuzzy_bboxes_func)
                if len(imgs.shape) == 5:
                    imgs = imgs.squeeze(dim=2)
                
                batch = np.array([np.array(elem, dtype=object) for elem in list(zip(*batch))])
                
                pathologies = torch.from_numpy(batch[:, -2].astype(float)).float()
                
                if _data_device_override is not None:
                    pathologies = pathologies.to(_data_device_override)

                if _data_device_override is not None:
                    imgs = imgs.to(_data_device_override)
                
                optim.zero_grad()
                
                preds = self.model(imgs)
                
                if len(preds.shape) != 1:
                    preds = preds.squeeze()
                    
                if len(preds.shape) == 0:
                    continue
                
                loss = criterion(preds, pathologies)
                
                loss.backward()
                optim.step()
                
                correct = torch.round(preds) == pathologies
                
                test_instances += len(batch)
                
                test_loss += loss.item()
                
                test_correct += correct.sum().item()
            
            test_losses.append(test_loss / test_instances)
            test_accs.append(test_correct / test_instances)
            
            if verbose:
                print(f'--evaluated {train_batches} train batches')
                print(f'->avg loss: train = {train_losses[-1]:.4f}, test = {test_losses[-1]:.4f}')
                print(f'->avg acc: train = {train_accs[-1]:.4f}, test = {test_accs[-1]:.4f}')
                print('-'*32)
                
        results = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs,
        }
            
        return results
    
    
    
    def __augment(self, imgs, callbacks):
        new_imgs= []
        
        device = imgs[0].device
        
        for img in imgs:
            temp_img = img.cpu().numpy()
        
            temp = np.swapaxes(temp_img, 0, 2)
            
            for callback in callbacks:
                temp, _ = callback(temp, np.ones((1, 4)))
                
                if len(temp.shape) != 3:
                    temp = temp[:temp[2], :temp[1], np.newaxis]
                
            temp = np.swapaxes(temp, 0, 2) 
            
            temp_img = temp
                
            new_imgs.append(temp_img)
        
        new_imgs = np.array(new_imgs)
        
        new_imgs = torch.from_numpy(new_imgs.copy()).float().to(device)
            
        return new_imgs