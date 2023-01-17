# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:30:26 2023

@author: Gavin
"""

import torch

from torch.optim import Adam
from torch.nn import BCELoss

from projectio.loading import format_segmentation_rois, FuzzyBoundingBoxes
from optim.criterions import BinaryDiceLoss
from optim.metrics import batch_mask_ious

class CGANOptimiser:
    
    def __init__(self, gen, disc, train_loader, test_loader):
        self.gen = gen
        self.disc = disc
        self.train_loader = train_loader
        self.test_loader = test_loader
        
    
    
    def execute(
            self,
            epochs=150,
            adversarial_weight=1,
            content_weight=150,
            verbose=True,
            augmentation_callbacks=None,
            fuzzy_bboxes_func=FuzzyBoundingBoxes(
                x_tolerance=0.05,
                y_tolerance=0.05
            ),
            _data_device_override='cuda',
            **kwargs
        ):
        
        adversarial_criterion = BCELoss()
        content_criterion = BinaryDiceLoss()
        
        if len(kwargs) == 0:
            gen_optim = Adam(self.gen.parameters(), betas=(0.5, 0.999), lr=2e-4)
            disc_optim = Adam(self.disc.parameters(), betas=(0.5, 0.999), lr=2e-4)
        else:
            gen_optim = Adam(self.gen.parameters(), **kwargs)
            disc_optim = Adam(self.disc.parameters(), **kwargs)
            
        train_gen_losses, test_gen_losses = [], []
        train_disc_losses, test_disc_losses = [], []
        
        train_dices, test_dices = [], []
        train_ious, test_ious = [], []
        
        if verbose:
            print('#'*32)
            print('beginning cGAN optimisation loop')
            print('#'*32)
            
        for epoch in range(1, epochs + 1):
            if verbose:
                print(f'--executing epoch {epoch}...')
                
            train_gen_loss, test_gen_loss = [], []
            train_disc_loss, test_disc_loss = [], []
            
            train_dice, test_dice = [], []
            train_iou, test_iou = [], []
            
            self.gen.train()
            self.disc.train()
            
            for batch in self.train_loader:
                data = format_segmentation_rois(batch, fuzzy_bboxes_func)
                
                imgs, masks = data[:, 0], data[:, 1]
                
                if _data_device_override is not None:
                    imgs = imgs.to(_data_device_override)
                    masks = masks.to(_data_device_override)
                    
                gen_optim.zero_grad()
                
                fake_masks = self.gen(imgs)
                disc_preds = self.disc(imgs, fake_masks)
                
                gen_labels = torch.zeros((batch.shape[0],))
                
                adversarial_loss = adversarial_criterion(
                    disc_preds,
                    gen_labels
                )
                
                content_loss = content_criterion(
                    fake_masks,
                    masks
                )
                
                gen_loss = adversarial_loss * adversarial_weight + content_loss * content_weight
                
                gen_loss.backward()
                gen_optim.step()
                
                disc_optim.zero_grad()
                
                real_output = self.disc(imgs, masks)
                fake_output = self.disc(imgs, fake_masks.detach())
                
                real_disc_loss = torch.mean(-torch.log(real_output))
                fake_disc_loss = torch.mean(-torch.log(1 - fake_output))
                
                disc_loss = real_disc_loss + fake_disc_loss
                
                disc_loss.backward()
                disc_optim.step()
                
                mask_ious = batch_mask_ious(masks, fake_masks)
                
                train_gen_loss.append(gen_loss.item() / len(batch))
                train_disc_loss.append(disc_loss.item() / len(batch))
                
                train_dice.append(content_loss.item() / len(batch))
                train_iou.extend(list(mask_ious))
                
            if verbose:
                print(f'evaluated {len(train_gen_losses)} train batches')
                print(f'->avg generator loss: {train_gen_losses[-1]}')
                print(f'->avg discriminator loss: {train_disc_losses[-1]}')
                print('-'*32)
            
            self.gen.eval()
            self.disc.eval()
            
            train_gen_losses.append(sum(train_gen_loss) / len(train_gen_loss))
            train_disc_losses.append(sum(train_disc_loss) / len(train_disc_loss))
            
            train_dices.append(sum(train_dice) / len(train_dice))
            train_ious.append(sum(train_iou) / len(train_iou))
            
            for batch in self.test_loader:
                data = format_segmentation_rois(batch, fuzzy_bboxes_func)
                
                imgs, masks = data[:, 0], data[:, 1]
                
                if _data_device_override is not None:
                    imgs = imgs.to(_data_device_override)
                    masks = masks.to(_data_device_override)
                
                fake_masks = self.gen(imgs)
                disc_preds = self.disc(imgs, fake_masks)
                
                gen_labels = torch.zeros((batch.shape[0],))
                
                adversarial_loss = adversarial_criterion(
                    disc_preds,
                    gen_labels
                )
                
                content_loss = content_criterion(
                    fake_masks,
                    masks
                )
                
                gen_loss = adversarial_loss * adversarial_weight + content_loss * content_weight
                
                real_output = self.disc(imgs, masks)
                fake_output = self.disc(imgs, fake_masks.detach())
                
                real_disc_loss = torch.mean(-torch.log(real_output))
                fake_disc_loss = torch.mean(-torch.log(1 - fake_output))
                
                disc_loss = real_disc_loss + fake_disc_loss
                
                mask_ious = batch_mask_ious(masks, fake_masks)
                
                test_gen_loss.append(gen_loss.item() / len(batch))
                test_disc_loss.append(disc_loss.item() / len(batch))
                
                test_dice.append(content_loss.item() / len(batch))
                test_iou.extend(list(mask_ious))
            
            test_gen_losses.append(sum(test_gen_loss) / len(test_gen_loss))
            test_disc_losses.append(sum(test_disc_loss) / len(test_disc_loss))
            
            test_dices.append(sum(test_dice) / len(test_dice))
            test_ious.append(sum(test_iou) / len(test_iou))
            
            results = {
                'train_gen_losses': train_gen_losses,
                'train_disc_losses': train_disc_losses,
                'train_dices': train_dice,
                'train_ious': train_ious,
                'test_gen_losses': test_gen_losses,
                'test_disc_losses': test_disc_losses,
                'test_dices': test_dices,
                'test_ious': test_ious,
            }
            
            return results