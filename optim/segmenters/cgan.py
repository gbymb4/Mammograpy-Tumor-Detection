# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:30:26 2023

@author: Gavin
"""

import torch

import numpy as np

from torch.optim import Adam

from projectio.loading import format_segmentation_rois, FuzzyBoundingBoxes

from optim.criterions import BinaryDiceLoss, DiscriminatorBCE
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
        
        discriminator_criterion = DiscriminatorBCE()
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
            
            for i, batch in enumerate(self.train_loader):
                data = format_segmentation_rois(batch, fuzzy_bboxes_func)
                
                imgs, masks = data[:, 0], data[:, 1]
                
                if augmentation_callbacks is not None:
                    imgs, masks = self.__augment(
                        imgs,
                        masks,
                        augmentation_callbacks
                    )
                
                if _data_device_override is not None:
                    imgs = imgs.to(_data_device_override)
                    masks = masks.to(_data_device_override)
                
                gen_optim.zero_grad()
                
                fake_masks = self.gen(imgs)
                disc_preds = self.disc(imgs, fake_masks)
                
                adversarial_loss = torch.mean(-torch.log(1 - disc_preds))
                
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
                
                disc_loss = discriminator_criterion(real_output, fake_output)
                
                disc_loss.backward()
                disc_optim.step()
                
                mask_ious = batch_mask_ious(masks, fake_masks > 0.5)
                
                train_gen_loss.append(gen_loss.item() / len(batch))
                train_disc_loss.append(disc_loss.item() / len(batch))
                
                train_dice.append(content_loss.item() / len(batch))
                train_iou.extend(list(mask_ious))
            
            self.gen.eval()
            self.disc.eval()
            
            train_gen_losses.append(sum(train_gen_loss) / len(train_gen_loss))
            train_disc_losses.append(sum(train_disc_loss) / len(train_disc_loss))
            
            train_dices.append(sum(train_dice) / len(train_dice))
            train_ious.append(sum(train_iou) / len(train_iou))
            
            for i, batch in enumerate(self.test_loader):
                data = format_segmentation_rois(batch, fuzzy_bboxes_func)
                
                imgs, masks = data[:, 0], data[:, 1]
                
                if _data_device_override is not None:
                    imgs = imgs.to(_data_device_override)
                    masks = masks.to(_data_device_override)
                
                fake_masks = self.gen(imgs)
                disc_preds = self.disc(imgs, fake_masks)
                '''
                import matplotlib.pyplot as plt
                if i == 0:
                    fig, axs = plt.subplots(len(batch), 4, figsize=(40, 10 * len(batch)))
                
                    axs[0][0].set_title('Tumor ROI', fontsize=64)
                    axs[0][1].set_title('Mask Ground Truth', fontsize=64)
                    axs[0][2].set_title('Raw Output', fontsize=64)
                    axs[0][3].set_title('Thresholded Mask', fontsize=64)
                
                    for img, real, fake, ax_row in zip(imgs, masks, fake_masks, axs):
                        ax_row[0].imshow(np.swapaxes(img.detach().cpu().numpy(), 0, 2), cmap='gray')
                        ax_row[1].imshow(np.swapaxes(real.detach().cpu().numpy(), 0, 2), cmap='gray')
                        ax_row[2].imshow(np.swapaxes(fake.detach().cpu().numpy(), 0, 2), cmap='gray')
                        
                        fake_copy = fake.clone() > 0.5
                        
                        ax_row[3].imshow(np.swapaxes(fake_copy.detach().cpu().numpy(), 0, 2), cmap='gray')
                        
                        for ax in ax_row:
                            ax.axis('off')      
                            
                    fig.tight_layout()
                    
                    plt.show()
                '''
                adversarial_loss = torch.mean(-torch.log(1 - disc_preds))
                
                content_loss = content_criterion(
                    fake_masks,
                    masks
                )
                
                gen_loss = adversarial_loss * adversarial_weight + content_loss * content_weight
                
                real_output = self.disc(imgs, masks)
                fake_output = self.disc(imgs, fake_masks.detach())
                
                disc_loss = discriminator_criterion(real_output, fake_output)
                
                mask_ious = batch_mask_ious(masks, fake_masks > 0.5)
                
                test_gen_loss.append(gen_loss.item() / len(batch))
                test_disc_loss.append(disc_loss.item() / len(batch))
                
                test_dice.append(content_loss.item() / len(batch))
                test_iou.extend(list(mask_ious))
            
            test_gen_losses.append(sum(test_gen_loss) / len(test_gen_loss))
            test_disc_losses.append(sum(test_disc_loss) / len(test_disc_loss))
            
            test_dices.append(sum(test_dice) / len(test_dice))
            test_ious.append(sum(test_iou) / len(test_iou))
            
            if verbose:
                print(f'--evaluated {len(train_gen_loss)} train batches')
                print(f'->avg generator loss: train = {train_gen_losses[-1]:.4f}, test = {test_gen_losses[-1]:.4f}')
                print(f'->avg discriminator loss: train = {train_disc_losses[-1]:.4f}, test = {test_disc_losses[-1]:.4f}')
                print(f'->avg IoU: train = {train_ious[-1]:.4f}, test = {test_ious[-1]:.4f}')
                print(f'->avg DICE: train = {train_dices[-1]:.4f}, test = {test_dices[-1]:.4f}')
                print('-'*32)
                
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
    
    
    
    def __augment(self, imgs, masks, callbacks):
        new_imgs, new_masks = [], []
        
        device = imgs[0].device
        
        for img, mask in zip(imgs, masks):
            temp_img = img.cpu().numpy()
            temp_mask = mask.cpu().numpy()
            
            temp = np.concatenate((temp_img, temp_mask), axis=0)
            temp = np.swapaxes(temp, 0, 2)
            
            for callback in callbacks:
                temp, _ = callback(temp, np.ones((1, 4)))
                
                if len(temp.shape) != 3:
                    temp = temp[:temp[2], :temp[1], np.newaxis]
                
            temp = np.swapaxes(temp, 0, 2) 
            
            temp_img, temp_mask = temp
                
            new_imgs.append(temp_img)
            new_masks.append(temp_mask)
        
        new_imgs = np.array(new_imgs)
        new_masks = np.array(new_masks)
        
        new_imgs = torch.from_numpy(new_imgs.copy()).float().to(device)
        new_masks = torch.from_numpy(new_masks.copy()).float().to(device)
            
        return new_imgs, new_masks