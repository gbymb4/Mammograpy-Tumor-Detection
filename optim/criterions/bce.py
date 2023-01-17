# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 08:20:56 2023

@author: Gavin
"""

import torch

from torch import nn

class DiscriminatorBCE(nn.Module):
    
    def forward(self, real_output, fake_output):
        real_disc_loss = torch.mean(-torch.log(real_output))
        fake_disc_loss = torch.mean(-torch.log(1 - fake_output))
        
        disc_loss = real_disc_loss + fake_disc_loss
        
        return disc_loss