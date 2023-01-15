# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 18:41:15 2023

@author: Gavin
"""

import torch

from torch import nn

class GeneratorConvBlock(nn.Module):
    
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel, 
            stride
        ):
        
        a = nn.LeakyReLU()
        cn = nn.Conv2d(in_channels, out_channels, kernel, stride)
        bn = nn.BatchNorm2d(out_channels)
        
        self.features = nn.Sequential(a, cn, bn)
        
        
        
    def forward(self, x):
        return self.features(x)
    
    
    
class GeneratorDeconvBlock(nn.Module):
    
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel, 
            stride, 
            use_dropout=True
        ):
    
        a = nn.ReLU()
        dn = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride)
        bn = nn.BatchNorm2d(out_channels)
        
        if use_dropout:
            do = nn.Dropout2d()
            
            self.features = nn.Sequential(a, dn, bn, do)
            
        else:
            self.features = nn.Sequential(a, dn, bn)
           
            
           
        def forward(self, x):
            return self.features(x)
        
    

class CGANGenerator(nn.Module):
    
    def __init__(self):
        kernel = (4, 4)
        stride = (2, 2)
        
        self.cn1 = nn.Conv2d(1, 32, kernel, stride)
        
        self.gcb2 = GeneratorConvBlock(32, 64, kernel, stride)
        self.gcb3 = GeneratorConvBlock(64, 128, kernel, stride)
        self.gcb4 = GeneratorConvBlock(128, 256, kernel, stride)
        self.gcb5 = GeneratorConvBlock(256, 256, kernel, stride)
        self.gcb6 = GeneratorConvBlock(256, 256, kernel, stride)
        self.gcb7 = GeneratorConvBlock(256, 256, kernel, stride)
        
        self.ac8 = nn.LeakyReLU()
        self.cn8 = nn.Conv2d(256, 256, kernel, stride)
        
        self.gdb1 = GeneratorDeconvBlock(256, 256, kernel, stride)
        self.gdb2 = GeneratorDeconvBlock(512, 256, kernel, stride)
        self.gdb3 = GeneratorDeconvBlock(512, 256, kernel, stride)
        self.gdb4 = GeneratorDeconvBlock(512, 256, kernel, stride, False)
        self.gdb5 = GeneratorDeconvBlock(512, 128, kernel, stride, False)
        self.gdb6 = GeneratorDeconvBlock(256, 64, kernel, stride, False)
        self.gdb7 = GeneratorDeconvBlock(128, 32, kernel, stride, False)
        
        self.ad8 = nn.ReLU()
        self.dn8 = nn.ConvTranspose2d(64, 1, kernel, stride)
        
        self.tan = nn.Tanh()
        
    
    def forward(self, x):
        out1 = self.cn1(x)
        
        out2 = self.gcb2(out1)
        out3 = self.gcb3(out2)
        out4 = self.gcb4(out3)
        out5 = self.gcb5(out4)
        out6 = self.gcb6(out5)
        out7 = self.gcb7(out6)

        out8 = self.cn8(self.ac8(out7))  
        
        out9 = torch.cat((self.gdb1(out8), out7))
        out10 = torch.cat((self.gdb2(out9), out6))
        out11 = torch.cat((self.gdb3(out10), out5))
        out12 = torch.cat((self.gdb4(out11), out4))
        out13 = torch.cat((self.gdb5(out12), out3))
        out14 = torch.cat((self.gdb6(out13), out2))
        out15 = torch.cat((self.gdb7(out14), out1))

        out16 = self.dn8(self.ad8(out15))
        
        mask = self.tan(out16)
        
        return mask
    
    

class CGANDiscriminator(nn.Module):
    
    def __init__(self):
        kernel = (4, 4)
        stride = (2, 2)
        
        cn1 = nn.Conv2d(1, 32, kernel, stride)
        
        a2 = nn.LeakyReLU()
        cn2 = nn.Conv2d(32, 64, kernel, stride)
        bn2 = nn.BatchNorm2d(64)
        
        a3 = nn.LeakyReLU()
        cn3 = nn.Conv2d(64, 128, kernel, stride)
        bn3 = nn.BatchNorm2d(128)
        
        a4 = nn.LeakyReLU()
        cn4 = nn.Conv2d(128, 256, kernel, stride)
        bn4 = nn.BatchNorm2d(256)
        
        a5 = nn.LeakyReLU()
        cn5 = nn.Conv2d(256, 1, kernel, stride)
        
        sig = nn.Sigmoid()
        
        self.features = nn.Sequential(
            cn1,
            a2, cn2, bn2,
            a3, cn3, bn3,
            a4, cn4, bn4,
            a5, cn5,
            sig
        )
        
        
    
    def forward(self, x):
        return self.features(x)