# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:40:59 2022

@author: Gavin
"""

def compute_batch_metric(metric_func, batch):
    return [metric_func(*batch_elem) for batch_elem in batch]