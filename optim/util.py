# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:45:59 2022

@author: Gavin
"""

import numpy as np

def exec_optim_on_folds(
        model_cls,
        model_kwargs,
        optim_cls,
        optim_exec,
        folds,
        device='cuda'
    ):
    
    results = []
    for fold in folds:
        train, test = fold
        
        model = model_cls(**model_kwargs).to(device)
        
        optim = optim_cls(model, train, test)
        stats = optim.execute(**optim_exec)
        
        results.append(stats)
        
    results = __avg_cv_metrics(results)
    
    return results
    
    
    
def __avg_cv_metrics(results_list):
    keys = results_list[0].keys()
    
    avg_results_dict = {}
    for key in keys:
        metric_scores = [results_list[i][key] for i in range(len(results_list))]
        metric_scores = np.array(metric_scores)
        
        avg_scores = metric_scores.mean(axis=1)
        
        avg_results_dict[key] = avg_scores
        
    return avg_results_dict