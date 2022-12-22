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
        device='cuda',
        ranking_metric=None,
    ):
    
    best_model = None
    best_score = float('-inf')
    
    results = []
    for fold in folds:
        train, test = fold
        
        model = model_cls(**model_kwargs).to(device)
        
        optim = optim_cls(model, train, test)
        stats = optim.execute(**optim_exec)
        
        score = ranking_metric(stats)
        
        if score > best_score:
            best_model = model
        
        results.append(stats)
        
    results = __avg_cv_metrics(results)
    
    return best_model, results
    
    
    
def plr_ranking(hist_dict):
    tpr = hist_dict['test_tpr'][-1]
    fpr = hist_dict['test_fpr'][-1]
    
    plr = tpr / fpr
    
    return plr
    

    
def __avg_cv_metrics(results_list):
    keys = results_list[0].keys()
    
    avg_results_dict = {}
    for key in keys:
        metric_scores = [results_list[i][key] for i in range(len(results_list))]
        metric_scores = np.array(metric_scores)
        
        avg_scores = metric_scores.mean(axis=0)
        
        avg_results_dict[key] = avg_scores
        
    return avg_results_dict



