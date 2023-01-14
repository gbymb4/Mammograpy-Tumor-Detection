# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:39:11 2022

@author: Gavin
"""

#import torch

import numpy as np

def compute_tpr(detections, true_boxes, **kwargs):
    pred_boxes = detections['boxes']
    true_boxes = true_boxes.cpu().detach().tolist()
    
    matches = compare_bounding_boxes(pred_boxes, true_boxes, **kwargs)
    
    true_positives = len(matches)
    
    total_ground_truth = len(true_boxes)
    
    tpr = true_positives / total_ground_truth
    
    return tpr



def compute_fpr(detections, true_boxes, **kwargs):
    pred_boxes = detections['boxes']
    true_boxes = true_boxes.cpu().detach().tolist()
    
    matches = compare_bounding_boxes(pred_boxes, true_boxes, **kwargs)
    
    total_predictions = detections['trialed_boxes']
    
    true_positives = len(matches)
    false_positives = total_predictions - true_positives
    
    fpr = false_positives / total_predictions
    
    return fpr



def compare_bounding_boxes(predictions, ground_truth, intersection_threshold=0.6):
    gt_boxes = np.array(ground_truth)
    pred_boxes = np.array(predictions)
    
    x1, y1, x2, y2 = np.split(gt_boxes, 4, axis=-1)
    
    gt_area = (x2-x1)*(y2-y1)
    
    x1, y1, x2, y2 = np.split(pred_boxes, 4, axis=-1)
    
    x1 = np.maximum(x1, np.transpose(x1))
    y1 = np.maximum(y1, np.transpose(y1))
    x2 = np.minimum(x2, np.transpose(x2))
    y2 = np.minimum(y2, np.transpose(y2))
    
    width_i = np.maximum(x2 - x1, 0)
    height_i = np.maximum(y2 - y1, 0)
    
    area_i = width_i * height_i
    area_i = np.where(area_i > 0, area_i, np.inf)
    area_i = np.min(area_i, axis=1)
    
    matches = np.where(area_i/gt_area >= intersection_threshold)
    
    return matches

'''
def compute_tpr(detections, true_boxes, intersection_threshold=0.6):
    pred_boxes = detections['boxes']
    true_boxes = true_boxes.cpu().detach().tolist()
    
    num_true_boxes = len(true_boxes)
    
    true_detections = 0
    for pred_box in pred_boxes:
        iou_with_trues = []
        
        for true_box in true_boxes:
            iou = jaccard_iou(
                pred_box.unsqueeze(0).detach(),
                torch.Tensor(true_box).unsqueeze(0).detach()
            )

            iou_with_trues.append(iou)
            
        iou_with_trues = np.array(iou_with_trues)
        
        if len(iou_with_trues) == 0:
            break
        
        idx_max = iou_with_trues.argmax()
        max_iou = iou_with_trues[idx_max]
        
        if max_iou >= intersection_threshold:
            del true_boxes[idx_max]
            
            true_detections += 1
            
    tpr = true_detections / num_true_boxes
    
    return tpr



def compute_fpr(detections, true_boxes, intersection_threshold=0.6):
    pred_boxes = detections['boxes'].cpu().detach().tolist()
    true_boxes = true_boxes.cpu().detach().tolist()
    
    negatives = detections['trialed_boxes'] - len(true_boxes)
    
    for true_box in true_boxes:
        iou_with_preds = []
        
        for pred_box in pred_boxes:
            iou = jaccard_iou(
                torch.Tensor(pred_box).unsqueeze(0).detach(),
                torch.Tensor(true_box).unsqueeze(0).detach()
            )

            iou_with_preds.append(iou)
            
        iou_with_preds = np.array(iou_with_preds)
        
        if len(iou_with_preds) == 0:
            break
        
        idx_max = iou_with_preds.argmax()
        max_iou = iou_with_preds[idx_max]
        
        if max_iou >= intersection_threshold:
            del pred_boxes[idx_max]
    
    false_positives = len(pred_boxes)
    
    fpr = false_positives / negatives
    
    return fpr
    


# sourced from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py
def bbox_intersection(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]



# sourced from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py
def jaccard_iou(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = bbox_intersection(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
'''