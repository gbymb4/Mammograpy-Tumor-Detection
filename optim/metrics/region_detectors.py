# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:39:11 2022

@author: Gavin
"""

import torch

import numpy as np
'''
def compute_tpr(detections, true_boxes, **kwargs):
    pred_boxes = detections['boxes']
    true_boxes = true_boxes.cpu().detach().tolist()
    
    matches = compare_bounding_boxes(pred_boxes, true_boxes, **kwargs)
    
    true_positives = len(matches)
    
    total_ground_truth = len(true_boxes)
    
    tpr = true_positives / total_ground_truth
    print(f'{tpr=:.4f}%')
    return tpr



def compute_fpr(detections, true_boxes, **kwargs):
    pred_boxes = detections['boxes']
    true_boxes = true_boxes.cpu().detach().tolist()
    
    matches = compare_bounding_boxes(pred_boxes, true_boxes, **kwargs)
    #print(len(pred_boxes), len(true_boxes), len(matches))
    total_predictions = detections['trialed_boxes']
    
    true_positives = len(matches)
    false_positives = total_predictions - true_positives
    
    fpr = false_positives / total_predictions
    print(f'{fpr=:.4f}%')
    return fpr



def compare_bounding_boxes(predictions, ground_truths, intersection_threshold=0.6):
    gt_boxes = np.array(ground_truths)
    pred_boxes = predictions.detach().cpu().numpy()
    
    if all(len(i)==4 for i in predictions) and all(len(i)==4 for i in ground_truths):
        x1 = np.maximum(pred_boxes[:, None, 0], gt_boxes[:, 0])
        y1 = np.maximum(pred_boxes[:, None, 1], gt_boxes[:, 1])
        x2 = np.minimum(pred_boxes[:, None, 2], gt_boxes[:, 2])
        y2 = np.minimum(pred_boxes[:, None, 3], gt_boxes[:, 3])
        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        union_area = pred_area[:, None] + gt_area - inter_area
    
        iou = inter_area / union_area
    
        matched_indices = np.argmax(iou, axis=1)
    
        matched_indices = matched_indices[iou[np.arange(iou.shape[0]), matched_indices] >= 0.6]
    
        matched_predictions = pred_boxes[np.unique(matched_indices)]
        
        return matched_predictions
    
    else:
        return np.array([])



def iou(box1, box2):
    x1 = np.maximum(box1[:, 0], box2[:, 0])
    y1 = np.maximum(box1[:, 1], box2[:, 1])
    x2 = np.minimum(box1[:, 2], box2[:, 2])
    y2 = np.minimum(box1[:, 3], box2[:, 3])

    inter_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)

    box1_area = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou
'''
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
'''


def compute_tpr_ssd(detections, true_boxes, intersection_threshold=0.6):
    pred_boxes = detections['boxes'].detach().cpu()
    true_boxes = true_boxes.detach().cpu()

    num_true_boxes = len(true_boxes)
    
    ious = jaccard_iou(pred_boxes, true_boxes)
    max_iou_idx = ious.argmax(axis=1)
    max_ious = ious[np.arange(len(pred_boxes)), max_iou_idx]

    true_detections = (max_ious >= intersection_threshold).sum()
    tpr = true_detections / num_true_boxes
    
    return tpr


'''
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
'''  


def compute_fpr_ssd(detections, true_boxes, intersection_threshold=0.6):
    pred_boxes = detections['boxes'].detach().cpu()
    true_boxes = true_boxes.cpu()

    negatives = detections['trialed_boxes'] - len(true_boxes)

    ious = jaccard_iou(pred_boxes, true_boxes)
    max_iou_idx = ious.argmax(axis=1)
    max_ious = ious[np.arange(len(pred_boxes)), max_iou_idx]

    pred_boxes = pred_boxes[max_ious < intersection_threshold]
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
