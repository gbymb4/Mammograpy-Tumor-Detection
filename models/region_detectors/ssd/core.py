# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:39:24 2022

@author: Gavin
"""

import torch, warnings

from abc import ABC
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional

from torch import nn, Tensor
from torchvision.ops import box_iou, clip_boxes_to_image, batched_nms

import torch.nn.functional as F

from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator

def bounding_box_generator(num_feature_maps, aspect_ratios=None, **kwargs):
    if aspect_ratios is None:
        aspect_ratios = [[2]] * num_feature_maps
    
    boxes = DefaultBoxGenerator(aspect_ratios, **kwargs)
    
    return boxes



class GeneralSSD(SSD, ABC):
    
    def __init__(
            self, backbone, size, 
            anchor_generator=None,
            num_feature_maps=None,
            **kwargs
        ):
        if num_feature_maps is None:
            raise ValueError(f'num_feature_maps not assigned value in {type(self)} implementation')
        
        self.num_feature_maps = num_feature_maps
        
        if anchor_generator is None:
            anchor_generator = bounding_box_generator(self.num_feature_maps)
        
        super().__init__(backbone, anchor_generator, size, 2, **kwargs) 
        
        
        
    def eager_outputs(self, losses, detections):
        # override superclass function to analyse detections during training
        return losses, detections
    
    
    
    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        # small edit to superclass forward method to access detections during training
        # can now also access losses during inference
        
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        
        matched_idxs = []
        if targets is None:...
            #torch._assert(False, "targets should not be none when in training mode")
        else:
            for anchors_per_image, targets_per_image in zip(anchors, targets):
                if targets_per_image["boxes"].numel() == 0:
                    matched_idxs.append(
                        torch.full(
                            (anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device
                        )
                    )
                    continue

                match_quality_matrix = box_iou(targets_per_image["boxes"], anchors_per_image)
                matched_idxs.append(self.proposal_matcher(match_quality_matrix))

            losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)

        detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("SSD always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        
        return self.eager_outputs(losses, detections)
    
    
    
    def postprocess_detections(
        self, head_outputs: Dict[str, Tensor], image_anchors: List[Tensor], image_shapes: List[Tuple[int, int]]
    ) -> List[Dict[str, Tensor]]:
        # send image tensors to cpu for postprocessing due to NotImplementedError for gpu
        
        bbox_regression = head_outputs["bbox_regression"]
        pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1)

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections: List[Dict[str, Tensor]] = []

        for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
            boxes = self.box_coder.decode_single(boxes, anchors)
            boxes = clip_boxes_to_image(boxes, image_shape)

            image_boxes = []
            image_scores = []
            image_labels = []
            for label in range(1, num_classes):
                score = scores[:, label]

                keep_idxs = score > self.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # keep only topk scoring predictions
                num_topk = _topk_min(score, self.topk_candidates, 0)
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))

            image_boxes = torch.cat(image_boxes, dim=0).to('cpu')
            image_scores = torch.cat(image_scores, dim=0).to('cpu')
            image_labels = torch.cat(image_labels, dim=0).to('cpu')
            
            # non-maximum suppression
            keep = batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                    "trialed_boxes": head_outputs['cls_logits'].shape[1]
                }
            )
        return detections
    
    
    
class GeneralSSDFeatureExtractor(nn.Module, ABC):
    
    def forward(self, x):
        x = self.features(x)
        rescaled = self.scale_weight.view(1, -1, 1, 1) * F.normalize(x)
        output = [rescaled]

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])
    


@torch.jit.unused
def _fake_cast_onnx(v: Tensor) -> int:
    # copied from inaccessible torch source code
    return v  # type: ignore[return-value]


def _topk_min(input: Tensor, orig_kval: int, axis: int) -> int:
    """
    ONNX spec requires the k-value to be less than or equal to the number of inputs along
    provided dim. Certain models use the number of elements along a particular axis instead of K
    if K exceeds the number of elements along that axis. Previously, python's min() function was
    used to determine whether to use the provided k-value or the specified dim axis value.

    However in cases where the model is being exported in tracing mode, python min() is
    static causing the model to be traced incorrectly and eventually fail at the topk node.
    In order to avoid this situation, in tracing mode, torch.min() is used instead.

    Args:
        input (Tensor): The orignal input tensor.
        orig_kval (int): The provided k-value.
        axis(int): Axis along which we retreive the input size.

    Returns:
        min_kval (int): Appropriately selected k-value.
    """
    # copied from inaccessible torch source code
    
    if not torch.jit.is_tracing():
        return min(orig_kval, input.size(axis))
    axis_dim_val = torch._shape_as_tensor(input)[axis].unsqueeze(0)
    min_kval = torch.min(torch.cat((torch.tensor([orig_kval], dtype=axis_dim_val.dtype), axis_dim_val), 0))
    return _fake_cast_onnx(min_kval)