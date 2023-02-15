# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 18:37:27 2023

@author: Gavin
"""

import torch, warnings

from typing import List, Tuple, OrderedDict

from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

from .core import filter_override, roi_align_override, roi_postprocessing_override

class FRCNNMobileNetV3FPN:
    
    def __init__(self, weights='DEFAULT', **kwargs):
        model = fasterrcnn_mobilenet_v3_large_fpn(
            #weights=weights,
            num_classes=2,
            **kwargs
        )
        
        model.rpn.filter_proposals = filter_override(model.rpn)
        model.roi_heads.postprocess_detections = roi_postprocessing_override(model.roi_heads)
        model.roi_heads.box_roi_pool.forward = roi_align_override(model.roi_heads.box_roi_pool)
        
        self.rcnn_model = model
        self.rcnn_model.forward = self.__forward_impl
        
        self.train = model.train
    
    
    
    def __forward_impl(self, images, targets=None):
        # slightly editted forward method from rcnn_model_class
        if self.rcnn_model.training:
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

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.rcnn_model.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.rcnn_model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rcnn_model.rpn(images, features, targets)
        detections, detector_losses = self.rcnn_model.roi_heads(features, proposals, images.image_sizes, targets)
        #print(detections)
        detections = self.rcnn_model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self.rcnn_model._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self.rcnn_model._has_warned = True
            return losses, detections
        else:
            return self.__eager_outputs(losses, detections)
        
    
    
    @torch.jit.unused
    def __eager_outputs(self, losses, detections):
        return losses, detections