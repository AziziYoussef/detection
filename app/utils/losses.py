"""
Loss Functions for Lost Objects Detection Model Training
Implements Focal Loss, IoU Loss, and other specialized loss functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Paper: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0, 
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            inputs: Predictions [N, C] or [N, C, H, W]
            targets: Ground truth labels [N] or [N, H, W]
            
        Returns:
            Focal loss value
        """
        # Flatten if needed
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # [N, C, H*W]
            inputs = inputs.transpose(1, 2)    # [N, H*W, C]
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # [N*H*W, C]
        
        if targets.dim() > 1:
            targets = targets.view(-1)  # [N*H*W]
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            num_classes = inputs.size(1)
            targets_one_hot = F.one_hot(targets, num_classes).float()
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                             self.label_smoothing / num_classes
        else:
            targets_one_hot = F.one_hot(targets, inputs.size(1)).float()
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute p_t
        pt = torch.exp(-ce_loss)
        
        # Compute alpha term
        alpha_t = self.alpha * targets_one_hot + (1 - self.alpha) * (1 - targets_one_hot)
        alpha_t = alpha_t.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class IoULoss(nn.Module):
    """
    IoU Loss for bounding box regression
    Supports different IoU variants: IoU, GIoU, DIoU, CIoU
    """
    
    def __init__(self, loss_type: str = 'iou', eps: float = 1e-7):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.eps = eps
        
        assert self.loss_type in ['iou', 'giou', 'diou', 'ciou'], \
            f"Unsupported IoU loss type: {loss_type}"
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU-based loss
        
        Args:
            pred_boxes: Predicted boxes [N, 4] (x1, y1, x2, y2)
            target_boxes: Target boxes [N, 4] (x1, y1, x2, y2)
            
        Returns:
            IoU loss value
        """
        if self.loss_type == 'iou':
            iou = self._compute_iou(pred_boxes, target_boxes)
            return 1 - iou.mean()
        
        elif self.loss_type == 'giou':
            giou = self._compute_giou(pred_boxes, target_boxes)
            return 1 - giou.mean()
        
        elif self.loss_type == 'diou':
            diou = self._compute_diou(pred_boxes, target_boxes)
            return 1 - diou.mean()
        
        elif self.loss_type == 'ciou':
            ciou = self._compute_ciou(pred_boxes, target_boxes)
            return 1 - ciou.mean()
    
    def _compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute standard IoU"""
        # Intersection area
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Union area
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - intersection
        
        return intersection / (union + self.eps)
    
    def _compute_giou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute Generalized IoU"""
        iou = self._compute_iou(boxes1, boxes2)
        
        # Enclosing box
        x1_c = torch.min(boxes1[:, 0], boxes2[:, 0])
        y1_c = torch.min(boxes1[:, 1], boxes2[:, 1])
        x2_c = torch.max(boxes1[:, 2], boxes2[:, 2])
        y2_c = torch.max(boxes1[:, 3], boxes2[:, 3])
        
        enclosing_area = (x2_c - x1_c) * (y2_c - y1_c)
        
        # Union area
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - iou * area1  # iou * area1 is intersection
        
        return iou - (enclosing_area - union) / (enclosing_area + self.eps)
    
    def _compute_diou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute Distance IoU"""
        iou = self._compute_iou(boxes1, boxes2)
        
        # Centers
        cx1 = (boxes1[:, 0] + boxes1[:, 2]) / 2
        cy1 = (boxes1[:, 1] + boxes1[:, 3]) / 2
        cx2 = (boxes2[:, 0] + boxes2[:, 2]) / 2
        cy2 = (boxes2[:, 1] + boxes2[:, 3]) / 2
        
        # Distance between centers
        center_distance = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
        
        # Diagonal of enclosing box
        x1_c = torch.min(boxes1[:, 0], boxes2[:, 0])
        y1_c = torch.min(boxes1[:, 1], boxes2[:, 1])
        x2_c = torch.max(boxes1[:, 2], boxes2[:, 2])
        y2_c = torch.max(boxes1[:, 3], boxes2[:, 3])
        
        diagonal_distance = (x2_c - x1_c) ** 2 + (y2_c - y1_c) ** 2
        
        return iou - center_distance / (diagonal_distance + self.eps)
    
    def _compute_ciou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute Complete IoU"""
        diou = self._compute_diou(boxes1, boxes2)
        
        # Aspect ratio consistency
        w1 = boxes1[:, 2] - boxes1[:, 0]
        h1 = boxes1[:, 3] - boxes1[:, 1]
        w2 = boxes2[:, 2] - boxes2[:, 0]
        h2 = boxes2[:, 3] - boxes2[:, 1]
        
        v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + self.eps)) - 
                                          torch.atan(w1 / (h1 + self.eps)), 2)
        
        iou = self._compute_iou(boxes1, boxes2)
        alpha = v / (1 - iou + v + self.eps)
        
        return diou - alpha * v

class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss for bounding box regression
    Less sensitive to outliers than L1 loss
    """
    
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute smooth L1 loss
        
        Args:
            input: Predicted values
            target: Target values
            
        Returns:
            Smooth L1 loss
        """
        diff = torch.abs(input - target)
        
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss from FCOS
    Combines classification and localization quality
    """
    
    def __init__(self, beta: float = 2.0):
        super().__init__()
        self.beta = beta
    
    def forward(
        self, 
        pred_scores: torch.Tensor, 
        target_scores: torch.Tensor,
        target_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quality focal loss
        
        Args:
            pred_scores: Predicted classification scores
            target_scores: Target quality scores (e.g., IoU)
            target_labels: Target class labels
            
        Returns:
            Quality focal loss
        """
        # Only compute loss for positive samples
        pos_mask = target_labels > 0
        
        if pos_mask.sum() == 0:
            return pred_scores.sum() * 0  # Return zero loss
        
        pos_pred_scores = pred_scores[pos_mask]
        pos_target_scores = target_scores[pos_mask]
        pos_target_labels = target_labels[pos_mask]
        
        # Compute sigmoid focal loss with quality targets
        sigmoid_pred = torch.sigmoid(pos_pred_scores)
        
        # Create one-hot targets weighted by quality scores
        one_hot = F.one_hot(pos_target_labels.long(), pred_scores.size(1)).float()
        quality_targets = one_hot * pos_target_scores.unsqueeze(1)
        
        # Focal weight
        pt = sigmoid_pred * quality_targets + (1 - sigmoid_pred) * (1 - quality_targets)
        focal_weight = torch.pow(1 - pt, self.beta)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(
            pos_pred_scores, quality_targets, reduction='none'
        )
        
        loss = focal_weight * bce
        return loss.sum() / max(pos_mask.sum(), 1)

class DetectionLoss(nn.Module):
    """
    Combined loss for object detection
    Integrates classification and regression losses
    """
    
    def __init__(
        self,
        num_classes: int,
        classification_loss: str = 'focal',
        regression_loss: str = 'iou',
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        
        # Classification loss
        if classification_loss == 'focal':
            self.cls_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif classification_loss == 'ce':
            self.cls_loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported classification loss: {classification_loss}")
        
        # Regression loss
        if regression_loss == 'iou':
            self.reg_loss_fn = IoULoss(loss_type='iou')
        elif regression_loss == 'giou':
            self.reg_loss_fn = IoULoss(loss_type='giou')
        elif regression_loss == 'smooth_l1':
            self.reg_loss_fn = SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported regression loss: {regression_loss}")
    
    def forward(
        self,
        predictions: dict,
        targets: list,
        anchors: torch.Tensor
    ) -> dict:
        """
        Compute combined detection loss
        
        Args:
            predictions: Model predictions containing 'classification' and 'regression'
            targets: List of target dictionaries with 'boxes' and 'labels'
            anchors: Anchor boxes
            
        Returns:
            Dictionary with individual and total losses
        """
        cls_pred = predictions['classification']
        reg_pred = predictions['regression']
        
        # Prepare targets for loss computation
        cls_targets, reg_targets, pos_mask = self._prepare_targets(anchors, targets)
        
        # Classification loss
        cls_loss = self.cls_loss_fn(cls_pred, cls_targets)
        
        # Regression loss (only on positive samples)
        if pos_mask.sum() > 0:
            pos_reg_pred = reg_pred[pos_mask]
            pos_reg_targets = reg_targets[pos_mask]
            
            # Decode predictions if needed (for IoU-based losses)
            if isinstance(self.reg_loss_fn, IoULoss):
                pos_anchors = anchors[pos_mask]
                pos_pred_boxes = self._decode_boxes(pos_reg_pred, pos_anchors)
                reg_loss = self.reg_loss_fn(pos_pred_boxes, pos_reg_targets)
            else:
                reg_loss = self.reg_loss_fn(pos_reg_pred, pos_reg_targets)
        else:
            reg_loss = torch.tensor(0.0, device=cls_pred.device, requires_grad=True)
        
        # Total loss
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        
        return {
            'classification_loss': cls_loss,
            'regression_loss': reg_loss,
            'total_loss': total_loss,
            'loss_stats': {
                'num_positive_anchors': pos_mask.sum().item(),
                'cls_loss_value': cls_loss.item(),
                'reg_loss_value': reg_loss.item()
            }
        }
    
    def _prepare_targets(
        self, 
        anchors: torch.Tensor, 
        targets: list
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare targets for loss computation"""
        from ..utils.box_utils import box_iou
        
        batch_size = len(targets)
        num_anchors = anchors.shape[0]
        
        cls_targets = []
        reg_targets = []
        pos_masks = []
        
        for batch_idx in range(batch_size):
            target = targets[batch_idx]
            gt_boxes = target['boxes']
            gt_labels = target['labels']
            
            # Initialize targets
            cls_target = torch.zeros(num_anchors, dtype=torch.long, device=anchors.device)
            reg_target = torch.zeros(num_anchors, 4, device=anchors.device)
            
            if len(gt_boxes) > 0:
                # Compute IoU between anchors and ground truth
                ious = box_iou(anchors, gt_boxes)
                max_iou, matched_gt_idx = ious.max(dim=1)
                
                # Positive anchors (IoU > 0.5)
                pos_mask = max_iou > 0.5
                cls_target[pos_mask] = gt_labels[matched_gt_idx[pos_mask]]
                
                # Encode regression targets
                if pos_mask.sum() > 0:
                    matched_boxes = gt_boxes[matched_gt_idx[pos_mask]]
                    reg_target[pos_mask] = self._encode_boxes(
                        anchors[pos_mask], matched_boxes
                    )
            else:
                pos_mask = torch.zeros(num_anchors, dtype=torch.bool, device=anchors.device)
            
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            pos_masks.append(pos_mask)
        
        return (
            torch.stack(cls_targets),
            torch.stack(reg_targets),
            torch.stack(pos_masks)
        )
    
    def _encode_boxes(self, anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """Encode ground truth boxes relative to anchors"""
        # Convert to center format
        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights
        
        gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
        gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights
        
        # Encode
        dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
        dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights
        dw = torch.log(gt_widths / anchor_widths)
        dh = torch.log(gt_heights / anchor_heights)
        
        return torch.stack([dx, dy, dw, dh], dim=1)
    
    def _decode_boxes(self, deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode box predictions to absolute coordinates"""
        # Convert anchors to center format
        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights
        
        # Apply deltas
        dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
        
        pred_ctr_x = dx * anchor_widths + anchor_ctr_x
        pred_ctr_y = dy * anchor_heights + anchor_ctr_y
        pred_w = torch.exp(dw) * anchor_widths
        pred_h = torch.exp(dh) * anchor_heights
        
        # Convert back to corner format
        pred_x1 = pred_ctr_x - 0.5 * pred_w
        pred_y1 = pred_ctr_y - 0.5 * pred_h
        pred_x2 = pred_ctr_x + 0.5 * pred_w
        pred_y2 = pred_ctr_y + 0.5 * pred_h
        
        return torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)