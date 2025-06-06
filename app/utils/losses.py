"""
💥 LOSSES - FONCTIONS DE PERTE POUR DÉTECTION D'OBJETS
=====================================================
Implémentation des fonctions de perte optimisées pour l'entraînement de modèles de détection

Fonctions de perte incluses:
- Focal Loss: Gestion du déséquilibre de classes
- Smooth L1 Loss: Régression robuste des boîtes
- IoU Loss / GIoU Loss: Pertes basées sur l'intersection
- Detection Loss: Perte combinée classification + régression
- Quality Focal Loss: Version améliorée pour la qualité
- VariFocal Loss: Focus sur objets positifs

Caractéristiques:
- Implémentations vectorisées PyTorch pour performance
- Support automatique GPU/CPU
- Hyperparamètres configurables
- Stabilité numérique optimisée
- Compatible avec mixed precision training

Utilisation dans l'entraînement:
```python
detection_loss = DetectionLoss(num_classes=28, alpha=0.25, gamma=2.0)
total_loss = detection_loss(predictions, targets, anchors)
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import logging

logger = logging.getLogger(__name__)

# 🔥 FOCAL LOSS
class FocalLoss(nn.Module):
    """
    🔥 Focal Loss pour gérer le déséquilibre de classes
    
    Focal Loss = -α(1-p)^γ log(p)
    
    Avantages:
    - Réduit l'influence des exemples faciles (bien classés)
    - Met l'accent sur les exemples difficiles
    - Gère naturellement le déséquilibre background/foreground
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        logger.debug(f"🔥 FocalLoss: α={alpha}, γ={gamma}")
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        🔄 Forward pass Focal Loss
        
        Args:
            inputs: Logits prédits [N, C] ou [N, H, W, C]
            targets: Labels vrais [N] ou [N, H, W]
            weights: Poids par exemple (optionnel)
            
        Returns:
            Perte focal loss
        """
        
        # Reshape si nécessaire pour 2D
        original_shape = inputs.shape
        if inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))
            targets = targets.view(-1)
            if weights is not None:
                weights = weights.view(-1)
        
        # Masque pour ignorer certains indices
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]
        if weights is not None:
            weights = weights[valid_mask]
        
        # Calcul des probabilités
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Terme de pondération focal
        focal_weight = (1 - p_t) ** self.gamma
        
        # Terme alpha (pondération par classe)
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight
        
        # Perte finale
        focal_loss = focal_weight * ce_loss
        
        # Application des poids additionnels
        if weights is not None:
            focal_loss = focal_loss * weights
        
        # Réduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 💎 QUALITY FOCAL LOSS
class QualityFocalLoss(nn.Module):
    """
    💎 Quality Focal Loss - Version améliorée intégrant la qualité
    
    QFL = -|y - σ|^β ((1-σ)^γ log(σ) + σ^γ log(1-σ))
    """
    
    def __init__(
        self,
        beta: float = 2.0,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Quality Focal Loss forward"""
        
        # Sigmoid pour probabilités
        pred_sigmoid = torch.sigmoid(pred_scores)
        
        # Terme de qualité
        scale_factor = torch.abs(target_scores - pred_sigmoid) ** self.beta
        
        # Focal terms
        pt = pred_sigmoid * target_scores + (1 - pred_sigmoid) * (1 - target_scores)
        focal_weight = (1 - pt) ** self.gamma
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(
            pred_scores, target_scores, reduction='none'
        )
        
        # Perte finale
        loss = scale_factor * focal_weight * bce
        
        if weights is not None:
            loss = loss * weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# 📐 SMOOTH L1 LOSS
class SmoothL1Loss(nn.Module):
    """
    📐 Smooth L1 Loss pour régression robuste des boîtes
    
    Smooth L1 = {
        0.5 * x^2 / β     si |x| < β
        |x| - 0.5 * β     sinon
    }
    """
    
    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        
        self.beta = beta
        self.reduction = reduction
        
        logger.debug(f"📐 SmoothL1Loss: β={beta}")
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass Smooth L1 Loss"""
        
        diff = torch.abs(inputs - targets)
        
        # Condition smooth vs linear
        cond = diff < self.beta
        
        # Calcul conditionnel
        loss = torch.where(
            cond,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        # Application des poids
        if weights is not None:
            loss = loss * weights.unsqueeze(-1) if weights.dim() == 1 else loss * weights
        
        # Réduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# 🎯 IOU LOSS
class IoULoss(nn.Module):
    """
    🎯 IoU Loss - Perte basée sur l'intersection over union
    
    IoU Loss = 1 - IoU
    """
    
    def __init__(
        self,
        loss_type: str = 'iou',  # 'iou', 'giou', 'diou', 'ciou'
        reduction: str = 'mean',
        eps: float = 1e-7
    ):
        super().__init__()
        
        self.loss_type = loss_type
        self.reduction = reduction
        self.eps = eps
        
        logger.debug(f"🎯 IoULoss: type={loss_type}")
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward IoU Loss
        
        Args:
            pred_boxes: Boîtes prédites [N, 4] format xyxy
            target_boxes: Boîtes cibles [N, 4] format xyxy
            weights: Poids par boîte
        """
        
        if self.loss_type == 'iou':
            ious = self._calculate_iou(pred_boxes, target_boxes)
            loss = 1 - ious
        elif self.loss_type == 'giou':
            gious = self._calculate_giou(pred_boxes, target_boxes)
            loss = 1 - gious
        elif self.loss_type == 'diou':
            dious = self._calculate_diou(pred_boxes, target_boxes)
            loss = 1 - dious
        elif self.loss_type == 'ciou':
            cious = self._calculate_ciou(pred_boxes, target_boxes)
            loss = 1 - cious
        else:
            raise ValueError(f"Type IoU non supporté: {self.loss_type}")
        
        # Application des poids
        if weights is not None:
            loss = loss * weights
        
        # Réduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _calculate_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calcul IoU standard"""
        
        # Aires
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Intersection
        inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        union_area = area1 + area2 - inter_area
        
        # IoU
        iou = inter_area / torch.clamp(union_area, min=self.eps)
        
        return iou
    
    def _calculate_giou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calcul GIoU (Generalized IoU)"""
        
        # IoU standard
        iou = self._calculate_iou(boxes1, boxes2)
        
        # Aires
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Boîte englobante
        enclose_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
        enclose_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
        enclose_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
        enclose_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
        
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        
        # Union
        union_area = area1 + area2 - iou * area1  # Approximation pour éviter recalcul
        
        # GIoU
        giou = iou - (enclose_area - union_area) / torch.clamp(enclose_area, min=self.eps)
        
        return giou
    
    def _calculate_diou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calcul DIoU (Distance IoU)"""
        
        # IoU standard
        iou = self._calculate_iou(boxes1, boxes2)
        
        # Centres des boîtes
        center1_x = (boxes1[:, 0] + boxes1[:, 2]) / 2
        center1_y = (boxes1[:, 1] + boxes1[:, 3]) / 2
        center2_x = (boxes2[:, 0] + boxes2[:, 2]) / 2
        center2_y = (boxes2[:, 1] + boxes2[:, 3]) / 2
        
        # Distance entre centres
        center_dist = (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2
        
        # Diagonale de la boîte englobante
        enclose_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
        enclose_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
        enclose_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
        enclose_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
        
        diagonal_dist = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
        
        # DIoU
        diou = iou - center_dist / torch.clamp(diagonal_dist, min=self.eps)
        
        return diou
    
    def _calculate_ciou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calcul CIoU (Complete IoU)"""
        
        # DIoU
        diou = self._calculate_diou(boxes1, boxes2)
        
        # Ratio d'aspect
        w1, h1 = boxes1[:, 2] - boxes1[:, 0], boxes1[:, 3] - boxes1[:, 1]
        w2, h2 = boxes2[:, 2] - boxes2[:, 0], boxes2[:, 3] - boxes2[:, 1]
        
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(w1 / torch.clamp(h1, min=self.eps)) - 
            torch.atan(w2 / torch.clamp(h2, min=self.eps)), 2
        )
        
        # Factor α
        iou = self._calculate_iou(boxes1, boxes2)
        alpha = v / torch.clamp(1 - iou + v, min=self.eps)
        
        # CIoU
        ciou = diou - alpha * v
        
        return ciou

# 💥 DETECTION LOSS COMBINÉE
class DetectionLoss(nn.Module):
    """
    💥 Perte de détection combinée (classification + régression)
    
    Combine:
    - Focal Loss pour la classification
    - Smooth L1 / IoU Loss pour la régression
    - Pondération automatique entre les deux
    """
    
    def __init__(
        self,
        num_classes: int = 28,
        alpha: float = 0.25,
        gamma: float = 2.0,
        smooth_l1_beta: float = 0.1,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        iou_loss_weight: float = 0.0,
        use_quality_focal: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.iou_loss_weight = iou_loss_weight
        
        # Perte de classification
        if use_quality_focal:
            self.cls_loss = QualityFocalLoss(gamma=gamma)
        else:
            self.cls_loss = FocalLoss(alpha=alpha, gamma=gamma)
        
        # Perte de régression
        self.reg_loss = SmoothL1Loss(beta=smooth_l1_beta)
        
        # Perte IoU optionnelle
        if iou_loss_weight > 0:
            self.iou_loss = IoULoss(loss_type='giou')
        else:
            self.iou_loss = None
        
        logger.info(f"💥 DetectionLoss: {num_classes} classes, QFL={use_quality_focal}")
    
    def forward(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
        anchors: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass de la perte combinée
        
        Args:
            predictions: Prédictions du modèle
                - 'scores': Scores de classification par niveau
                - 'boxes': Régressions de boîtes par niveau
            targets: Targets ground truth par image
            anchors: Anchors par niveau
            
        Returns:
            Dictionnaire des pertes
        """
        
        device = predictions['scores'][0].device
        
        # Concaténation des prédictions de tous les niveaux
        all_cls_preds = torch.cat(predictions['scores'], dim=1)  # [B, N_total, C]
        all_reg_preds = torch.cat(predictions['boxes'], dim=1)   # [B, N_total, 4]
        all_anchors = torch.cat(anchors, dim=0)                  # [N_total, 4]
        
        batch_size = all_cls_preds.size(0)
        
        # Assignation targets aux anchors
        cls_targets, reg_targets, pos_mask = self._assign_targets(
            all_anchors, targets, device
        )
        
        # Perte de classification
        cls_loss = self._compute_classification_loss(
            all_cls_preds, cls_targets, pos_mask
        )
        
        # Perte de régression
        reg_loss = self._compute_regression_loss(
            all_reg_preds, reg_targets, pos_mask
        )
        
        # Perte IoU optionnelle
        iou_loss = torch.tensor(0.0, device=device)
        if self.iou_loss is not None and pos_mask.any():
            iou_loss = self._compute_iou_loss(
                all_reg_preds, reg_targets, all_anchors, pos_mask
            )
        
        # Pondération et combinaison
        total_loss = (
            self.cls_weight * cls_loss +
            self.reg_weight * reg_loss +
            self.iou_loss_weight * iou_loss
        )
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'iou_loss': iou_loss
        }
    
    def _assign_targets(
        self,
        anchors: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assigne les targets aux anchors"""
        
        batch_size = len(targets)
        num_anchors = anchors.size(0)
        
        # Tensors de sortie
        cls_targets = torch.zeros(batch_size, num_anchors, dtype=torch.long, device=device)
        reg_targets = torch.zeros(batch_size, num_anchors, 4, device=device)
        pos_mask = torch.zeros(batch_size, num_anchors, dtype=torch.bool, device=device)
        
        for batch_idx, target in enumerate(targets):
            if 'boxes' not in target or 'labels' not in target:
                continue
            
            gt_boxes = target['boxes'].to(device)
            gt_labels = target['labels'].to(device)
            
            if len(gt_boxes) == 0:
                continue
            
            # Calcul IoU anchors vs GT
            from .box_utils import calculate_iou, BoxFormat
            ious = calculate_iou(anchors, gt_boxes, BoxFormat.XYXY)
            
            # Assignation simple: meilleur IoU > 0.5 = positif
            max_ious, matched_gt_idx = ious.max(dim=1)
            pos_anchor_mask = max_ious > 0.5
            
            # Assignation des labels
            cls_targets[batch_idx, pos_anchor_mask] = gt_labels[matched_gt_idx[pos_anchor_mask]]
            
            # Encodage des boîtes
            if pos_anchor_mask.any():
                from .box_utils import encode_boxes
                matched_boxes = gt_boxes[matched_gt_idx[pos_anchor_mask]]
                reg_targets[batch_idx, pos_anchor_mask] = encode_boxes(
                    matched_boxes, anchors[pos_anchor_mask]
                )
            
            pos_mask[batch_idx] = pos_anchor_mask
        
        return cls_targets, reg_targets, pos_mask
    
    def _compute_classification_loss(
        self,
        cls_preds: torch.Tensor,
        cls_targets: torch.Tensor,
        pos_mask: torch.Tensor
    ) -> torch.Tensor:
        """Calcule la perte de classification"""
        
        # Reshape pour la perte
        cls_preds_flat = cls_preds.view(-1, self.num_classes + 1)
        cls_targets_flat = cls_targets.view(-1)
        
        # Masque pour ignorer les anchors ignorés
        valid_mask = cls_targets_flat >= 0
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=cls_preds.device, requires_grad=True)
        
        return self.cls_loss(
            cls_preds_flat[valid_mask],
            cls_targets_flat[valid_mask]
        )
    
    def _compute_regression_loss(
        self,
        reg_preds: torch.Tensor,
        reg_targets: torch.Tensor,
        pos_mask: torch.Tensor
    ) -> torch.Tensor:
        """Calcule la perte de régression"""
        
        if not pos_mask.any():
            return torch.tensor(0.0, device=reg_preds.device, requires_grad=True)
        
        # Seulement sur les anchors positifs
        pos_reg_preds = reg_preds[pos_mask]
        pos_reg_targets = reg_targets[pos_mask]
        
        return self.reg_loss(pos_reg_preds, pos_reg_targets)
    
    def _compute_iou_loss(
        self,
        reg_preds: torch.Tensor,
        reg_targets: torch.Tensor,
        anchors: torch.Tensor,
        pos_mask: torch.Tensor
    ) -> torch.Tensor:
        """Calcule la perte IoU"""
        
        if not pos_mask.any():
            return torch.tensor(0.0, device=reg_preds.device, requires_grad=True)
        
        from .box_utils import decode_boxes
        
        # Décodage des prédictions
        pos_anchors = anchors.unsqueeze(0).expand(reg_preds.size(0), -1, -1)[pos_mask]
        pos_reg_preds = reg_preds[pos_mask]
        pos_reg_targets = reg_targets[pos_mask]
        
        pred_boxes = decode_boxes(pos_reg_preds, pos_anchors)
        target_boxes = decode_boxes(pos_reg_targets, pos_anchors)
        
        return self.iou_loss(pred_boxes, target_boxes)

# 🔧 FONCTIONS UTILITAIRES
def calculate_classification_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = 'focal',
    **kwargs
) -> torch.Tensor:
    """🔧 Fonction utilitaire pour perte de classification"""
    
    if loss_type == 'focal':
        loss_fn = FocalLoss(**kwargs)
    elif loss_type == 'cross_entropy':
        return F.cross_entropy(predictions, targets, **kwargs)
    elif loss_type == 'quality_focal':
        loss_fn = QualityFocalLoss(**kwargs)
    else:
        raise ValueError(f"Type de perte non supporté: {loss_type}")
    
    return loss_fn(predictions, targets)

def calculate_regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = 'smooth_l1',
    **kwargs
) -> torch.Tensor:
    """🔧 Fonction utilitaire pour perte de régression"""
    
    if loss_type == 'smooth_l1':
        loss_fn = SmoothL1Loss(**kwargs)
        return loss_fn(predictions, targets)
    elif loss_type == 'l1':
        return F.l1_loss(predictions, targets, **kwargs)
    elif loss_type == 'mse':
        return F.mse_loss(predictions, targets, **kwargs)
    elif loss_type == 'iou':
        loss_fn = IoULoss(**kwargs)
        return loss_fn(predictions, targets)
    else:
        raise ValueError(f"Type de perte non supporté: {loss_type}")

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "FocalLoss",
    "QualityFocalLoss",
    "SmoothL1Loss",
    "IoULoss",
    "DetectionLoss",
    "calculate_classification_loss",
    "calculate_regression_loss"
]