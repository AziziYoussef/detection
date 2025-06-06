"""
📦 BOX_UTILS - UTILITAIRES POUR BOÎTES ENGLOBANTES
=================================================
Fonctions essentielles pour manipuler les boîtes englobantes en détection d'objets

Fonctionnalités:
- Calcul IoU (Intersection over Union) optimisé
- Non-Maximum Suppression (NMS) standard et Soft-NMS
- Encodage/décodage relatif aux anchors
- Conversion entre formats (xyxy, xywh, center)
- Clipping et filtrage des boîtes
- Métriques et évaluation

Formats de boîtes supportés:
- XYXY: [x1, y1, x2, y2] (coins opposés)
- XYWH: [x, y, w, h] (coin + dimensions)
- CENTER: [cx, cy, w, h] (centre + dimensions)
- COCO: [x, y, w, h] (coin top-left + dimensions)

Performance:
- Implémentations vectorisées PyTorch
- Support GPU/CPU automatique
- Optimisations numériques pour stabilité
- Cache pour opérations répétées
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Union, Optional, Dict, Any
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# 📋 ÉNUMÉRATIONS
class BoxFormat(str, Enum):
    """📋 Formats de boîtes supportés"""
    XYXY = "xyxy"           # [x1, y1, x2, y2]
    XYWH = "xywh"           # [x, y, w, h]  
    CENTER = "center"       # [cx, cy, w, h]
    COCO = "coco"           # [x, y, w, h] (format COCO)

# 📐 CALCUL IOU
def calculate_iou(
    boxes1: torch.Tensor, 
    boxes2: torch.Tensor,
    box_format: BoxFormat = BoxFormat.XYXY,
    epsilon: float = 1e-7
) -> torch.Tensor:
    """
    📐 Calcule l'IoU entre deux ensembles de boîtes
    
    Args:
        boxes1: Tensor [N, 4] première série de boîtes
        boxes2: Tensor [M, 4] deuxième série de boîtes
        box_format: Format des boîtes d'entrée
        epsilon: Petite valeur pour éviter division par 0
        
    Returns:
        Tensor [N, M] matrice des IoU
    """
    
    # Conversion vers format XYXY si nécessaire
    if box_format != BoxFormat.XYXY:
        boxes1 = convert_box_format(boxes1, box_format, BoxFormat.XYXY)
        boxes2 = convert_box_format(boxes2, box_format, BoxFormat.XYXY)
    
    # Dimensions
    N = boxes1.size(0)
    M = boxes2.size(0)
    
    # Expansion pour broadcasting
    # boxes1: [N, 1, 4] boxes2: [1, M, 4]
    boxes1_expanded = boxes1.unsqueeze(1).expand(N, M, 4)
    boxes2_expanded = boxes2.unsqueeze(0).expand(N, M, 4)
    
    # Coordonnées intersection
    inter_x1 = torch.max(boxes1_expanded[:, :, 0], boxes2_expanded[:, :, 0])
    inter_y1 = torch.max(boxes1_expanded[:, :, 1], boxes2_expanded[:, :, 1])
    inter_x2 = torch.min(boxes1_expanded[:, :, 2], boxes2_expanded[:, :, 2])
    inter_y2 = torch.min(boxes1_expanded[:, :, 3], boxes2_expanded[:, :, 3])
    
    # Aire intersection
    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_width * inter_height
    
    # Aires des boîtes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Expansion pour broadcasting
    area1_expanded = area1.unsqueeze(1).expand(N, M)
    area2_expanded = area2.unsqueeze(0).expand(N, M)
    
    # Aire union
    union_area = area1_expanded + area2_expanded - inter_area
    
    # IoU avec protection division par zéro
    iou = inter_area / torch.clamp(union_area, min=epsilon)
    
    return iou

def calculate_giou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor, 
    box_format: BoxFormat = BoxFormat.XYXY,
    epsilon: float = 1e-7
) -> torch.Tensor:
    """
    📐 Calcule le GIoU (Generalized IoU) entre boîtes
    
    GIoU = IoU - |C - A ∪ B| / |C|
    où C est la plus petite boîte englobant A et B
    """
    
    # Conversion vers XYXY
    if box_format != BoxFormat.XYXY:
        boxes1 = convert_box_format(boxes1, box_format, BoxFormat.XYXY)
        boxes2 = convert_box_format(boxes2, box_format, BoxFormat.XYXY)
    
    # IoU standard
    iou = calculate_iou(boxes1, boxes2, BoxFormat.XYXY, epsilon)
    
    N, M = iou.shape
    
    # Expansion pour broadcasting
    boxes1_expanded = boxes1.unsqueeze(1).expand(N, M, 4)
    boxes2_expanded = boxes2.unsqueeze(0).expand(N, M, 4)
    
    # Boîte englobante C
    c_x1 = torch.min(boxes1_expanded[:, :, 0], boxes2_expanded[:, :, 0])
    c_y1 = torch.min(boxes1_expanded[:, :, 1], boxes2_expanded[:, :, 1])
    c_x2 = torch.max(boxes1_expanded[:, :, 2], boxes2_expanded[:, :, 2])
    c_y2 = torch.max(boxes1_expanded[:, :, 3], boxes2_expanded[:, :, 3])
    
    # Aire de C
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
    
    # Aires originales
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    area1_expanded = area1.unsqueeze(1).expand(N, M)
    area2_expanded = area2.unsqueeze(0).expand(N, M)
    
    # Union
    union_area = area1_expanded + area2_expanded - iou * area1_expanded * area2_expanded / (iou + epsilon)
    
    # GIoU
    giou = iou - (c_area - union_area) / torch.clamp(c_area, min=epsilon)
    
    return giou

# 🎯 NON-MAXIMUM SUPPRESSION
def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    score_threshold: float = 0.05,
    nms_threshold: float = 0.5,
    max_detections: int = 100,
    box_format: BoxFormat = BoxFormat.XYXY
) -> torch.Tensor:
    """
    🎯 Applique Non-Maximum Suppression
    
    Args:
        boxes: Boîtes [N, 4]
        scores: Scores de confiance [N]
        score_threshold: Seuil minimum de score
        nms_threshold: Seuil IoU pour suppression
        max_detections: Nombre max de détections
        box_format: Format des boîtes
        
    Returns:
        Indices des boîtes conservées
    """
    
    # Filtrage par score
    valid_mask = scores > score_threshold
    if not valid_mask.any():
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    
    boxes = boxes[valid_mask]
    scores = scores[valid_mask]
    valid_indices = torch.where(valid_mask)[0]
    
    # Conversion vers XYXY si nécessaire
    if box_format != BoxFormat.XYXY:
        boxes_xyxy = convert_box_format(boxes, box_format, BoxFormat.XYXY)
    else:
        boxes_xyxy = boxes
    
    # NMS PyTorch optimisé
    try:
        from torchvision.ops import nms
        keep_indices = nms(boxes_xyxy, scores, nms_threshold)
    except ImportError:
        # Fallback vers implémentation manuelle
        keep_indices = _manual_nms(boxes_xyxy, scores, nms_threshold)
    
    # Limitation du nombre de détections
    if len(keep_indices) > max_detections:
        keep_indices = keep_indices[:max_detections]
    
    # Retourner indices originaux
    return valid_indices[keep_indices]

def apply_soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    score_threshold: float = 0.05,
    nms_threshold: float = 0.5,
    sigma: float = 0.5,
    max_detections: int = 100,
    box_format: BoxFormat = BoxFormat.XYXY
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    🎯 Applique Soft-NMS (diminue scores au lieu de supprimer)
    
    Returns:
        Tuple (indices_conservés, scores_ajustés)
    """
    
    # Filtrage initial
    valid_mask = scores > score_threshold
    if not valid_mask.any():
        empty_indices = torch.empty(0, dtype=torch.long, device=boxes.device)
        empty_scores = torch.empty(0, device=boxes.device)
        return empty_indices, empty_scores
    
    boxes = boxes[valid_mask].clone()
    scores = scores[valid_mask].clone()
    valid_indices = torch.where(valid_mask)[0]
    
    # Conversion vers XYXY
    if box_format != BoxFormat.XYXY:
        boxes = convert_box_format(boxes, box_format, BoxFormat.XYXY)
    
    # Soft-NMS
    keep_indices = []
    
    while len(scores) > 0 and len(keep_indices) < max_detections:
        # Prendre le meilleur score
        max_idx = torch.argmax(scores)
        keep_indices.append(max_idx.item())
        
        if len(scores) == 1:
            break
        
        # Calculer IoU avec les autres boîtes
        current_box = boxes[max_idx:max_idx+1]
        other_boxes = torch.cat([boxes[:max_idx], boxes[max_idx+1:]])
        
        if len(other_boxes) == 0:
            break
        
        ious = calculate_iou(current_box, other_boxes, BoxFormat.XYXY).squeeze(0)
        
        # Diminution soft des scores
        decay_factors = torch.exp(-(ious ** 2) / sigma)
        
        # Application des facteurs
        other_scores = torch.cat([scores[:max_idx], scores[max_idx+1:]])
        other_scores *= decay_factors
        
        # Supprimer la boîte courante et mettre à jour
        boxes = other_boxes
        scores = other_scores
        
        # Réindexation
        keep_indices[-1] = valid_indices[max_idx].item()
        valid_indices = torch.cat([valid_indices[:max_idx], valid_indices[max_idx+1:]])
        
        # Filtrage par seuil mis à jour
        score_mask = scores > score_threshold
        boxes = boxes[score_mask]
        scores = scores[score_mask]
        valid_indices = valid_indices[score_mask]
    
    final_indices = torch.tensor(keep_indices, device=boxes.device, dtype=torch.long)
    final_scores = scores[:len(keep_indices)] if len(scores) >= len(keep_indices) else scores
    
    return final_indices, final_scores

def _manual_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor, 
    threshold: float
) -> torch.Tensor:
    """🔧 Implémentation manuelle NMS (fallback)"""
    
    # Tri par scores décroissants
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep = []
    while len(sorted_indices) > 0:
        # Prendre le meilleur
        current = sorted_indices[0]
        keep.append(current.item())
        
        if len(sorted_indices) == 1:
            break
        
        # Calculer IoU avec les autres
        current_box = boxes[current:current+1]
        other_boxes = boxes[sorted_indices[1:]]
        
        ious = calculate_iou(current_box, other_boxes, BoxFormat.XYXY).squeeze(0)
        
        # Garder ceux avec IoU < threshold
        suppress_mask = ious < threshold
        sorted_indices = sorted_indices[1:][suppress_mask]
    
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)

# 🔄 CONVERSION DE FORMATS
def convert_box_format(
    boxes: torch.Tensor,
    input_format: BoxFormat,
    output_format: BoxFormat
) -> torch.Tensor:
    """
    🔄 Convertit entre différents formats de boîtes
    
    Args:
        boxes: Boîtes d'entrée [N, 4]
        input_format: Format d'entrée
        output_format: Format de sortie
        
    Returns:
        Boîtes converties [N, 4]
    """
    
    if input_format == output_format:
        return boxes.clone()
    
    # Conversion via format intermédiaire XYXY
    if input_format != BoxFormat.XYXY:
        boxes = _to_xyxy(boxes, input_format)
    
    if output_format != BoxFormat.XYXY:
        boxes = _from_xyxy(boxes, output_format)
    
    return boxes

def _to_xyxy(boxes: torch.Tensor, input_format: BoxFormat) -> torch.Tensor:
    """🔄 Convertit vers format XYXY"""
    
    if input_format == BoxFormat.XYWH or input_format == BoxFormat.COCO:
        # [x, y, w, h] → [x1, y1, x2, y2]
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    elif input_format == BoxFormat.CENTER:
        # [cx, cy, w, h] → [x1, y1, x2, y2]
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1, y1 = cx - w/2, cy - h/2
        x2, y2 = cx + w/2, cy + h/2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    else:
        raise ValueError(f"Format d'entrée non supporté: {input_format}")

def _from_xyxy(boxes: torch.Tensor, output_format: BoxFormat) -> torch.Tensor:
    """🔄 Convertit depuis format XYXY"""
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    if output_format == BoxFormat.XYWH or output_format == BoxFormat.COCO:
        # [x1, y1, x2, y2] → [x, y, w, h]
        x, y = x1, y1
        w, h = x2 - x1, y2 - y1
        return torch.stack([x, y, w, h], dim=1)
    
    elif output_format == BoxFormat.CENTER:
        # [x1, y1, x2, y2] → [cx, cy, w, h]
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        return torch.stack([cx, cy, w, h], dim=1)
    
    else:
        raise ValueError(f"Format de sortie non supporté: {output_format}")

# 📦 ENCODAGE/DÉCODAGE POUR ANCHORS
def encode_boxes(
    gt_boxes: torch.Tensor,
    anchor_boxes: torch.Tensor,
    weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
) -> torch.Tensor:
    """
    📦 Encode les GT boxes relativement aux anchors
    
    Args:
        gt_boxes: Ground truth boxes [N, 4] format CENTER
        anchor_boxes: Anchor boxes [N, 4] format CENTER
        weights: Poids d'encodage (dx, dy, dw, dh)
        
    Returns:
        Deltas encodés [N, 4]
    """
    
    # Assurer format CENTER
    if gt_boxes.size(-1) != 4 or anchor_boxes.size(-1) != 4:
        raise ValueError("Boxes doivent être au format [cx, cy, w, h]")
    
    # Extraction composants
    gt_cx, gt_cy, gt_w, gt_h = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
    anc_cx, anc_cy, anc_w, anc_h = anchor_boxes[:, 0], anchor_boxes[:, 1], anchor_boxes[:, 2], anchor_boxes[:, 3]
    
    # Encodage
    # Δx = (gt_cx - anc_cx) / anc_w
    # Δy = (gt_cy - anc_cy) / anc_h  
    # Δw = log(gt_w / anc_w)
    # Δh = log(gt_h / anc_h)
    
    dx = (gt_cx - anc_cx) / torch.clamp(anc_w, min=1e-6)
    dy = (gt_cy - anc_cy) / torch.clamp(anc_h, min=1e-6)
    dw = torch.log(torch.clamp(gt_w / torch.clamp(anc_w, min=1e-6), min=1e-6))
    dh = torch.log(torch.clamp(gt_h / torch.clamp(anc_h, min=1e-6), min=1e-6))
    
    # Application des poids
    dx *= weights[0]
    dy *= weights[1]
    dw *= weights[2]
    dh *= weights[3]
    
    return torch.stack([dx, dy, dw, dh], dim=1)

def decode_boxes(
    deltas: torch.Tensor,
    anchor_boxes: torch.Tensor,
    weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
) -> torch.Tensor:
    """
    📦 Décode les deltas en boîtes absolues
    
    Args:
        deltas: Deltas prédits [N, 4]
        anchor_boxes: Anchor boxes [N, 4] format CENTER
        weights: Poids de décodage
        
    Returns:
        Boîtes décodées [N, 4] format CENTER
    """
    
    # Application des poids inverses
    dx = deltas[:, 0] / weights[0]
    dy = deltas[:, 1] / weights[1]
    dw = deltas[:, 2] / weights[2]
    dh = deltas[:, 3] / weights[3]
    
    # Composants anchors
    anc_cx, anc_cy, anc_w, anc_h = anchor_boxes[:, 0], anchor_boxes[:, 1], anchor_boxes[:, 2], anchor_boxes[:, 3]
    
    # Décodage
    pred_cx = dx * anc_w + anc_cx
    pred_cy = dy * anc_h + anc_cy
    pred_w = torch.exp(torch.clamp(dw, max=4.135)) * anc_w  # exp(4.135) ≈ 62
    pred_h = torch.exp(torch.clamp(dh, max=4.135)) * anc_h
    
    return torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1)

# 🔧 UTILITAIRES ADDITIONNELS
def box_area(boxes: torch.Tensor, box_format: BoxFormat = BoxFormat.XYXY) -> torch.Tensor:
    """📏 Calcule l'aire des boîtes"""
    
    if box_format == BoxFormat.XYXY:
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    elif box_format in [BoxFormat.XYWH, BoxFormat.COCO, BoxFormat.CENTER]:
        return boxes[:, 2] * boxes[:, 3]
    else:
        raise ValueError(f"Format non supporté: {box_format}")

def clip_boxes_to_image(
    boxes: torch.Tensor,
    image_size: Tuple[int, int],
    box_format: BoxFormat = BoxFormat.XYXY
) -> torch.Tensor:
    """✂️ Clippe les boîtes aux bords de l'image"""
    
    height, width = image_size
    
    if box_format == BoxFormat.XYXY:
        boxes = boxes.clone()
        boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=width)   # x1
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=height)  # y1
        boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=width)   # x2
        boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=height)  # y2
        return boxes
    
    else:
        # Convertir vers XYXY, clipper, reconvertir
        boxes_xyxy = convert_box_format(boxes, box_format, BoxFormat.XYXY)
        clipped_xyxy = clip_boxes_to_image(boxes_xyxy, image_size, BoxFormat.XYXY)
        return convert_box_format(clipped_xyxy, BoxFormat.XYXY, box_format)

def filter_boxes_by_score(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """🔍 Filtre les boîtes par score de confiance"""
    
    valid_mask = scores > threshold
    return boxes[valid_mask], scores[valid_mask]

def filter_boxes_by_area(
    boxes: torch.Tensor,
    min_area: float = 0.0,
    max_area: float = float('inf'),
    box_format: BoxFormat = BoxFormat.XYXY
) -> torch.Tensor:
    """🔍 Filtre les boîtes par aire"""
    
    areas = box_area(boxes, box_format)
    valid_mask = (areas >= min_area) & (areas <= max_area)
    return boxes[valid_mask]

def remove_small_boxes(
    boxes: torch.Tensor,
    min_size: float = 1.0,
    box_format: BoxFormat = BoxFormat.XYXY
) -> torch.Tensor:
    """🔍 Supprime les petites boîtes"""
    
    if box_format == BoxFormat.XYXY:
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
    elif box_format in [BoxFormat.XYWH, BoxFormat.COCO, BoxFormat.CENTER]:
        widths = boxes[:, 2]
        heights = boxes[:, 3]
    else:
        raise ValueError(f"Format non supporté: {box_format}")
    
    valid_mask = (widths >= min_size) & (heights >= min_size)
    return boxes[valid_mask]

# 📊 MÉTRIQUES
def calculate_average_precision(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = 0.5
) -> float:
    """📊 Calcule l'Average Precision"""
    
    # Tri par scores décroissants
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes_sorted = pred_boxes[sorted_indices]
    pred_scores_sorted = pred_scores[sorted_indices]
    
    # Calcul IoU avec GT
    ious = calculate_iou(pred_boxes_sorted, gt_boxes)
    
    # Matching greedy
    matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
    tp = torch.zeros(len(pred_boxes_sorted))
    fp = torch.zeros(len(pred_boxes_sorted))
    
    for i in range(len(pred_boxes_sorted)):
        max_iou, max_idx = torch.max(ious[i], dim=0)
        
        if max_iou >= iou_threshold and not matched_gt[max_idx]:
            tp[i] = 1
            matched_gt[max_idx] = True
        else:
            fp[i] = 1
    
    # Calcul précision/rappel
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    
    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # Average Precision (aire sous courbe P-R)
    ap = torch.trapz(precisions, recalls).item()
    
    return ap

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "BoxFormat",
    "calculate_iou",
    "calculate_giou", 
    "apply_nms",
    "apply_soft_nms",
    "convert_box_format",
    "encode_boxes",
    "decode_boxes",
    "box_area",
    "clip_boxes_to_image",
    "filter_boxes_by_score",
    "filter_boxes_by_area", 
    "remove_small_boxes",
    "calculate_average_precision"
]