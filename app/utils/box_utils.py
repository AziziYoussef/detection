# app/utils/box_utils.py
import numpy as np
import torch
from typing import List, Tuple, Union
from app.schemas.detection import BoundingBox

def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calcule l'IoU entre deux boîtes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU score
    """
    # Coordonnées de l'intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Aire de l'intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Aires des boîtes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Union
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def boxes_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calcule l'IoU entre deux ensembles de boîtes
    
    Args:
        boxes1: [N, 4] array de boîtes
        boxes2: [M, 4] array de boîtes
        
    Returns:
        [N, M] array d'IoU scores
    """
    N, M = len(boxes1), len(boxes2)
    ious = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            ious[i, j] = box_iou(boxes1[i], boxes2[j])
    
    return ious

def non_max_suppression(detections: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """
    Applique Non-Maximum Suppression
    
    Args:
        detections: [N, 6] array (x1, y1, x2, y2, conf, class)
        iou_threshold: Seuil IoU pour suppression
        
    Returns:
        Indices des détections conservées
    """
    if len(detections) == 0:
        return np.array([], dtype=int)
    
    # Tri par confiance décroissante
    indices = np.argsort(detections[:, 4])[::-1]
    keep = []
    
    while len(indices) > 0:
        # Garde la boîte avec la plus haute confiance
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calcule IoU avec les autres boîtes
        current_box = detections[current, :4]
        other_boxes = detections[indices[1:], :4]
        
        ious = np.array([box_iou(current_box, box) for box in other_boxes])
        
        # Garde seulement les boîtes avec IoU < seuil
        indices = indices[1:][ious < iou_threshold]
    
    return np.array(keep)

def nms_by_class(detections: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """
    Applique NMS par classe séparément
    
    Args:
        detections: [N, 6] array (x1, y1, x2, y2, conf, class)
        iou_threshold: Seuil IoU
        
    Returns:
        Détections filtrées
    """
    if len(detections) == 0:
        return detections
    
    keep_indices = []
    unique_classes = np.unique(detections[:, 5])
    
    for cls in unique_classes:
        # Filtre par classe
        class_mask = detections[:, 5] == cls
        class_detections = detections[class_mask]
        class_indices = np.where(class_mask)[0]
        
        # Applique NMS sur cette classe
        keep_class = non_max_suppression(class_detections, iou_threshold)
        
        # Ajoute les indices originaux
        keep_indices.extend(class_indices[keep_class])
    
    return detections[keep_indices]

def filter_by_confidence(detections: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Filtre les détections par confiance"""
    if len(detections) == 0:
        return detections
    
    return detections[detections[:, 4] >= threshold]

def filter_by_size(detections: np.ndarray, min_size: int = 10, 
                  max_size: int = None) -> np.ndarray:
    """Filtre les détections par taille"""
    if len(detections) == 0:
        return detections
    
    # Calcule largeur et hauteur
    widths = detections[:, 2] - detections[:, 0]
    heights = detections[:, 3] - detections[:, 1]
    
    # Filtre par taille minimale
    size_mask = (widths >= min_size) & (heights >= min_size)
    
    # Filtre par taille maximale si spécifiée
    if max_size is not None:
        size_mask &= (widths <= max_size) & (heights <= max_size)
    
    return detections[size_mask]

def clip_boxes(boxes: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """Clipe les boîtes aux limites de l'image"""
    h, w = img_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)  # x1, x2
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)  # y1, y2
    return boxes

def scale_boxes(boxes: np.ndarray, scale_factor: Union[float, Tuple[float, float]]) -> np.ndarray:
    """Redimensionne les boîtes selon un facteur d'échelle"""
    if isinstance(scale_factor, (int, float)):
        scale_x = scale_y = scale_factor
    else:
        scale_x, scale_y = scale_factor
    
    boxes[:, [0, 2]] *= scale_x  # x1, x2
    boxes[:, [1, 3]] *= scale_y  # y1, y2
    return boxes

def center_to_corner(boxes: np.ndarray) -> np.ndarray:
    """Convertit du format (cx, cy, w, h) vers (x1, y1, x2, y2)"""
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    return np.stack([x1, y1, x2, y2], axis=1)

def corner_to_center(boxes: np.ndarray) -> np.ndarray:
    """Convertit du format (x1, y1, x2, y2) vers (cx, cy, w, h)"""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    
    return np.stack([cx, cy, w, h], axis=1)

def box_area(boxes: np.ndarray) -> np.ndarray:
    """Calcule l'aire des boîtes"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_distance(box1: np.ndarray, box2: np.ndarray, 
                distance_type: str = 'center') -> float:
    """
    Calcule la distance entre deux boîtes
    
    Args:
        box1, box2: [x1, y1, x2, y2]
        distance_type: 'center', 'nearest', 'furthest'
    """
    if distance_type == 'center':
        c1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
        c2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
    
    elif distance_type == 'nearest':
        # Distance minimale entre les bords
        dx = max(0, max(box1[0] - box2[2], box2[0] - box1[2]))
        dy = max(0, max(box1[1] - box2[3], box2[1] - box1[3]))
        return np.sqrt(dx**2 + dy**2)
    
    elif distance_type == 'furthest':
        # Distance maximale entre les coins
        distances = []
        for i in [0, 2]:
            for j in [1, 3]:
                for k in [0, 2]:
                    for l in [1, 3]:
                        distances.append(np.sqrt((box1[i] - box2[k])**2 + 
                                               (box1[j] - box2[l])**2))
        return max(distances)
    
    else:
        raise ValueError(f"Type de distance inconnu: {distance_type}")

def merge_overlapping_boxes(detections: np.ndarray, 
                          iou_threshold: float = 0.7) -> np.ndarray:
    """Fusionne les boîtes qui se chevauchent beaucoup"""
    if len(detections) == 0:
        return detections
    
    merged = []
    used = set()
    
    for i, det in enumerate(detections):
        if i in used:
            continue
        
        # Trouve toutes les boîtes qui se chevauchent
        overlapping = [i]
        for j in range(i + 1, len(detections)):
            if j in used:
                continue
            
            if box_iou(det[:4], detections[j, :4]) > iou_threshold:
                overlapping.append(j)
                used.add(j)
        
        # Fusionne les boîtes
        if len(overlapping) == 1:
            merged.append(det)
        else:
            overlap_dets = detections[overlapping]
            
            # Moyenne pondérée par confiance
            weights = overlap_dets[:, 4]
            weights = weights / weights.sum()
            
            merged_box = np.average(overlap_dets[:, :4], axis=0, weights=weights)
            merged_conf = overlap_dets[:, 4].max()  # Confiance maximale
            merged_class = overlap_dets[np.argmax(overlap_dets[:, 4]), 5]  # Classe de la meilleure
            
            merged.append([*merged_box, merged_conf, merged_class])
        
        used.add(i)
    
    return np.array(merged) if merged else np.array([]).reshape(0, 6)

def convert_to_schema(detections: np.ndarray, class_names: List[str]) -> List[BoundingBox]:
    """Convertit les détections en schémas Pydantic"""
    boxes = []
    
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        
        box = BoundingBox(
            x=float(x1),
            y=float(y1),
            width=float(x2 - x1),
            height=float(y2 - y1)
        )
        boxes.append(box)
    
    return boxes

def validate_boxes(boxes: np.ndarray) -> np.ndarray:
    """Valide et corrige les boîtes invalides"""
    if len(boxes) == 0:
        return boxes
    
    # Corrige x1 > x2 ou y1 > y2
    boxes[:, [0, 2]] = np.sort(boxes[:, [0, 2]], axis=1)
    boxes[:, [1, 3]] = np.sort(boxes[:, [1, 3]], axis=1)
    
    # Supprime les boîtes de taille nulle
    valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    
    return boxes[valid_mask]