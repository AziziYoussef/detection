"""
⚓ ANCHORS - SYSTÈME D'ANCHORS POUR DÉTECTION D'OBJETS
====================================================
Implémentation du système d'anchors pour les modèles de détection basés sur anchors

Le système d'anchors génère des boîtes de référence à différentes échelles et ratios
pour permettre la détection d'objets de tailles variées à travers l'image.

Types d'anchors:
- Standard anchors: Grille régulière avec différents ratios/échelles
- RetinaNet anchors: Multi-échelle avec FPN
- SSD anchors: Anchors par niveau de feature map
- YOLO anchors: Anchors pré-calculés optimisés

Architecture:
                 Feature Maps (P2-P7)
                         │
               ┌─────────┼─────────┐
               │         │         │
               ▼         ▼         ▼
           Anchors    Anchors   Anchors
            P2         P3        P4...
        (small obj) (medium)   (large)
             │         │         │
             └─────────┼─────────┘
                       │
                 All Anchors
               [N_total, 4]

Chaque anchor = [x_center, y_center, width, height]
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union, Optional
from dataclasses import dataclass, field
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)

# 📋 CONFIGURATION ANCHORS
@dataclass
class AnchorConfig:
    """⚙️ Configuration du système d'anchors"""
    # Tailles de base par niveau FPN
    sizes: List[float] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    
    # Ratios d'aspect (width/height)
    ratios: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    
    # Échelles multiples par taille
    scales: List[float] = field(default_factory=lambda: [1.0, 1.26, 1.587])
    
    # Nombre de niveaux FPN
    num_levels: int = 5
    
    # Stride par niveau (downsampling factor)
    strides: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64])
    
    # Décalage pour centrage (0.5 = centre de pixel)
    offset: float = 0.5
    
    # Clipping des anchors aux bords de l'image
    clip_anchors: bool = True
    
    def __post_init__(self):
        """Validation et ajustements post-initialisation"""
        # Vérifier cohérence tailles/strides/niveaux
        if len(self.sizes) != self.num_levels:
            logger.warning(f"⚠️ Ajustement sizes: {len(self.sizes)} → {self.num_levels}")
            if len(self.sizes) < self.num_levels:
                # Extrapoler les tailles manquantes
                factor = self.sizes[-1] / self.sizes[-2] if len(self.sizes) > 1 else 2.0
                while len(self.sizes) < self.num_levels:
                    self.sizes.append(self.sizes[-1] * factor)
            else:
                self.sizes = self.sizes[:self.num_levels]
        
        if len(self.strides) != self.num_levels:
            logger.warning(f"⚠️ Ajustement strides: {len(self.strides)} → {self.num_levels}")
            if len(self.strides) < self.num_levels:
                # Doubler le stride pour chaque niveau manquant
                factor = 2
                while len(self.strides) < self.num_levels:
                    self.strides.append(self.strides[-1] * factor)
            else:
                self.strides = self.strides[:self.num_levels]
    
    def get_anchors_per_location(self) -> int:
        """📊 Nombre d'anchors par position de grille"""
        return len(self.ratios) * len(self.scales)
    
    def get_total_anchors(self, feature_sizes: List[Tuple[int, int]]) -> int:
        """📊 Nombre total d'anchors pour des tailles de features données"""
        total = 0
        anchors_per_loc = self.get_anchors_per_location()
        
        for h, w in feature_sizes:
            total += h * w * anchors_per_loc
        
        return total

# ⚓ GÉNÉRATEUR D'ANCHORS
class AnchorGenerator(nn.Module):
    """⚓ Générateur d'anchors pour détection multi-échelle"""
    
    def __init__(self, config: AnchorConfig = None):
        super().__init__()
        
        self.config = config or AnchorConfig()
        
        # Pré-calcul des anchors de base
        self.base_anchors = self._generate_base_anchors()
        
        # Cache pour éviter recalculs
        self._cached_anchors = {}
        self._cache_keys = {}
        
        logger.info(f"⚓ AnchorGenerator: {len(self.config.sizes)} niveaux, "
                   f"{self.config.get_anchors_per_location()} anchors/position")
    
    def _generate_base_anchors(self) -> List[torch.Tensor]:
        """🔧 Génère les anchors de base pour chaque niveau"""
        
        base_anchors = []
        
        for i, (size, stride) in enumerate(zip(self.config.sizes, self.config.strides)):
            level_anchors = []
            
            # Génération pour chaque combinaison ratio/scale
            for ratio in self.config.ratios:
                for scale in self.config.scales:
                    # Calcul dimensions
                    anchor_size = size * scale
                    h = anchor_size / math.sqrt(ratio)
                    w = anchor_size * math.sqrt(ratio)
                    
                    # Anchor centré sur origine [x_c, y_c, w, h]
                    anchor = torch.tensor([0.0, 0.0, w, h], dtype=torch.float32)
                    level_anchors.append(anchor)
            
            # Stack anchors pour ce niveau
            level_anchors = torch.stack(level_anchors)
            base_anchors.append(level_anchors)
            
            logger.debug(f"⚓ Niveau {i}: {len(level_anchors)} anchors de base, "
                        f"size={size}, stride={stride}")
        
        return base_anchors
    
    def forward(
        self, 
        images: torch.Tensor,
        features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        🔄 Génère tous les anchors pour un batch d'images
        
        Args:
            images: Batch d'images [N, C, H, W]
            features: Features FPN [niveau0, niveau1, ...]
            
        Returns:
            Liste d'anchors par niveau [A_level, 4]
        """
        
        image_size = images.shape[-2:]  # (H, W)
        feature_sizes = [feat.shape[-2:] for feat in features]
        device = images.device
        
        # Vérifier cache
        cache_key = (tuple(image_size), tuple(tuple(fs) for fs in feature_sizes), str(device))
        
        if cache_key in self._cached_anchors:
            return self._cached_anchors[cache_key]
        
        # Générer anchors pour chaque niveau
        all_anchors = []
        
        for level, (base_anchors, feature_size, stride) in enumerate(
            zip(self.base_anchors, feature_sizes, self.config.strides)
        ):
            level_anchors = self._generate_level_anchors(
                base_anchors, feature_size, stride, device, image_size
            )
            all_anchors.append(level_anchors)
        
        # Mise en cache
        self._cached_anchors[cache_key] = all_anchors
        
        # Nettoyage cache si trop grand
        if len(self._cached_anchors) > 10:
            oldest_key = next(iter(self._cached_anchors))
            del self._cached_anchors[oldest_key]
        
        return all_anchors
    
    def _generate_level_anchors(
        self,
        base_anchors: torch.Tensor,
        feature_size: Tuple[int, int],
        stride: int,
        device: torch.device,
        image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """🔧 Génère anchors pour un niveau spécifique"""
        
        feat_h, feat_w = feature_size
        img_h, img_w = image_size
        
        # Déplacer base_anchors vers device
        base_anchors = base_anchors.to(device)
        
        # Grille de positions
        shifts_x = torch.arange(
            0, feat_w, dtype=torch.float32, device=device
        ) * stride + self.config.offset * stride
        
        shifts_y = torch.arange(
            0, feat_h, dtype=torch.float32, device=device  
        ) * stride + self.config.offset * stride
        
        # Meshgrid
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        
        # Aplatir les grilles
        shifts = torch.stack([
            shift_x.flatten(),
            shift_y.flatten(),
            torch.zeros_like(shift_x.flatten()),  # width offset = 0
            torch.zeros_like(shift_y.flatten())   # height offset = 0
        ], dim=1)
        
        # Répliquer base_anchors pour chaque position
        # base_anchors: [A, 4], shifts: [H*W, 4]
        num_anchors = base_anchors.shape[0]
        num_locations = shifts.shape[0]
        
        anchors = base_anchors.view(1, num_anchors, 4) + shifts.view(num_locations, 1, 4)
        anchors = anchors.view(-1, 4)  # [H*W*A, 4]
        
        # Clipping aux bords de l'image si demandé
        if self.config.clip_anchors:
            anchors = self._clip_anchors_to_image(anchors, image_size)
        
        return anchors
    
    def _clip_anchors_to_image(
        self, 
        anchors: torch.Tensor, 
        image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """✂️ Clippe les anchors aux bords de l'image"""
        
        img_h, img_w = image_size
        
        # Conversion center format → corner format
        x_c, y_c, w, h = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        
        # Clipping
        x1 = torch.clamp(x1, min=0, max=img_w)
        y1 = torch.clamp(y1, min=0, max=img_h)
        x2 = torch.clamp(x2, min=0, max=img_w)
        y2 = torch.clamp(y2, min=0, max=img_h)
        
        # Reconversion center format
        clipped_w = x2 - x1
        clipped_h = y2 - y1
        clipped_x_c = x1 + clipped_w / 2
        clipped_y_c = y1 + clipped_h / 2
        
        return torch.stack([clipped_x_c, clipped_y_c, clipped_w, clipped_h], dim=1)
    
    def get_anchors_info(self) -> Dict[str, any]:
        """📋 Informations sur les anchors générés"""
        
        total_base_anchors = sum(len(base) for base in self.base_anchors)
        
        info = {
            "num_levels": len(self.base_anchors),
            "anchors_per_location": self.config.get_anchors_per_location(),
            "total_base_anchors": total_base_anchors,
            "sizes": self.config.sizes,
            "ratios": self.config.ratios,
            "scales": self.config.scales,
            "strides": self.config.strides,
            "cached_versions": len(self._cached_anchors)
        }
        
        return info

# 🎯 UTILITAIRES ANCHORS
class AnchorUtils:
    """🎯 Utilitaires pour manipulation d'anchors"""
    
    @staticmethod
    def box_iou(anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """📐 Calcule IoU entre anchors et ground truth boxes"""
        
        # Conversion format center → corner
        anchors_corners = AnchorUtils.center_to_corner(anchors)
        gt_corners = AnchorUtils.center_to_corner(gt_boxes)
        
        # Intersection
        inter_mins = torch.max(anchors_corners[:, None, :2], gt_corners[None, :, :2])
        inter_maxs = torch.min(anchors_corners[:, None, 2:], gt_corners[None, :, 2:])
        inter_wh = torch.clamp(inter_maxs - inter_mins, min=0)
        inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]
        
        # Aires des boîtes
        anchors_area = ((anchors_corners[:, 2] - anchors_corners[:, 0]) * 
                       (anchors_corners[:, 3] - anchors_corners[:, 1]))
        gt_area = ((gt_corners[:, 2] - gt_corners[:, 0]) * 
                  (gt_corners[:, 3] - gt_corners[:, 1]))
        
        # Union
        union_area = anchors_area[:, None] + gt_area[None, :] - inter_area
        
        # IoU
        iou = inter_area / torch.clamp(union_area, min=1e-6)
        
        return iou
    
    @staticmethod
    def center_to_corner(boxes: torch.Tensor) -> torch.Tensor:
        """🔄 Convertit format center vers corner"""
        x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    @staticmethod
    def corner_to_center(boxes: torch.Tensor) -> torch.Tensor:
        """🔄 Convertit format corner vers center"""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x_c = (x1 + x2) / 2
        y_c = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([x_c, y_c, w, h], dim=1)
    
    @staticmethod
    def encode_boxes(anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """📦 Encode les GT boxes relativement aux anchors"""
        
        # Format center
        anchor_centers = anchors[:, :2]
        anchor_sizes = anchors[:, 2:]
        
        gt_centers = gt_boxes[:, :2]
        gt_sizes = gt_boxes[:, 2:]
        
        # Encodage
        # Δxy = (gt_center - anchor_center) / anchor_size
        delta_xy = (gt_centers - anchor_centers) / anchor_sizes
        
        # Δwh = log(gt_size / anchor_size)
        delta_wh = torch.log(gt_sizes / anchor_sizes)
        
        return torch.cat([delta_xy, delta_wh], dim=1)
    
    @staticmethod
    def decode_boxes(
        anchors: torch.Tensor, 
        deltas: torch.Tensor,
        weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    ) -> torch.Tensor:
        """📦 Décode les prédictions en boîtes absolues"""
        
        # Application des poids
        delta_xy = deltas[:, :2] / torch.tensor(weights[:2], device=deltas.device)
        delta_wh = deltas[:, 2:] / torch.tensor(weights[2:], device=deltas.device)
        
        # Format center
        anchor_centers = anchors[:, :2]
        anchor_sizes = anchors[:, 2:]
        
        # Décodage
        # pred_center = anchor_center + Δxy * anchor_size
        pred_centers = anchor_centers + delta_xy * anchor_sizes
        
        # pred_size = anchor_size * exp(Δwh)
        pred_sizes = anchor_sizes * torch.exp(delta_wh)
        
        return torch.cat([pred_centers, pred_sizes], dim=1)
    
    @staticmethod
    def filter_anchors_by_size(
        anchors: torch.Tensor,
        min_size: float = 0.0,
        max_size: float = float('inf')
    ) -> torch.Tensor:
        """🔍 Filtre les anchors par taille"""
        
        areas = anchors[:, 2] * anchors[:, 3]
        valid_mask = (areas >= min_size) & (areas <= max_size)
        
        return anchors[valid_mask]

# 🏭 FONCTIONS FACTORY
def generate_anchors(
    sizes: List[float],
    ratios: List[float],
    scales: List[float],
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    🏭 Génère des anchors de base pour tailles/ratios/échelles donnés
    
    Args:
        sizes: Tailles de base
        ratios: Ratios d'aspect
        scales: Échelles multiples
        dtype: Type de données
        
    Returns:
        Anchors [N, 4] au format center
    """
    
    anchors = []
    
    for size in sizes:
        for ratio in ratios:
            for scale in scales:
                anchor_size = size * scale
                h = anchor_size / math.sqrt(ratio)
                w = anchor_size * math.sqrt(ratio)
                
                anchor = torch.tensor([0.0, 0.0, w, h], dtype=dtype)
                anchors.append(anchor)
    
    return torch.stack(anchors)

def create_retinanet_anchors() -> AnchorConfig:
    """🏭 Configuration anchors pour RetinaNet"""
    return AnchorConfig(
        sizes=[32, 64, 128, 256, 512],
        ratios=[0.5, 1.0, 2.0],
        scales=[1.0, 1.26, 1.587],
        strides=[8, 16, 32, 64, 128],
        offset=0.5
    )

def create_ssd_anchors() -> AnchorConfig:
    """🏭 Configuration anchors pour SSD"""
    return AnchorConfig(
        sizes=[30, 60, 111, 162, 213, 264],
        ratios=[0.5, 1.0, 2.0, 3.0],
        scales=[1.0],
        strides=[8, 16, 32, 64, 100, 300],
        offset=0.5
    )

def create_yolo_anchors() -> AnchorConfig:
    """🏭 Configuration anchors pour YOLO"""
    return AnchorConfig(
        # Anchors optimisés YOLO (exemples)
        sizes=[10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
        ratios=[1.0],  # YOLO utilise des anchors pré-calculés
        scales=[1.0],
        strides=[8, 16, 32],
        offset=0.5
    )

def create_mobile_anchors() -> AnchorConfig:
    """🏭 Configuration anchors optimisée mobile"""
    return AnchorConfig(
        sizes=[16, 32, 64, 128],  # Moins de niveaux
        ratios=[0.5, 1.0, 2.0],   # Ratios standards
        scales=[1.0, 1.4],        # Moins d'échelles
        strides=[8, 16, 32, 64],
        offset=0.5
    )

# 📊 ANALYSEUR D'ANCHORS
class AnchorAnalyzer:
    """📊 Analyse et visualisation des anchors"""
    
    def __init__(self, anchor_generator: AnchorGenerator):
        self.anchor_generator = anchor_generator
    
    def analyze_coverage(
        self, 
        image_size: Tuple[int, int],
        feature_sizes: List[Tuple[int, int]]
    ) -> Dict[str, any]:
        """📊 Analyse la couverture des anchors"""
        
        # Génération anchors fictifs
        dummy_images = torch.zeros(1, 3, *image_size)
        dummy_features = [
            torch.zeros(1, 256, *fs) for fs in feature_sizes
        ]
        
        all_anchors = self.anchor_generator(dummy_images, dummy_features)
        
        # Statistiques par niveau
        level_stats = []
        total_anchors = 0
        
        for i, anchors in enumerate(all_anchors):
            areas = anchors[:, 2] * anchors[:, 3]
            
            stats = {
                "level": i,
                "num_anchors": len(anchors),
                "min_area": float(areas.min()),
                "max_area": float(areas.max()),
                "mean_area": float(areas.mean()),
                "std_area": float(areas.std())
            }
            level_stats.append(stats)
            total_anchors += len(anchors)
        
        return {
            "total_anchors": total_anchors,
            "levels": level_stats,
            "image_size": image_size,
            "coverage_ratio": total_anchors / (image_size[0] * image_size[1])
        }
    
    def get_anchor_statistics(self) -> Dict[str, any]:
        """📊 Statistiques des anchors de base"""
        
        all_base_areas = []
        all_base_ratios = []
        
        for base_anchors in self.anchor_generator.base_anchors:
            areas = base_anchors[:, 2] * base_anchors[:, 3]
            ratios = base_anchors[:, 2] / base_anchors[:, 3]
            
            all_base_areas.extend(areas.tolist())
            all_base_ratios.extend(ratios.tolist())
        
        return {
            "num_base_anchors": len(all_base_areas),
            "area_range": (min(all_base_areas), max(all_base_areas)),
            "ratio_range": (min(all_base_ratios), max(all_base_ratios)),
            "unique_areas": len(set(all_base_areas)),
            "unique_ratios": len(set(all_base_ratios))
        }

# Instance globale des utilitaires
anchor_utils = AnchorUtils()

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "AnchorGenerator",
    "AnchorConfig", 
    "AnchorUtils",
    "AnchorAnalyzer",
    "generate_anchors",
    "create_retinanet_anchors",
    "create_ssd_anchors", 
    "create_yolo_anchors",
    "create_mobile_anchors",
    "anchor_utils"
]