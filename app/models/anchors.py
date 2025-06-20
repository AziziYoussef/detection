"""
⚓ ANCHORS - GÉNÉRATEUR D'ANCHORS POUR DÉTECTION D'OBJETS
======================================================
Implémentation du système d'anchors pour réseaux de détection

Fonctionnalités:
- Génération d'anchors multi-échelle
- Support de différents ratios d'aspect
- Optimisation pour différentes résolutions
- Compatible avec FPN (Feature Pyramid Networks)
- Anchors adaptatifs selon le dataset
"""

import logging
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class AnchorConfig:
    """⚙️ Configuration du générateur d'anchors"""
    sizes: List[float]  # Tailles d'anchors (en pixels)
    ratios: List[float]  # Ratios d'aspect (width/height)
    scales: List[float]  # Échelles multiplicatives
    strides: List[int]  # Strides pour chaque niveau FPN
    num_levels: int = 5  # Nombre de niveaux FPN
    clip_to_image: bool = True  # Clipper aux dimensions de l'image
    
    def __post_init__(self):
        """Validation et calculs automatiques"""
        if len(self.sizes) != self.num_levels:
            # Si une seule taille, la distribuer sur tous les niveaux
            if len(self.sizes) == 1:
                base_size = self.sizes[0]
                self.sizes = [base_size * (2 ** i) for i in range(self.num_levels)]
            else:
                raise ValueError(f"sizes doit avoir {self.num_levels} éléments ou 1 élément")
        
        if not self.strides:
            # Strides par défaut : 2^(n+3) pour FPN standard
            self.strides = [2 ** (i + 3) for i in range(self.num_levels)]
        
        if len(self.strides) != self.num_levels:
            raise ValueError(f"strides doit avoir {self.num_levels} éléments")

class AnchorGenerator(nn.Module):
    """⚓ Générateur d'anchors principal"""
    
    def __init__(self, config: AnchorConfig):
        super().__init__()
        self.config = config
        
        # Pré-calcul des anchors de base pour chaque niveau
        self.base_anchors = self._generate_base_anchors()
        
        # Cache pour éviter la régénération
        self._anchor_cache = {}
        
        logger.info(f"⚓ AnchorGenerator initialisé: {len(self.base_anchors)} niveaux")
        
        # Log des informations
        for i, (stride, base_anchor) in enumerate(zip(config.strides, self.base_anchors)):
            logger.debug(f"Niveau {i}: stride={stride}, {len(base_anchor)} anchors")
    
    def _generate_base_anchors(self) -> List[torch.Tensor]:
        """🔧 Génère les anchors de base pour chaque niveau"""
        
        base_anchors = []
        
        for level in range(self.config.num_levels):
            size = self.config.sizes[level]
            stride = self.config.strides[level]
            
            # Génération des anchors pour ce niveau
            level_anchors = []
            
            for ratio in self.config.ratios:
                for scale in self.config.scales:
                    # Calcul des dimensions
                    anchor_size = size * scale
                    w = anchor_size * math.sqrt(ratio)
                    h = anchor_size / math.sqrt(ratio)
                    
                    # Anchor centré en (0, 0)
                    anchor = torch.tensor([
                        -w / 2, -h / 2, w / 2, h / 2
                    ], dtype=torch.float32)
                    
                    level_anchors.append(anchor)
            
            # Stack des anchors pour ce niveau
            base_anchors.append(torch.stack(level_anchors))
        
        return base_anchors
    
    def forward(
        self, 
        image: torch.Tensor, 
        features: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        🔄 Génère les anchors pour une image
        
        Args:
            image: Image d'entrée [B, C, H, W]
            features: Features FPN par niveau
            
        Returns:
            Liste d'anchors par niveau [num_anchors, 4]
        """
        
        device = image.device
        image_height, image_width = image.shape[-2:]
        
        # Clé de cache
        cache_key = (image_height, image_width, device)
        
        if cache_key in self._anchor_cache:
            return self._anchor_cache[cache_key]
        
        all_anchors = []
        
        # Génération pour chaque niveau FPN
        for level, (level_name, feature) in enumerate(sorted(features.items())):
            if level >= len(self.base_anchors):
                break
            
            # Dimensions de la feature map
            feature_height, feature_width = feature.shape[-2:]
            stride = self.config.strides[level]
            
            # Génération de la grille d'anchors
            level_anchors = self._generate_level_anchors(
                feature_height, feature_width, stride, level, device
            )
            
            all_anchors.append(level_anchors)
        
        # Clipping aux dimensions de l'image si demandé
        if self.config.clip_to_image:
            all_anchors = [
                self._clip_anchors_to_image(anchors, image_width, image_height)
                for anchors in all_anchors
            ]
        
        # Mise en cache
        self._anchor_cache[cache_key] = all_anchors
        
        return all_anchors
    
    def _generate_level_anchors(
        self, 
        feature_height: int, 
        feature_width: int,
        stride: int,
        level: int,
        device: torch.device
    ) -> torch.Tensor:
        """🔧 Génère les anchors pour un niveau spécifique"""
        
        # Anchors de base pour ce niveau
        base_anchors = self.base_anchors[level].to(device)
        num_base_anchors = len(base_anchors)
        
        # Grille de positions
        shift_x = torch.arange(0, feature_width, device=device) * stride
        shift_y = torch.arange(0, feature_height, device=device) * stride
        
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        
        # Shifts pour chaque position
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=2)
        shifts = shifts.reshape(-1, 4)
        
        # Application des shifts aux anchors de base
        anchors = base_anchors.view(1, num_base_anchors, 4) + shifts.view(-1, 1, 4)
        anchors = anchors.reshape(-1, 4)
        
        return anchors
    
    def _clip_anchors_to_image(
        self, 
        anchors: torch.Tensor, 
        image_width: int, 
        image_height: int
    ) -> torch.Tensor:
        """✂️ Clippe les anchors aux dimensions de l'image"""
        
        clipped = anchors.clone()
        clipped[:, 0] = torch.clamp(clipped[:, 0], min=0, max=image_width)   # x1
        clipped[:, 1] = torch.clamp(clipped[:, 1], min=0, max=image_height)  # y1
        clipped[:, 2] = torch.clamp(clipped[:, 2], min=0, max=image_width)   # x2
        clipped[:, 3] = torch.clamp(clipped[:, 3], min=0, max=image_height)  # y2
        
        return clipped
    
    def get_num_anchors_per_level(self) -> int:
        """📊 Retourne le nombre d'anchors par position"""
        return len(self.config.ratios) * len(self.config.scales)
    
    def get_total_anchors(self, image_size: Tuple[int, int]) -> int:
        """📊 Retourne le nombre total d'anchors pour une taille d'image"""
        
        image_height, image_width = image_size
        total = 0
        
        for level in range(self.config.num_levels):
            stride = self.config.strides[level]
            feature_height = image_height // stride
            feature_width = image_width // stride
            level_anchors = feature_height * feature_width * self.get_num_anchors_per_level()
            total += level_anchors
        
        return total
    
    def clear_cache(self):
        """🧹 Vide le cache d'anchors"""
        self._anchor_cache.clear()

class AdaptiveAnchorGenerator(nn.Module):
    """⚓ Générateur d'anchors adaptatif basé sur les statistiques du dataset"""
    
    def __init__(self, config: AnchorConfig, bbox_stats: Optional[Dict] = None):
        super().__init__()
        self.config = config
        self.bbox_stats = bbox_stats or {}
        
        # Adaptation des anchors selon les statistiques
        if bbox_stats:
            self.config = self._adapt_config_to_stats(config, bbox_stats)
        
        # Générateur de base
        self.anchor_generator = AnchorGenerator(self.config)
    
    def _adapt_config_to_stats(self, config: AnchorConfig, stats: Dict) -> AnchorConfig:
        """🎯 Adapte la configuration selon les statistiques du dataset"""
        
        adapted_config = AnchorConfig(
            sizes=config.sizes.copy(),
            ratios=config.ratios.copy(),
            scales=config.scales.copy(),
            strides=config.strides.copy(),
            num_levels=config.num_levels,
            clip_to_image=config.clip_to_image
        )
        
        # Adaptation des ratios selon la distribution des bounding boxes
        if 'aspect_ratios' in stats:
            common_ratios = stats['aspect_ratios']
            # Garde les ratios les plus fréquents
            adapted_config.ratios = common_ratios[:len(config.ratios)]
            logger.info(f"🎯 Ratios adaptés: {adapted_config.ratios}")
        
        # Adaptation des tailles selon la distribution des aires
        if 'area_distribution' in stats:
            area_percentiles = stats['area_distribution']
            # Ajuste les tailles selon les percentiles
            base_sizes = []
            for i, percentile in enumerate([25, 50, 75, 90, 95]):
                if f'p{percentile}' in area_percentiles:
                    size = math.sqrt(area_percentiles[f'p{percentile}'])
                    base_sizes.append(size)
            
            if base_sizes:
                adapted_config.sizes = base_sizes[:config.num_levels]
                logger.info(f"🎯 Tailles adaptées: {adapted_config.sizes}")
        
        return adapted_config
    
    def forward(self, image: torch.Tensor, features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """🔄 Forward pass du générateur adaptatif"""
        return self.anchor_generator(image, features)

class DynamicAnchorGenerator(nn.Module):
    """⚓ Générateur d'anchors dynamique avec apprentissage"""
    
    def __init__(self, config: AnchorConfig):
        super().__init__()
        self.config = config
        
        # Paramètres apprenables pour ajuster les anchors
        self.anchor_adjustments = nn.ParameterList()
        
        for level in range(config.num_levels):
            num_anchors = len(config.ratios) * len(config.scales)
            # Paramètres d'ajustement pour chaque anchor de base
            adjustment = nn.Parameter(torch.zeros(num_anchors, 4))
            self.anchor_adjustments.append(adjustment)
        
        # Générateur de base
        self.base_generator = AnchorGenerator(config)
        
        logger.info("⚓ DynamicAnchorGenerator initialisé avec paramètres apprenables")
    
    def forward(self, image: torch.Tensor, features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """🔄 Génère des anchors avec ajustements appris"""
        
        # Anchors de base
        base_anchors = self.base_generator(image, features)
        
        # Application des ajustements appris
        adjusted_anchors = []
        
        for level, anchors in enumerate(base_anchors):
            if level < len(self.anchor_adjustments):
                adjustment = self.anchor_adjustments[level]
                
                # Reshape pour application
                num_positions = anchors.shape[0] // adjustment.shape[0]
                adjustment_expanded = adjustment.repeat(num_positions, 1)
                
                # Application de l'ajustement
                adjusted = anchors + adjustment_expanded
                adjusted_anchors.append(adjusted)
            else:
                adjusted_anchors.append(anchors)
        
        return adjusted_anchors

class CenterNetAnchors(nn.Module):
    """⚓ Générateur de points centraux pour CenterNet-style"""
    
    def __init__(self, strides: List[int] = [4, 8, 16, 32, 64]):
        super().__init__()
        self.strides = strides
    
    def forward(self, image: torch.Tensor, features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """🔄 Génère des points centraux au lieu d'anchors"""
        
        device = image.device
        all_centers = []
        
        for level, (level_name, feature) in enumerate(sorted(features.items())):
            if level >= len(self.strides):
                break
            
            feature_height, feature_width = feature.shape[-2:]
            stride = self.strides[level]
            
            # Grille de centres
            shift_x = (torch.arange(0, feature_width, device=device) + 0.5) * stride
            shift_y = (torch.arange(0, feature_height, device=device) + 0.5) * stride
            
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            
            # Points centraux [x, y]
            centers = torch.stack([shift_x.flatten(), shift_y.flatten()], dim=1)
            all_centers.append(centers)
        
        return all_centers

# === FONCTIONS UTILITAIRES ===

def create_anchor_generator(
    anchor_type: str = "standard",
    sizes: List[float] = [32, 64, 128, 256, 512],
    ratios: List[float] = [0.5, 1.0, 2.0],
    scales: List[float] = [1.0, 1.26, 1.587],
    **kwargs
) -> nn.Module:
    """🏭 Factory pour créer un générateur d'anchors"""
    
    config = AnchorConfig(
        sizes=sizes,
        ratios=ratios,
        scales=scales,
        **kwargs
    )
    
    if anchor_type == "standard":
        return AnchorGenerator(config)
    elif anchor_type == "adaptive":
        return AdaptiveAnchorGenerator(config)
    elif anchor_type == "dynamic":
        return DynamicAnchorGenerator(config)
    elif anchor_type == "centernet":
        return CenterNetAnchors()
    else:
        raise ValueError(f"Type d'anchor générateur inconnu: {anchor_type}")

def get_anchor_config_for_model(model_type: str) -> AnchorConfig:
    """⚙️ Configuration d'anchors optimisée par type de modèle"""
    
    configs = {
        "epoch_30": AnchorConfig(
            sizes=[32, 64, 128, 256, 512],
            ratios=[0.5, 1.0, 2.0],
            scales=[1.0, 1.26, 1.587],
            strides=[8, 16, 32, 64, 128],
            num_levels=5
        ),
        "fast": AnchorConfig(
            sizes=[32, 64, 128, 256],
            ratios=[0.5, 1.0, 2.0],
            scales=[1.0, 1.41],  # Moins d'échelles pour vitesse
            strides=[8, 16, 32, 64],
            num_levels=4
        ),
        "mobile": AnchorConfig(
            sizes=[64, 128, 256],
            ratios=[1.0, 2.0],  # Ratios simplifiés
            scales=[1.0],       # Une seule échelle
            strides=[16, 32, 64],
            num_levels=3
        )
    }
    
    return configs.get(model_type, configs["epoch_30"])

def analyze_bbox_statistics(bboxes: List[List[float]]) -> Dict:
    """📊 Analyse les statistiques des bounding boxes pour adaptation"""
    
    if not bboxes:
        return {}
    
    # Conversion en numpy pour calculs
    boxes = np.array(bboxes)  # Format: [x1, y1, x2, y2]
    
    # Calcul des dimensions
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    aspect_ratios = widths / heights
    
    # Statistiques
    stats = {
        'num_boxes': len(boxes),
        'area_distribution': {
            'mean': float(np.mean(areas)),
            'std': float(np.std(areas)),
            'p25': float(np.percentile(areas, 25)),
            'p50': float(np.percentile(areas, 50)),
            'p75': float(np.percentile(areas, 75)),
            'p90': float(np.percentile(areas, 90)),
            'p95': float(np.percentile(areas, 95))
        },
        'aspect_ratios': {
            'mean': float(np.mean(aspect_ratios)),
            'std': float(np.std(aspect_ratios)),
            'common_ratios': []
        }
    }
    
    # Ratios les plus fréquents (par bins)
    hist, bins = np.histogram(aspect_ratios, bins=20)
    common_bins = bins[np.argsort(hist)[-5:]]  # Top 5
    stats['aspect_ratios']['common_ratios'] = [float(r) for r in sorted(common_bins)]
    
    return stats

def visualize_anchors(
    anchors: List[torch.Tensor],
    image_size: Tuple[int, int],
    max_anchors_per_level: int = 100
) -> Dict[str, List[List[float]]]:
    """👁️ Prépare les anchors pour visualisation"""
    
    visualization_data = {}
    
    for level, level_anchors in enumerate(anchors):
        # Limitation pour visualisation
        num_anchors = min(len(level_anchors), max_anchors_per_level)
        sample_anchors = level_anchors[:num_anchors]
        
        # Conversion en listes pour sérialisation
        anchor_list = sample_anchors.cpu().numpy().tolist()
        
        visualization_data[f"level_{level}"] = anchor_list
    
    return visualization_data

# === EXPORTS ===
__all__ = [
    "AnchorGenerator",
    "AdaptiveAnchorGenerator",
    "DynamicAnchorGenerator", 
    "CenterNetAnchors",
    "AnchorConfig",
    "create_anchor_generator",
    "get_anchor_config_for_model",
    "analyze_bbox_statistics",
    "visualize_anchors"
]