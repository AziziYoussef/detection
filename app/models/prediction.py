"""
🎯 PREDICTION - TÊTES DE PRÉDICTION POUR DÉTECTION
=================================================
Implémentation des têtes de prédiction pour classification et régression

Composants:
- PredictionHead: Tête principale combinant classification et régression
- ClassificationHead: Classification d'objets (28 classes d'objets perdus)  
- RegressionHead: Régression de boîtes englobantes
- Têtes spécialisées selon l'architecture (RetinaNet, FCOS, etc.)

Architecture:
                  FPN Features (P2-P7)
                         │
                    ┌────┴────┐
                    │         │
        ┌───────────▼─┐   ┌───▼──────────┐
        │Classification│   │  Regression  │
        │    Head      │   │    Head      │
        │              │   │              │
        │ 28 classes   │   │ 4 coords     │
        │ + objectness │   │ (x,y,w,h)    │
        └──────────────┘   └──────────────┘
                │                   │
                ▼                   ▼
        Class Predictions    Box Predictions
         [N, A, C+1]         [N, A, 4]

Optimisations:
- Partage de features entre têtes
- Convolutions séparables pour efficacité
- Normalisation adaptative
- Initialisation focal loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)

# 📋 CONFIGURATION PRÉDICTION
@dataclass
class PredictionConfig:
    """⚙️ Configuration têtes de prédiction"""
    # Architecture
    in_channels: int = 256          # Canaux d'entrée (du FPN)
    num_classes: int = 28           # Nombre de classes d'objets
    num_anchors: int = 9            # Anchors par position
    hidden_channels: int = 256      # Canaux intermédiaires
    num_layers: int = 4             # Nombre de couches
    
    # Paramètres
    use_bias: bool = True           # Bias dans convolutions
    dropout_rate: float = 0.1       # Dropout
    activation: str = "relu"        # Activation
    
    # Normalisation
    use_batch_norm: bool = False    # Batch normalization
    use_group_norm: bool = True     # Group normalization (meilleur)
    group_norm_groups: int = 32     # Groupes pour GroupNorm
    
    # Optimisations
    use_separable_conv: bool = False # Convolutions séparables
    shared_conv: bool = True        # Partager convolutions entre têtes
    
    # Initialisation
    focal_init: bool = True         # Initialisation focal loss
    prior_prob: float = 0.01        # Probabilité prior pour focal
    
    # Spécialisations
    use_objectness: bool = True     # Prédiction objectness
    use_centerness: bool = False    # Prédiction centerness (FCOS)
    
    def get_total_classes(self) -> int:
        """📊 Nombre total de classes (avec background)"""
        return self.num_classes + (1 if self.use_objectness else 0)

# 🎯 TÊTE DE CLASSIFICATION
class ClassificationHead(nn.Module):
    """🎯 Tête de classification pour détection d'objets"""
    
    def __init__(self, config: PredictionConfig):
        super().__init__()
        
        self.config = config
        self.num_outputs = config.num_anchors * config.get_total_classes()
        
        # Construction des couches
        self.layers = self._build_layers()
        
        # Couche de sortie
        self.output_conv = nn.Conv2d(
            config.hidden_channels,
            self.num_outputs,
            kernel_size=3,
            padding=1,
            bias=config.use_bias
        )
        
        # Initialisation spécialisée
        self._initialize_weights()
        
        logger.debug(f"🎯 ClassificationHead: {config.num_classes} classes, {config.num_anchors} anchors")
    
    def _build_layers(self) -> nn.ModuleList:
        """🏗️ Construction des couches intermédiaires"""
        
        layers = nn.ModuleList()
        
        for i in range(self.config.num_layers):
            in_ch = self.config.in_channels if i == 0 else self.config.hidden_channels
            out_ch = self.config.hidden_channels
            
            # Couche principale
            if self.config.use_separable_conv:
                conv = self._make_separable_conv(in_ch, out_ch)
            else:
                conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=self.config.use_bias)
            
            layer_modules = [conv]
            
            # Normalisation
            if self.config.use_group_norm:
                groups = min(self.config.group_norm_groups, out_ch)
                layer_modules.append(nn.GroupNorm(groups, out_ch))
            elif self.config.use_batch_norm:
                layer_modules.append(nn.BatchNorm2d(out_ch))
            
            # Activation
            if self.config.activation == "relu":
                layer_modules.append(nn.ReLU(inplace=True))
            elif self.config.activation == "swish":
                layer_modules.append(nn.SiLU(inplace=True))
            
            # Dropout
            if self.config.dropout_rate > 0:
                layer_modules.append(nn.Dropout2d(self.config.dropout_rate))
            
            layers.append(nn.Sequential(*layer_modules))
        
        return layers
    
    def _make_separable_conv(self, in_channels: int, out_channels: int) -> nn.Module:
        """🔧 Création convolution séparable"""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, padding=1, 
                     groups=in_channels, bias=False),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1, bias=self.config.use_bias)
        )
    
    def _initialize_weights(self):
        """⚖️ Initialisation des poids"""
        
        # Couches intermédiaires
        for module in self.layers:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
        
        # Couche de sortie - initialisation focal loss
        if self.config.focal_init:
            # Initialisation pour focal loss (réduire les faux positifs)
            bias_value = -math.log((1 - self.config.prior_prob) / self.config.prior_prob)
            nn.init.constant_(self.output_conv.bias, bias_value)
            nn.init.normal_(self.output_conv.weight, std=0.01)
        else:
            nn.init.kaiming_normal_(self.output_conv.weight)
            if self.output_conv.bias is not None:
                nn.init.constant_(self.output_conv.bias, 0)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """🔄 Forward pass classification"""
        
        predictions = []
        
        for feature in features:
            x = feature
            
            # Passage dans les couches
            for layer in self.layers:
                x = layer(x)
            
            # Prédiction finale
            pred = self.output_conv(x)
            
            # Reshape: [N, A*C, H, W] → [N, A, C, H, W] → [N, A*H*W, C]
            N, _, H, W = pred.shape
            pred = pred.view(N, self.config.num_anchors, -1, H, W)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            pred = pred.view(N, -1, self.config.get_total_classes())
            
            predictions.append(pred)
        
        return predictions

# 📐 TÊTE DE RÉGRESSION
class RegressionHead(nn.Module):
    """📐 Tête de régression pour boîtes englobantes"""
    
    def __init__(self, config: PredictionConfig):
        super().__init__()
        
        self.config = config
        # 4 coordonnées par anchor (dx, dy, dw, dh)
        self.num_outputs = config.num_anchors * 4
        
        # Centerness si FCOS
        if config.use_centerness:
            self.num_outputs += config.num_anchors  # +1 pour centerness
        
        # Construction des couches
        self.layers = self._build_layers()
        
        # Couche de sortie
        self.output_conv = nn.Conv2d(
            config.hidden_channels,
            self.num_outputs,
            kernel_size=3,
            padding=1,
            bias=config.use_bias
        )
        
        # Initialisation
        self._initialize_weights()
        
        logger.debug(f"📐 RegressionHead: 4 coords × {config.num_anchors} anchors")
    
    def _build_layers(self) -> nn.ModuleList:
        """🏗️ Construction des couches (similaire à ClassificationHead)"""
        
        layers = nn.ModuleList()
        
        for i in range(self.config.num_layers):
            in_ch = self.config.in_channels if i == 0 else self.config.hidden_channels
            out_ch = self.config.hidden_channels
            
            if self.config.use_separable_conv:
                conv = self._make_separable_conv(in_ch, out_ch)
            else:
                conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=self.config.use_bias)
            
            layer_modules = [conv]
            
            # Normalisation
            if self.config.use_group_norm:
                groups = min(self.config.group_norm_groups, out_ch)
                layer_modules.append(nn.GroupNorm(groups, out_ch))
            elif self.config.use_batch_norm:
                layer_modules.append(nn.BatchNorm2d(out_ch))
            
            # Activation
            if self.config.activation == "relu":
                layer_modules.append(nn.ReLU(inplace=True))
            elif self.config.activation == "swish":
                layer_modules.append(nn.SiLU(inplace=True))
            
            # Dropout
            if self.config.dropout_rate > 0:
                layer_modules.append(nn.Dropout2d(self.config.dropout_rate))
            
            layers.append(nn.Sequential(*layer_modules))
        
        return layers
    
    def _make_separable_conv(self, in_channels: int, out_channels: int) -> nn.Module:
        """🔧 Création convolution séparable"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, 
                     groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=self.config.use_bias)
        )
    
    def _initialize_weights(self):
        """⚖️ Initialisation des poids"""
        
        # Couches intermédiaires
        for module in self.layers:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
        
        # Couche de sortie
        nn.init.kaiming_normal_(self.output_conv.weight)
        if self.output_conv.bias is not None:
            nn.init.constant_(self.output_conv.bias, 0)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """🔄 Forward pass régression"""
        
        predictions = []
        
        for feature in features:
            x = feature
            
            # Passage dans les couches
            for layer in self.layers:
                x = layer(x)
            
            # Prédiction finale
            pred = self.output_conv(x)
            
            # Reshape: [N, A*4, H, W] → [N, A, 4, H, W] → [N, A*H*W, 4]
            N, _, H, W = pred.shape
            coords_per_anchor = 4 + (1 if self.config.use_centerness else 0)
            pred = pred.view(N, self.config.num_anchors, coords_per_anchor, H, W)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            pred = pred.view(N, -1, coords_per_anchor)
            
            predictions.append(pred)
        
        return predictions

# 🎯 TÊTE PRINCIPALE
class PredictionHead(nn.Module):
    """🎯 Tête de prédiction principale combinant classification et régression"""
    
    def __init__(self, config: PredictionConfig):
        super().__init__()
        
        self.config = config
        
        # Construction des têtes
        if config.shared_conv:
            # Convolutions partagées
            self.shared_layers = self._build_shared_layers()
            
            # Têtes spécialisées (plus légères)
            cls_config = self._get_lightweight_config()
            reg_config = self._get_lightweight_config()
        else:
            # Têtes indépendantes
            self.shared_layers = None
            cls_config = config
            reg_config = config
        
        self.classification_head = ClassificationHead(cls_config)
        self.regression_head = RegressionHead(reg_config)
        
        logger.info(f"🎯 PredictionHead: {config.num_classes} classes, partage={config.shared_conv}")
    
    def _build_shared_layers(self) -> nn.ModuleList:
        """🏗️ Construction des couches partagées"""
        
        layers = nn.ModuleList()
        
        # Moins de couches partagées
        num_shared = max(1, self.config.num_layers // 2)
        
        for i in range(num_shared):
            in_ch = self.config.in_channels if i == 0 else self.config.hidden_channels
            out_ch = self.config.hidden_channels
            
            conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=self.config.use_bias)
            
            layer_modules = [conv]
            
            # Normalisation
            if self.config.use_group_norm:
                groups = min(self.config.group_norm_groups, out_ch)
                layer_modules.append(nn.GroupNorm(groups, out_ch))
            
            # Activation
            if self.config.activation == "relu":
                layer_modules.append(nn.ReLU(inplace=True))
            elif self.config.activation == "swish":
                layer_modules.append(nn.SiLU(inplace=True))
            
            layers.append(nn.Sequential(*layer_modules))
        
        return layers
    
    def _get_lightweight_config(self) -> PredictionConfig:
        """⚡ Configuration allégée pour têtes spécialisées"""
        config = PredictionConfig(
            in_channels=self.config.hidden_channels,  # Depuis shared layers
            num_classes=self.config.num_classes,
            num_anchors=self.config.num_anchors,
            hidden_channels=self.config.hidden_channels,
            num_layers=max(1, self.config.num_layers - len(self.shared_layers)) if self.shared_layers else self.config.num_layers,
            use_bias=self.config.use_bias,
            dropout_rate=self.config.dropout_rate,
            activation=self.config.activation,
            use_group_norm=self.config.use_group_norm,
            group_norm_groups=self.config.group_norm_groups,
            use_separable_conv=self.config.use_separable_conv,
            focal_init=self.config.focal_init,
            prior_prob=self.config.prior_prob,
            use_objectness=self.config.use_objectness,
            use_centerness=self.config.use_centerness
        )
        return config
    
    def forward(self, features: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """🔄 Forward pass tête principale"""
        
        # Features partagées si configuré
        if self.shared_layers is not None:
            shared_features = []
            for feature in features:
                x = feature
                for layer in self.shared_layers:
                    x = layer(x)
                shared_features.append(x)
            input_features = shared_features
        else:
            input_features = features
        
        # Prédictions des têtes spécialisées
        class_predictions = self.classification_head(input_features)
        box_predictions = self.regression_head(input_features)
        
        predictions = {
            'scores': class_predictions,  # [N, A*H*W, C] pour chaque niveau
            'boxes': box_predictions      # [N, A*H*W, 4] pour chaque niveau
        }
        
        return predictions

# 🔧 TÊTES SPÉCIALISÉES
class RetinaNetHead(PredictionHead):
    """🔧 Tête spécialisée RetinaNet"""
    
    def __init__(self, config: PredictionConfig):
        # Configuration spécifique RetinaNet
        config.use_objectness = False  # Pas d'objectness, juste classes
        config.focal_init = True       # Initialisation focal loss
        config.shared_conv = False     # Têtes séparées
        
        super().__init__(config)

class FCOSHead(PredictionHead):
    """🔧 Tête spécialisée FCOS (sans anchors)"""
    
    def __init__(self, config: PredictionConfig):
        # Configuration spécifique FCOS
        config.num_anchors = 1         # Pas d'anchors multiples
        config.use_centerness = True   # Prédiction centerness
        config.use_objectness = False  # Pas d'objectness
        
        super().__init__(config)

class YOLOHead(PredictionHead):
    """🔧 Tête spécialisée YOLO"""
    
    def __init__(self, config: PredictionConfig):
        # Configuration spécifique YOLO
        config.use_objectness = True   # Prédiction objectness
        config.focal_init = False      # Pas d'init focal
        config.shared_conv = True      # Convolutions partagées
        
        super().__init__(config)

# 🏭 FONCTIONS FACTORY
def create_prediction_head(
    head_type: str = "standard",
    num_classes: int = 28,
    in_channels: int = 256,
    **kwargs
) -> PredictionHead:
    """
    🏭 Factory pour créer une tête de prédiction
    
    Args:
        head_type: Type de tête ("standard", "retinanet", "fcos", "yolo")
        num_classes: Nombre de classes
        in_channels: Canaux d'entrée (du FPN)
        **kwargs: Arguments additionnels
        
    Returns:
        Instance de tête de prédiction
    """
    
    # Configuration de base
    config = PredictionConfig(
        in_channels=in_channels,
        num_classes=num_classes,
        **kwargs
    )
    
    # Création selon le type
    if head_type == "retinanet":
        return RetinaNetHead(config)
    elif head_type == "fcos":
        return FCOSHead(config)
    elif head_type == "yolo":
        return YOLOHead(config)
    else:
        return PredictionHead(config)

# 🎯 UTILITAIRES
def calculate_prediction_size(
    config: PredictionConfig,
    feature_sizes: List[Tuple[int, int]]
) -> Dict[str, int]:
    """📊 Calcule la taille des prédictions"""
    
    total_anchors = 0
    total_class_params = 0
    total_box_params = 0
    
    for h, w in feature_sizes:
        anchors_per_level = h * w * config.num_anchors
        total_anchors += anchors_per_level
        
        total_class_params += anchors_per_level * config.get_total_classes()
        coords = 4 + (1 if config.use_centerness else 0)
        total_box_params += anchors_per_level * coords
    
    return {
        "total_anchors": total_anchors,
        "class_predictions": total_class_params,
        "box_predictions": total_box_params,
        "total_predictions": total_class_params + total_box_params
    }

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "PredictionHead",
    "ClassificationHead",
    "RegressionHead",
    "RetinaNetHead",
    "FCOSHead", 
    "YOLOHead",
    "PredictionConfig",
    "create_prediction_head",
    "calculate_prediction_size"
]