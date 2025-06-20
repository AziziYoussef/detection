"""
🏗️ FPN - FEATURE PYRAMID NETWORK
================================
Implémentation du Feature Pyramid Network pour détection multi-échelle

Fonctionnalités:
- Fusion de features multi-résolution
- Top-down pathway avec lateral connections
- Support de différentes architectures backbone
- Optimisations pour vitesse et mémoire
- Configuration flexible du nombre de niveaux
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

@dataclass
class FPNConfig:
    """⚙️ Configuration du FPN"""
    in_channels: List[int]  # Canaux d'entrée par niveau
    out_channels: int = 256  # Canaux de sortie
    num_levels: int = 5  # Nombre de niveaux FPN
    extra_levels: int = 0  # Niveaux supplémentaires (P6, P7)
    use_bias: bool = True
    activation: str = "relu"
    norm_type: str = "none"  # "none", "bn", "gn"
    use_separable_conv: bool = False
    
    def __post_init__(self):
        """Validation de la configuration"""
        if len(self.in_channels) < 3:
            raise ValueError("Au moins 3 niveaux d'entrée requis")
        if self.num_levels < 3:
            raise ValueError("Au moins 3 niveaux FPN requis")

class SeparableConv2d(nn.Module):
    """🔧 Convolution séparable en profondeur"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, bias: bool = True):
        super().__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class FPNLayer(nn.Module):
    """🏗️ Couche individuelle du FPN"""
    
    def __init__(self, config: FPNConfig, level: int):
        super().__init__()
        self.config = config
        self.level = level
        
        # Convolution latérale (1x1)
        in_channels = config.in_channels[level] if level < len(config.in_channels) else config.out_channels
        
        self.lateral_conv = nn.Conv2d(
            in_channels, config.out_channels, 1, bias=config.use_bias
        )
        
        # Convolution de sortie (3x3)
        if config.use_separable_conv:
            self.output_conv = SeparableConv2d(
                config.out_channels, config.out_channels, 3, 1, 1, config.use_bias
            )
        else:
            self.output_conv = nn.Conv2d(
                config.out_channels, config.out_channels, 3, 1, 1, bias=config.use_bias
            )
        
        # Normalisation
        self.norm = self._build_norm_layer()
        
        # Activation
        self.activation = self._build_activation()
        
        # Initialisation des poids
        self._initialize_weights()
    
    def _build_norm_layer(self) -> Optional[nn.Module]:
        """🔧 Construit la couche de normalisation"""
        if self.config.norm_type == "bn":
            return nn.BatchNorm2d(self.config.out_channels)
        elif self.config.norm_type == "gn":
            return nn.GroupNorm(32, self.config.out_channels)
        else:
            return None
    
    def _build_activation(self) -> nn.Module:
        """🔧 Construit la fonction d'activation"""
        if self.config.activation == "relu":
            return nn.ReLU(inplace=True)
        elif self.config.activation == "swish":
            return nn.SiLU(inplace=True)
        elif self.config.activation == "gelu":
            return nn.GELU()
        else:
            return nn.ReLU(inplace=True)
    
    def _initialize_weights(self):
        """⚖️ Initialise les poids"""
        for module in [self.lateral_conv, self.output_conv]:
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, lateral_input: torch.Tensor, top_down_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """🔄 Forward pass de la couche FPN"""
        
        # Convolution latérale
        lateral = self.lateral_conv(lateral_input)
        
        # Fusion avec top-down si disponible
        if top_down_input is not None:
            # Upsampling du top-down pour correspondre à la taille latérale
            if top_down_input.shape[-2:] != lateral.shape[-2:]:
                top_down_input = F.interpolate(
                    top_down_input, size=lateral.shape[-2:], 
                    mode='nearest'
                )
            
            # Addition
            lateral = lateral + top_down_input
        
        # Convolution de sortie
        output = self.output_conv(lateral)
        
        # Normalisation
        if self.norm is not None:
            output = self.norm(output)
        
        # Activation
        output = self.activation(output)
        
        return output

class FeaturePyramidNetwork(nn.Module):
    """🏗️ Feature Pyramid Network complet"""
    
    def __init__(self, config: FPNConfig):
        super().__init__()
        self.config = config
        
        # Couches FPN pour chaque niveau
        self.fpn_layers = nn.ModuleList()
        
        # Niveaux principaux (ex: P3, P4, P5)
        for level in range(config.num_levels):
            self.fpn_layers.append(FPNLayer(config, level))
        
        # Niveaux supplémentaires si demandés (P6, P7)
        if config.extra_levels > 0:
            self.extra_layers = nn.ModuleList()
            
            for i in range(config.extra_levels):
                if i == 0:
                    # P6: stride 2 sur P5
                    layer = nn.Conv2d(
                        config.out_channels, config.out_channels, 
                        3, stride=2, padding=1, bias=config.use_bias
                    )
                else:
                    # P7+: stride 2 sur niveau précédent
                    layer = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            config.out_channels, config.out_channels,
                            3, stride=2, padding=1, bias=config.use_bias
                        )
                    )
                
                self.extra_layers.append(layer)
        else:
            self.extra_layers = None
        
        logger.info(f"🏗️ FPN initialisé: {config.num_levels} niveaux + {config.extra_levels} extra")
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        🔄 Forward pass du FPN
        
        Args:
            features: Dict avec clés 'C2', 'C3', 'C4', 'C5' (features backbone)
            
        Returns:
            Dict avec clés 'P2', 'P3', 'P4', 'P5' [+ 'P6', 'P7'] (features FPN)
        """
        
        # Extraction des features d'entrée dans l'ordre inverse (top-down)
        input_features = []
        feature_names = []
        
        for i in range(self.config.num_levels):
            level_name = f"C{i + 2}"  # C2, C3, C4, C5, ...
            if level_name in features:
                input_features.append(features[level_name])
                feature_names.append(f"P{i + 2}")
            else:
                logger.warning(f"⚠️ Feature manquante: {level_name}")
        
        # Inversion pour traitement top-down
        input_features = input_features[::-1]
        feature_names = feature_names[::-1]
        
        # Traitement top-down
        fpn_features = {}
        top_down = None
        
        for i, (feature, name) in enumerate(zip(input_features, feature_names)):
            fpn_layer = self.fpn_layers[len(input_features) - 1 - i]
            
            # Forward de la couche FPN
            fpn_feature = fpn_layer(feature, top_down)
            fpn_features[name] = fpn_feature
            
            # Mise à jour pour le niveau suivant
            top_down = fpn_feature
        
        # Niveaux supplémentaires
        if self.extra_layers is not None:
            last_feature = fpn_features[f"P{self.config.num_levels + 1}"]
            
            for i, extra_layer in enumerate(self.extra_layers):
                extra_feature = extra_layer(last_feature)
                fpn_features[f"P{self.config.num_levels + 2 + i}"] = extra_feature
                last_feature = extra_feature
        
        return fpn_features
    
    def get_output_channels(self) -> int:
        """📏 Retourne le nombre de canaux de sortie"""
        return self.config.out_channels
    
    def get_output_levels(self) -> List[str]:
        """📋 Retourne la liste des niveaux de sortie"""
        levels = [f"P{i + 2}" for i in range(self.config.num_levels)]
        
        if self.config.extra_levels > 0:
            extra_levels = [f"P{self.config.num_levels + 2 + i}" for i in range(self.config.extra_levels)]
            levels.extend(extra_levels)
        
        return levels

class SimpleFPN(nn.Module):
    """🏗️ Version simplifiée du FPN pour modèles légers"""
    
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        
        # Convolutions 1x1 pour uniformiser les canaux
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, out_channels, 1, bias=False)
            )
        
        # Convolutions 3x3 pour lisser
        self.smooth_convs = nn.ModuleList()
        for _ in range(len(in_channels)):
            self.smooth_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """🔄 Forward pass FPN simplifié"""
        
        # Récupération des features
        feature_list = []
        names = []
        
        for i, (name, feature) in enumerate(features.items()):
            if name.startswith('C'):
                feature_list.append(feature)
                names.append(name.replace('C', 'P'))
        
        # Inversion pour top-down
        feature_list = feature_list[::-1]
        names = names[::-1]
        
        # Traitement
        results = {}
        prev = None
        
        for i, (feature, name) in enumerate(zip(feature_list, names)):
            # Convolution latérale
            lateral = self.lateral_convs[len(feature_list) - 1 - i](feature)
            
            # Fusion avec niveau supérieur
            if prev is not None:
                lateral = lateral + F.interpolate(
                    prev, size=lateral.shape[-2:], mode='nearest'
                )
            
            # Lissage
            smooth = self.smooth_convs[len(feature_list) - 1 - i](lateral)
            results[name] = smooth
            
            prev = smooth
        
        return results

class PAN(nn.Module):
    """🔗 Path Aggregation Network (extension du FPN)"""
    
    def __init__(self, config: FPNConfig):
        super().__init__()
        self.config = config
        
        # FPN de base
        self.fpn = FeaturePyramidNetwork(config)
        
        # Bottom-up path augmentation
        self.bottom_up_layers = nn.ModuleList()
        
        for i in range(config.num_levels - 1):
            self.bottom_up_layers.append(
                nn.Conv2d(config.out_channels, config.out_channels, 3, stride=2, padding=1)
            )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """🔄 Forward pass PAN"""
        
        # FPN standard
        fpn_features = self.fpn(features)
        
        # Bottom-up augmentation
        feature_names = sorted([name for name in fpn_features.keys() if name.startswith('P')])
        
        # Partir du plus petit niveau et remonter
        enhanced_features = {}
        prev = None
        
        for i, name in enumerate(feature_names):
            current = fpn_features[name]
            
            if prev is not None:
                # Downsample du niveau précédent
                downsampled = self.bottom_up_layers[i-1](prev)
                current = current + downsampled
            
            enhanced_features[name] = current
            prev = current
        
        return enhanced_features

# === FONCTIONS UTILITAIRES ===

def create_fpn(
    backbone_channels: List[int],
    fpn_channels: int = 256,
    num_levels: int = 5,
    fpn_type: str = "standard"
) -> nn.Module:
    """🏭 Factory pour créer un FPN"""
    
    if fpn_type == "standard":
        config = FPNConfig(
            in_channels=backbone_channels,
            out_channels=fpn_channels,
            num_levels=num_levels
        )
        return FeaturePyramidNetwork(config)
    
    elif fpn_type == "simple":
        return SimpleFPN(backbone_channels, fpn_channels)
    
    elif fpn_type == "pan":
        config = FPNConfig(
            in_channels=backbone_channels,
            out_channels=fpn_channels,
            num_levels=num_levels
        )
        return PAN(config)
    
    else:
        raise ValueError(f"Type FPN inconnu: {fpn_type}")

def get_fpn_config_for_backbone(backbone_name: str) -> FPNConfig:
    """⚙️ Configuration FPN optimisée par backbone"""
    
    configs = {
        "mobilenet_v2": FPNConfig(
            in_channels=[24, 32, 96, 320],
            out_channels=128,
            num_levels=4,
            use_separable_conv=True
        ),
        "mobilenet_v3_large": FPNConfig(
            in_channels=[24, 40, 112, 960],
            out_channels=256,
            num_levels=4,
            use_separable_conv=True
        ),
        "efficientnet-b0": FPNConfig(
            in_channels=[24, 40, 112, 320],
            out_channels=256,
            num_levels=4
        ),
        "resnet50": FPNConfig(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_levels=4
        )
    }
    
    return configs.get(backbone_name, FPNConfig(
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_levels=4
    ))

# === EXPORTS ===
__all__ = [
    "FeaturePyramidNetwork",
    "SimpleFPN", 
    "PAN",
    "FPNConfig",
    "FPNLayer",
    "create_fpn",
    "get_fpn_config_for_backbone"
]