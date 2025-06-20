"""
🎯 PREDICTION - TÊTES DE PRÉDICTION POUR DÉTECTION D'OBJETS
=========================================================
Implémentation des têtes de prédiction pour classification et régression

Fonctionnalités:
- Classification multi-classes
- Régression de bounding boxes
- Support de différentes architectures (shared/separate heads)
- Optimisations pour vitesse et précision
- Initialisation adaptée pour la détection
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

@dataclass
class PredictionConfig:
    """⚙️ Configuration des têtes de prédiction"""
    in_channels: int  # Canaux d'entrée depuis FPN
    num_classes: int  # Nombre de classes (background inclus ou non selon architecture)
    num_anchors: int  # Nombre d'anchors par position
    hidden_channels: int = 256  # Canaux des couches cachées
    num_layers: int = 4  # Nombre de couches
    use_bias: bool = True
    dropout_rate: float = 0.0
    activation: str = "relu"  # "relu", "swish", "gelu"
    norm_type: str = "none"  # "none", "bn", "gn"
    share_heads: bool = True  # Partager les couches entre cls et reg
    use_separable_conv: bool = False
    prior_prob: float = 0.01  # Probabilité a priori pour initialisation
    
    def __post_init__(self):
        """Validation de la configuration"""
        if self.num_classes <= 0:
            raise ValueError("num_classes doit être > 0")
        if self.num_anchors <= 0:
            raise ValueError("num_anchors doit être > 0")

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

class ConvBlock(nn.Module):
    """🧱 Bloc de convolution avec normalisation et activation"""
    
    def __init__(self, in_channels: int, out_channels: int, config: PredictionConfig):
        super().__init__()
        
        # Convolution
        if config.use_separable_conv:
            self.conv = SeparableConv2d(in_channels, out_channels, 3, 1, 1, config.use_bias)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=config.use_bias)
        
        # Normalisation
        self.norm = self._build_norm_layer(out_channels, config.norm_type)
        
        # Activation
        self.activation = self._build_activation(config.activation)
        
        # Dropout
        self.dropout = nn.Dropout2d(config.dropout_rate) if config.dropout_rate > 0 else None
    
    def _build_norm_layer(self, channels: int, norm_type: str) -> Optional[nn.Module]:
        """🔧 Construit la couche de normalisation"""
        if norm_type == "bn":
            return nn.BatchNorm2d(channels)
        elif norm_type == "gn":
            return nn.GroupNorm(32, channels)
        else:
            return None
    
    def _build_activation(self, activation: str) -> nn.Module:
        """🔧 Construit la fonction d'activation"""
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "swish":
            return nn.SiLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        else:
            return nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        
        if self.norm is not None:
            x = self.norm(x)
        
        x = self.activation(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x

class ClassificationHead(nn.Module):
    """🏷️ Tête de classification"""
    
    def __init__(self, config: PredictionConfig):
        super().__init__()
        self.config = config
        
        # Couches cachées
        self.conv_layers = nn.ModuleList()
        
        for i in range(config.num_layers):
            in_ch = config.in_channels if i == 0 else config.hidden_channels
            out_ch = config.hidden_channels
            
            self.conv_layers.append(ConvBlock(in_ch, out_ch, config))
        
        # Couche de sortie
        self.output_conv = nn.Conv2d(
            config.hidden_channels,
            config.num_anchors * config.num_classes,
            3, padding=1, bias=config.use_bias
        )
        
        # Initialisation spécialisée
        self._initialize_weights()
    
    def _initialize_weights(self):
        """⚖️ Initialisation des poids pour la classification"""
        
        # Couches cachées
        for layer in self.conv_layers:
            if hasattr(layer, 'conv'):
                nn.init.kaiming_normal_(layer.conv.weight, mode='fan_out', nonlinearity='relu')
                if layer.conv.bias is not None:
                    nn.init.constant_(layer.conv.bias, 0)
        
        # Couche de sortie - initialisation spéciale pour éviter l'instabilité
        nn.init.kaiming_normal_(self.output_conv.weight, mode='fan_out', nonlinearity='relu')
        
        if self.output_conv.bias is not None:
            # Biais initialisé pour avoir une probabilité a priori faible
            bias_init = -math.log((1 - self.config.prior_prob) / self.config.prior_prob)
            nn.init.constant_(self.output_conv.bias, bias_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """🔄 Forward pass classification"""
        
        for layer in self.conv_layers:
            x = layer(x)
        
        # Prédiction finale
        classification = self.output_conv(x)
        
        # Reshape: [B, A*C, H, W] → [B, A, C, H, W] → [B, H, W, A, C]
        B, _, H, W = classification.shape
        A = self.config.num_anchors
        C = self.config.num_classes
        
        classification = classification.view(B, A, C, H, W)
        classification = classification.permute(0, 3, 4, 1, 2)  # [B, H, W, A, C]
        classification = classification.contiguous().view(B, H * W * A, C)
        
        return classification

class RegressionHead(nn.Module):
    """📐 Tête de régression pour bounding boxes"""
    
    def __init__(self, config: PredictionConfig):
        super().__init__()
        self.config = config
        
        # Couches cachées
        self.conv_layers = nn.ModuleList()
        
        for i in range(config.num_layers):
            in_ch = config.in_channels if i == 0 else config.hidden_channels
            out_ch = config.hidden_channels
            
            self.conv_layers.append(ConvBlock(in_ch, out_ch, config))
        
        # Couche de sortie (4 coordonnées par anchor)
        self.output_conv = nn.Conv2d(
            config.hidden_channels,
            config.num_anchors * 4,
            3, padding=1, bias=config.use_bias
        )
        
        # Initialisation
        self._initialize_weights()
    
    def _initialize_weights(self):
        """⚖️ Initialisation des poids pour la régression"""
        
        # Couches cachées
        for layer in self.conv_layers:
            if hasattr(layer, 'conv'):
                nn.init.kaiming_normal_(layer.conv.weight, mode='fan_out', nonlinearity='relu')
                if layer.conv.bias is not None:
                    nn.init.constant_(layer.conv.bias, 0)
        
        # Couche de sortie
        nn.init.kaiming_normal_(self.output_conv.weight, mode='fan_out', nonlinearity='relu')
        
        if self.output_conv.bias is not None:
            nn.init.constant_(self.output_conv.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """🔄 Forward pass régression"""
        
        for layer in self.conv_layers:
            x = layer(x)
        
        # Prédiction finale
        regression = self.output_conv(x)
        
        # Reshape: [B, A*4, H, W] → [B, A, 4, H, W] → [B, H, W, A, 4]
        B, _, H, W = regression.shape
        A = self.config.num_anchors
        
        regression = regression.view(B, A, 4, H, W)
        regression = regression.permute(0, 3, 4, 1, 2)  # [B, H, W, A, 4]
        regression = regression.contiguous().view(B, H * W * A, 4)
        
        return regression

class SharedHead(nn.Module):
    """🤝 Tête partagée pour classification et régression"""
    
    def __init__(self, config: PredictionConfig):
        super().__init__()
        self.config = config
        
        # Couches partagées
        self.shared_layers = nn.ModuleList()
        
        for i in range(config.num_layers - 1):  # Garder une couche pour spécialisation
            in_ch = config.in_channels if i == 0 else config.hidden_channels
            out_ch = config.hidden_channels
            
            self.shared_layers.append(ConvBlock(in_ch, out_ch, config))
        
        # Couches spécialisées
        self.cls_layer = ConvBlock(config.hidden_channels, config.hidden_channels, config)
        self.reg_layer = ConvBlock(config.hidden_channels, config.hidden_channels, config)
        
        # Couches de sortie
        self.cls_output = nn.Conv2d(
            config.hidden_channels,
            config.num_anchors * config.num_classes,
            3, padding=1, bias=config.use_bias
        )
        
        self.reg_output = nn.Conv2d(
            config.hidden_channels,
            config.num_anchors * 4,
            3, padding=1, bias=config.use_bias
        )
        
        # Initialisation
        self._initialize_weights()
    
    def _initialize_weights(self):
        """⚖️ Initialisation des poids"""
        
        # Couches partagées et spécialisées
        for layer in list(self.shared_layers) + [self.cls_layer, self.reg_layer]:
            if hasattr(layer, 'conv'):
                nn.init.kaiming_normal_(layer.conv.weight, mode='fan_out', nonlinearity='relu')
                if layer.conv.bias is not None:
                    nn.init.constant_(layer.conv.bias, 0)
        
        # Couche de classification
        nn.init.kaiming_normal_(self.cls_output.weight, mode='fan_out', nonlinearity='relu')
        if self.cls_output.bias is not None:
            bias_init = -math.log((1 - self.config.prior_prob) / self.config.prior_prob)
            nn.init.constant_(self.cls_output.bias, bias_init)
        
        # Couche de régression
        nn.init.kaiming_normal_(self.reg_output.weight, mode='fan_out', nonlinearity='relu')
        if self.reg_output.bias is not None:
            nn.init.constant_(self.reg_output.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """🔄 Forward pass partagé"""
        
        # Couches partagées
        shared_features = x
        for layer in self.shared_layers:
            shared_features = layer(shared_features)
        
        # Spécialisation
        cls_features = self.cls_layer(shared_features)
        reg_features = self.reg_layer(shared_features)
        
        # Sorties
        classification = self.cls_output(cls_features)
        regression = self.reg_output(reg_features)
        
        # Reshape pour correspondre au format attendu
        B, _, H, W = classification.shape
        A = self.config.num_anchors
        C = self.config.num_classes
        
        # Classification: [B, A*C, H, W] → [B, H*W*A, C]
        classification = classification.view(B, A, C, H, W)
        classification = classification.permute(0, 3, 4, 1, 2).contiguous()
        classification = classification.view(B, H * W * A, C)
        
        # Régression: [B, A*4, H, W] → [B, H*W*A, 4]
        regression = regression.view(B, A, 4, H, W)
        regression = regression.permute(0, 3, 4, 1, 2).contiguous()
        regression = regression.view(B, H * W * A, 4)
        
        return classification, regression

class PredictionHead(nn.Module):
    """🎯 Tête de prédiction principale"""
    
    def __init__(self, config: PredictionConfig):
        super().__init__()
        self.config = config
        
        if config.share_heads:
            self.head = SharedHead(config)
        else:
            self.cls_head = ClassificationHead(config)
            self.reg_head = RegressionHead(config)
        
        logger.info(f"🎯 PredictionHead initialisé: {config.num_classes} classes, {config.num_anchors} anchors")
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        🔄 Forward pass de la tête de prédiction
        
        Args:
            features: Features FPN par niveau {'P3': tensor, 'P4': tensor, ...}
            
        Returns:
            Dict avec 'classification' et 'regression' tensors
        """
        
        all_cls_outputs = []
        all_reg_outputs = []
        
        # Traitement de chaque niveau FPN
        for level_name in sorted(features.keys()):
            feature = features[level_name]
            
            if self.config.share_heads:
                cls_output, reg_output = self.head(feature)
            else:
                cls_output = self.cls_head(feature)
                reg_output = self.reg_head(feature)
            
            all_cls_outputs.append(cls_output)
            all_reg_outputs.append(reg_output)
        
        # Concaténation de tous les niveaux
        final_classification = torch.cat(all_cls_outputs, dim=1)
        final_regression = torch.cat(all_reg_outputs, dim=1)
        
        return {
            'classification': final_classification,
            'regression': final_regression
        }
    
    def get_num_parameters(self) -> int:
        """📊 Retourne le nombre de paramètres"""
        return sum(p.numel() for p in self.parameters())

class EfficientHead(nn.Module):
    """⚡ Tête de prédiction optimisée pour vitesse"""
    
    def __init__(self, config: PredictionConfig):
        super().__init__()
        self.config = config
        
        # Version allégée avec moins de couches
        self.shared_conv = ConvBlock(config.in_channels, config.hidden_channels // 2, config)
        
        # Sorties directes
        self.cls_output = nn.Conv2d(
            config.hidden_channels // 2,
            config.num_anchors * config.num_classes,
            1, bias=config.use_bias  # 1x1 conv pour vitesse
        )
        
        self.reg_output = nn.Conv2d(
            config.hidden_channels // 2,
            config.num_anchors * 4,
            1, bias=config.use_bias
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """⚖️ Initialisation des poids"""
        
        # Shared conv
        if hasattr(self.shared_conv, 'conv'):
            nn.init.kaiming_normal_(self.shared_conv.conv.weight, mode='fan_out', nonlinearity='relu')
        
        # Sorties
        nn.init.kaiming_normal_(self.cls_output.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.reg_output.weight, mode='fan_out', nonlinearity='relu')
        
        if self.cls_output.bias is not None:
            bias_init = -math.log((1 - self.config.prior_prob) / self.config.prior_prob)
            nn.init.constant_(self.cls_output.bias, bias_init)
        
        if self.reg_output.bias is not None:
            nn.init.constant_(self.reg_output.bias, 0)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """🔄 Forward pass efficace"""
        
        all_cls_outputs = []
        all_reg_outputs = []
        
        for level_name in sorted(features.keys()):
            feature = features[level_name]
            
            # Features partagées
            shared = self.shared_conv(feature)
            
            # Sorties
            cls_output = self.cls_output(shared)
            reg_output = self.reg_output(shared)
            
            # Reshape
            B, _, H, W = cls_output.shape
            A = self.config.num_anchors
            C = self.config.num_classes
            
            cls_output = cls_output.view(B, A, C, H, W).permute(0, 3, 4, 1, 2).contiguous()
            cls_output = cls_output.view(B, H * W * A, C)
            
            reg_output = reg_output.view(B, A, 4, H, W).permute(0, 3, 4, 1, 2).contiguous()
            reg_output = reg_output.view(B, H * W * A, 4)
            
            all_cls_outputs.append(cls_output)
            all_reg_outputs.append(reg_output)
        
        return {
            'classification': torch.cat(all_cls_outputs, dim=1),
            'regression': torch.cat(all_reg_outputs, dim=1)
        }

# === FONCTIONS UTILITAIRES ===

def create_prediction_head(
    in_channels: int,
    num_classes: int,
    num_anchors: int,
    head_type: str = "standard",
    **kwargs
) -> nn.Module:
    """🏭 Factory pour créer une tête de prédiction"""
    
    config = PredictionConfig(
        in_channels=in_channels,
        num_classes=num_classes,
        num_anchors=num_anchors,
        **kwargs
    )
    
    if head_type == "standard":
        return PredictionHead(config)
    elif head_type == "efficient":
        return EfficientHead(config)
    else:
        raise ValueError(f"Type de tête inconnu: {head_type}")

def get_prediction_config_for_model(model_type: str, num_classes: int) -> PredictionConfig:
    """⚙️ Configuration optimisée par type de modèle"""
    
    configs = {
        "epoch_30": PredictionConfig(
            in_channels=256,
            num_classes=num_classes,
            num_anchors=9,
            hidden_channels=256,
            num_layers=4,
            share_heads=True
        ),
        "fast": PredictionConfig(
            in_channels=128,
            num_classes=num_classes,
            num_anchors=6,
            hidden_channels=128,
            num_layers=2,
            share_heads=True,
            use_separable_conv=True
        ),
        "mobile": PredictionConfig(
            in_channels=96,
            num_classes=num_classes,
            num_anchors=3,
            hidden_channels=96,
            num_layers=2,
            share_heads=True,
            use_separable_conv=True
        )
    }
    
    return configs.get(model_type, configs["epoch_30"])

# === EXPORTS ===
__all__ = [
    "PredictionHead",
    "ClassificationHead", 
    "RegressionHead",
    "SharedHead",
    "EfficientHead",
    "PredictionConfig",
    "create_prediction_head",
    "get_prediction_config_for_model"
]