"""
🏗️ FPN - FEATURE PYRAMID NETWORK
================================
Implémentation du Feature Pyramid Network pour fusion multi-échelle

Le FPN permet de combiner les features de différentes résolutions du backbone
pour améliorer la détection d'objets de toutes tailles.

Types de FPN:
- FeaturePyramidNetwork: FPN standard (top-down + lateral connections)
- SimpleFPN: Version simplifiée pour modèles légers
- BiFPN: Bidirectional FPN (EfficientDet style)
- PANet: Path Aggregation Network

Architecture FPN standard:
     
    Backbone Features    Top-down Pathway    Final Features
    ┌─────────────┐     ┌─────────────┐    ┌─────────────┐
    │    C5       │────▶│     P5      │───▶│     F5      │
    │  (2048ch)   │     │  (256ch)    │    │  (256ch)    │ 
    └─────────────┘     └─────────────┘    └─────────────┘
           │                     │                  │
    ┌─────────────┐     ┌─────────────┐    ┌─────────────┐
    │    C4       │────▶│     P4      │───▶│     F4      │
    │  (1024ch)   │     │  (256ch)    │    │  (256ch)    │
    └─────────────┘     └─────────────┘    └─────────────┘
           │                     │                  │
    ┌─────────────┐     ┌─────────────┐    ┌─────────────┐
    │    C3       │────▶│     P3      │───▶│     F3      │
    │  (512ch)    │     │  (256ch)    │    │  (256ch)    │
    └─────────────┘     └─────────────┘    └─────────────┘
           │                     │                  │
    ┌─────────────┐     ┌─────────────┐    ┌─────────────┐
    │    C2       │────▶│     P2      │───▶│     F2      │
    │  (256ch)    │     │  (256ch)    │    │  (256ch)    │
    └─────────────┘     └─────────────┘    └─────────────┘

Features uniformes pour prédiction multi-échelle !
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# 📋 CONFIGURATION FPN
@dataclass
class FPNConfig:
    """⚙️ Configuration Feature Pyramid Network"""
    in_channels: List[int] = None  # Canaux d'entrée du backbone
    out_channels: int = 256        # Canaux de sortie uniformes
    num_levels: int = 5           # Nombre de niveaux FPN
    use_bias: bool = True         # Bias dans les convolutions
    use_batch_norm: bool = False  # Batch normalization
    activation: str = "relu"      # Fonction d'activation
    
    # Optimisations
    use_separable_conv: bool = False  # Convolutions séparables
    use_group_conv: bool = False      # Convolutions de groupe
    
    # Extra levels
    add_extra_levels: bool = True     # Ajouter P6, P7
    extra_levels_on_inputs: bool = False  # Extra levels depuis inputs
    
    def __post_init__(self):
        """Validation de la configuration"""
        if self.in_channels is None:
            # Valeurs par défaut ResNet-50
            self.in_channels = [256, 512, 1024, 2048]
        
        if len(self.in_channels) > self.num_levels:
            logger.warning(f"⚠️ Plus de canaux d'entrée ({len(self.in_channels)}) que de niveaux FPN ({self.num_levels})")

# 🏗️ FPN STANDARD
class FeaturePyramidNetwork(nn.Module):
    """🏗️ Feature Pyramid Network standard"""
    
    def __init__(self, config: FPNConfig = None):
        super().__init__()
        
        self.config = config or FPNConfig()
        
        # Convolutions latérales (réduction canaux)
        self.lateral_convs = nn.ModuleList()
        
        # Convolutions top-down (uniformisation)
        self.fpn_convs = nn.ModuleList()
        
        # Construction des couches
        self._build_layers()
        
        # Niveaux extra si demandés
        if self.config.add_extra_levels:
            self._build_extra_levels()
        
        # Initialisation des poids
        self._initialize_weights()
        
        logger.info(f"🏗️ FPN initialisé: {len(self.config.in_channels)} → {self.config.out_channels} canaux")
    
    def _build_layers(self):
        """🔧 Construction des couches FPN"""
        
        for i, in_ch in enumerate(self.config.in_channels):
            # Convolution latérale (1x1)
            lateral_conv = self._make_conv1x1(in_ch, self.config.out_channels)
            self.lateral_convs.append(lateral_conv)
            
            # Convolution FPN (3x3)
            fpn_conv = self._make_conv3x3(self.config.out_channels, self.config.out_channels)
            self.fpn_convs.append(fpn_conv)
    
    def _build_extra_levels(self):
        """🔧 Construction des niveaux supplémentaires"""
        
        # P6 et P7 pour détecter très gros objets
        if self.config.extra_levels_on_inputs:
            # P6 depuis C5
            self.p6_conv = self._make_conv3x3(
                self.config.in_channels[-1], 
                self.config.out_channels,
                stride=2
            )
        else:
            # P6 depuis P5
            self.p6_conv = self._make_conv3x3(
                self.config.out_channels,
                self.config.out_channels, 
                stride=2
            )
        
        # P7 depuis P6
        self.p7_conv = self._make_conv3x3(
            self.config.out_channels,
            self.config.out_channels,
            stride=2
        )
    
    def _make_conv1x1(self, in_channels: int, out_channels: int) -> nn.Module:
        """🔧 Création convolution 1x1"""
        
        layers = []
        
        # Convolution principale
        conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1,
            bias=self.config.use_bias
        )
        layers.append(conv)
        
        # Normalisation
        if self.config.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*layers)
    
    def _make_conv3x3(
        self, 
        in_channels: int, 
        out_channels: int,
        stride: int = 1
    ) -> nn.Module:
        """🔧 Création convolution 3x3"""
        
        layers = []
        
        # Type de convolution
        if self.config.use_separable_conv:
            # Convolution séparable (depthwise + pointwise)
            layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, 
                         groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, 1, bias=self.config.use_bias)
            ])
        elif self.config.use_group_conv:
            # Convolution de groupe
            groups = min(in_channels, out_channels, 32)
            conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, 
                           padding=1, groups=groups, bias=self.config.use_bias)
            layers.append(conv)
        else:
            # Convolution standard
            conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, 
                           padding=1, bias=self.config.use_bias)
            layers.append(conv)
        
        # Normalisation
        if self.config.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Activation
        if self.config.activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif self.config.activation == "swish":
            layers.append(nn.SiLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """⚖️ Initialisation des poids"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """🔄 Forward pass FPN"""
        
        # Récupération features backbone (C2, C3, C4, C5)
        backbone_features = [features[f'C{i+2}'] for i in range(len(self.config.in_channels))]
        
        # 1. Convolutions latérales
        lateral_features = []
        for i, (feat, lateral_conv) in enumerate(zip(backbone_features, self.lateral_convs)):
            lateral = lateral_conv(feat)
            lateral_features.append(lateral)
        
        # 2. Top-down pathway
        fpn_features = []
        
        # Commencer par le niveau le plus haut (C5)
        prev_feat = lateral_features[-1]
        fpn_feat = self.fpn_convs[-1](prev_feat)
        fpn_features.append(fpn_feat)
        
        # Descendre les niveaux
        for i in range(len(lateral_features) - 2, -1, -1):
            # Upsampling du niveau précédent
            upsampled = F.interpolate(
                prev_feat, 
                size=lateral_features[i].shape[-2:],
                mode='nearest'
            )
            
            # Addition avec feature latérale
            fused = lateral_features[i] + upsampled
            
            # Convolution finale
            fpn_feat = self.fpn_convs[i](fused)
            fpn_features.append(fpn_feat)
            
            prev_feat = fused
        
        # Inverser pour avoir P2, P3, P4, P5
        fpn_features = fpn_features[::-1]
        
        # 3. Construction dictionnaire de sortie
        output_features = {}
        for i, feat in enumerate(fpn_features):
            output_features[f'P{i+2}'] = feat
        
        # 4. Niveaux extra
        if self.config.add_extra_levels:
            if self.config.extra_levels_on_inputs:
                # P6 depuis C5
                p6 = self.p6_conv(backbone_features[-1])
            else:
                # P6 depuis P5
                p6 = self.p6_conv(fpn_features[-1])
            
            output_features['P6'] = p6
            
            # P7 depuis P6 avec ReLU
            p7 = self.p7_conv(F.relu(p6))
            output_features['P7'] = p7
        
        return output_features

# 🔧 FPN SIMPLIFIÉ
class SimpleFPN(nn.Module):
    """🔧 FPN simplifié pour modèles légers"""
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        simplified: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.simplified = simplified
        
        # Convolutions de réduction uniquement
        self.reduce_convs = nn.ModuleList()
        
        for in_ch in in_channels:
            conv = nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.reduce_convs.append(conv)
        
        logger.info(f"🔧 SimpleFPN initialisé: {in_channels} → {out_channels}")
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """🔄 Forward pass FPN simplifié"""
        
        backbone_features = [features[f'C{i+2}'] for i in range(len(self.in_channels))]
        
        output_features = {}
        
        if self.simplified:
            # Mode ultra-simplifié: juste réduction de canaux
            for i, (feat, conv) in enumerate(zip(backbone_features, self.reduce_convs)):
                reduced = conv(feat)
                output_features[f'P{i+2}'] = reduced
        else:
            # Mode avec fusion minimale
            reduced_features = []
            for feat, conv in zip(backbone_features, self.reduce_convs):
                reduced = conv(feat)
                reduced_features.append(reduced)
            
            # Fusion simple top-down
            prev_feat = reduced_features[-1]
            output_features[f'P{len(reduced_features)+1}'] = prev_feat
            
            for i in range(len(reduced_features) - 2, -1, -1):
                # Upsampling simple
                upsampled = F.interpolate(
                    prev_feat,
                    size=reduced_features[i].shape[-2:],
                    mode='nearest'
                )
                
                # Addition
                fused = reduced_features[i] + upsampled
                output_features[f'P{i+2}'] = fused
                prev_feat = fused
        
        return output_features

# 🔄 BIFPN (EfficientDet style)
class BiFPN(nn.Module):
    """🔄 Bidirectional Feature Pyramid Network"""
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_repeats: int = 1,
        attention: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_repeats = num_repeats
        self.attention = attention
        
        # Convolutions de réduction
        self.reduce_convs = nn.ModuleList()
        for in_ch in in_channels:
            conv = nn.Conv2d(in_ch, out_channels, 1, bias=False)
            self.reduce_convs.append(conv)
        
        # Blocs BiFPN
        self.bifpn_blocks = nn.ModuleList()
        for _ in range(num_repeats):
            block = BiFPNBlock(out_channels, attention)
            self.bifpn_blocks.append(block)
        
        logger.info(f"🔄 BiFPN initialisé: {num_repeats} blocs, attention={attention}")
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """🔄 Forward pass BiFPN"""
        
        backbone_features = [features[f'C{i+2}'] for i in range(len(self.in_channels))]
        
        # Réduction de canaux
        fpn_features = []
        for feat, conv in zip(backbone_features, self.reduce_convs):
            reduced = conv(feat)
            fpn_features.append(reduced)
        
        # Passage dans les blocs BiFPN
        for bifpn_block in self.bifpn_blocks:
            fpn_features = bifpn_block(fpn_features)
        
        # Construction dictionnaire de sortie
        output_features = {}
        for i, feat in enumerate(fpn_features):
            output_features[f'P{i+2}'] = feat
        
        return output_features

class BiFPNBlock(nn.Module):
    """🔄 Bloc BiFPN individuel"""
    
    def __init__(self, channels: int, attention: bool = True):
        super().__init__()
        
        self.channels = channels
        self.attention = attention
        
        # Convolutions pour fusion
        self.conv_up = nn.ModuleList()
        self.conv_down = nn.ModuleList()
        
        # 4 niveaux → 3 connections up + 3 connections down
        for _ in range(3):
            conv_up = self._make_fusion_conv(channels)
            conv_down = self._make_fusion_conv(channels)
            self.conv_up.append(conv_up)
            self.conv_down.append(conv_down)
        
        # Poids d'attention si activé
        if attention:
            self.attention_weights_up = nn.ParameterList([
                nn.Parameter(torch.ones(2)) for _ in range(3)
            ])
            self.attention_weights_down = nn.ParameterList([
                nn.Parameter(torch.ones(2)) for _ in range(3)
            ])
    
    def _make_fusion_conv(self, channels: int) -> nn.Module:
        """🔧 Création convolution de fusion"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """🔄 Forward pass bloc BiFPN"""
        
        # Top-down pathway
        up_features = [features[-1]]  # Commencer par P5
        
        for i in range(len(features) - 2, -1, -1):
            # Upsampling
            upsampled = F.interpolate(
                up_features[-1],
                size=features[i].shape[-2:],
                mode='nearest'
            )
            
            # Fusion avec attention
            if self.attention:
                weights = F.softmax(self.attention_weights_up[len(features)-2-i], dim=0)
                fused = weights[0] * features[i] + weights[1] * upsampled
            else:
                fused = features[i] + upsampled
            
            # Convolution
            conv_idx = len(features) - 2 - i
            fused = self.conv_up[conv_idx](fused)
            up_features.append(fused)
        
        up_features = up_features[::-1]  # P2, P3, P4, P5
        
        # Bottom-up pathway
        out_features = [up_features[0]]  # P2
        
        for i in range(1, len(up_features)):
            # Downsampling
            downsampled = F.interpolate(
                out_features[-1],
                size=up_features[i].shape[-2:],
                mode='nearest'
            )
            
            # Fusion avec attention
            if self.attention:
                weights = F.softmax(self.attention_weights_down[i-1], dim=0)
                fused = weights[0] * up_features[i] + weights[1] * downsampled
            else:
                fused = up_features[i] + downsampled
            
            # Convolution
            fused = self.conv_down[i-1](fused)
            out_features.append(fused)
        
        return out_features

# 🏭 FONCTIONS FACTORY
def create_fpn(
    fpn_type: str = "standard",
    in_channels: List[int] = None,
    out_channels: int = 256,
    **kwargs
) -> nn.Module:
    """
    🏭 Factory pour créer un FPN
    
    Args:
        fpn_type: Type de FPN ("standard", "simple", "bifpn")
        in_channels: Canaux d'entrée du backbone
        out_channels: Canaux de sortie
        **kwargs: Arguments additionnels
        
    Returns:
        Instance de FPN
    """
    
    in_channels = in_channels or [256, 512, 1024, 2048]
    
    if fpn_type == "standard":
        config = FPNConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        )
        return FeaturePyramidNetwork(config)
    
    elif fpn_type == "simple":
        return SimpleFPN(in_channels, out_channels, **kwargs)
    
    elif fpn_type == "bifpn":
        return BiFPN(in_channels, out_channels, **kwargs)
    
    else:
        logger.warning(f"⚠️ Type FPN inconnu: {fpn_type}, fallback vers standard")
        config = FPNConfig(in_channels=in_channels, out_channels=out_channels)
        return FeaturePyramidNetwork(config)

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "FeaturePyramidNetwork",
    "SimpleFPN", 
    "BiFPN",
    "BiFPNBlock",
    "FPNConfig",
    "create_fpn"
]