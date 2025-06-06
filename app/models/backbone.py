"""
🏗️ BACKBONE - ARCHITECTURES DE FEATURE EXTRACTION
=================================================
Implémentation des différents backbones pour l'extraction de features

Backbones supportés:
- MobileNet V2/V3: Optimisé mobile et efficacité
- EfficientNet B0-B7: Scalabilité et performance
- ResNet 18/34/50/101: Robustesse et précision
- Custom backbones: Architectures personnalisées

Caractéristiques:
- Feature extraction multi-échelle
- Support poids pré-entraînés ImageNet
- Gel de couches configurable
- Optimisations pour inférence
- Interface unifiée pour tous les backbones

Architecture:
Input → Conv Stem → Blocks → Multi-level Features
  ↓       ↓         ↓              ↓
Image → Early → {C2, C3, C4, C5} → FPN Input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import logging

try:
    import torchvision.models as tv_models
    from torchvision.models import mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small
    from torchvision.models import resnet18, resnet34, resnet50, resnet101
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logger.warning("⚠️ Torchvision non disponible, certains backbones seront limités")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("⚠️ timm non disponible, EfficientNet sera limité")

logger = logging.getLogger(__name__)

# 🏗️ CLASSE DE BASE
class BackboneBase(nn.Module, ABC):
    """🏗️ Classe de base pour tous les backbones"""
    
    def __init__(self, pretrained: bool = True, freeze_layers: int = 0):
        super().__init__()
        self.pretrained = pretrained
        self.freeze_layers = freeze_layers
        self._out_channels = []
        
    @property
    @abstractmethod
    def out_channels(self) -> List[int]:
        """📏 Canaux de sortie pour chaque niveau"""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """🔄 Forward pass retournant features multi-échelle"""
        pass
    
    def freeze_backbone_layers(self, num_layers: int):
        """❄️ Gèle les premières couches du backbone"""
        if num_layers <= 0:
            return
        
        layer_count = 0
        for name, param in self.named_parameters():
            if layer_count < num_layers:
                param.requires_grad = False
                layer_count += 1
                logger.debug(f"❄️ Couche gelée: {name}")
    
    def get_feature_info(self) -> Dict[str, int]:
        """📋 Informations sur les features extraites"""
        return {
            f"C{i+2}": channels  # C2, C3, C4, C5
            for i, channels in enumerate(self.out_channels)
        }

# 📱 MOBILENET BACKBONE
class MobileNetBackbone(BackboneBase):
    """📱 Backbone MobileNet optimisé mobile"""
    
    def __init__(
        self, 
        version: str = "v3_large",
        pretrained: bool = True,
        freeze_layers: int = 0,
        width_mult: float = 1.0
    ):
        super().__init__(pretrained, freeze_layers)
        
        self.version = version
        self.width_mult = width_mult
        
        # Construction du modèle selon la version
        if version == "v2":
            self._build_mobilenet_v2()
        elif version == "v3_large":
            self._build_mobilenet_v3_large()
        elif version == "v3_small":
            self._build_mobilenet_v3_small()
        else:
            raise ValueError(f"Version MobileNet non supportée: {version}")
        
        # Gel des couches si demandé
        if freeze_layers > 0:
            self.freeze_backbone_layers(freeze_layers)
        
        logger.info(f"📱 MobileNet {version} initialisé (pretrained={pretrained})")
    
    def _build_mobilenet_v2(self):
        """🏗️ Construction MobileNet V2"""
        if not TORCHVISION_AVAILABLE:
            raise ImportError("Torchvision requis pour MobileNet V2")
        
        # Chargement modèle pré-entraîné
        base_model = mobilenet_v2(pretrained=self.pretrained)
        
        # Extraction des features à différents niveaux
        self.stem = nn.Sequential(
            base_model.features[0],  # Conv2d + BN + ReLU6
            base_model.features[1],  # InvertedResidual
        )
        
        # Blocs principaux
        self.layer1 = nn.Sequential(*base_model.features[2:4])   # C2: 24 channels
        self.layer2 = nn.Sequential(*base_model.features[4:7])   # C3: 32 channels  
        self.layer3 = nn.Sequential(*base_model.features[7:14])  # C4: 96 channels
        self.layer4 = nn.Sequential(*base_model.features[14:])   # C5: 320 channels
        
        self._out_channels = [24, 32, 96, 320]
    
    def _build_mobilenet_v3_large(self):
        """🏗️ Construction MobileNet V3 Large"""
        if not TORCHVISION_AVAILABLE:
            raise ImportError("Torchvision requis pour MobileNet V3")
        
        base_model = mobilenet_v3_large(pretrained=self.pretrained)
        
        # Stem
        self.stem = base_model.features[0]  # Conv2d + BN + Hardswish
        
        # Extraction des features
        features = base_model.features[1:]
        
        # Regroupement par résolution
        self.layer1 = nn.Sequential(*features[0:3])   # C2: 24 channels
        self.layer2 = nn.Sequential(*features[3:6])   # C3: 40 channels
        self.layer3 = nn.Sequential(*features[6:12])  # C4: 112 channels  
        self.layer4 = nn.Sequential(*features[12:])   # C5: 960 channels
        
        self._out_channels = [24, 40, 112, 960]
    
    def _build_mobilenet_v3_small(self):
        """🏗️ Construction MobileNet V3 Small"""
        if not TORCHVISION_AVAILABLE:
            raise ImportError("Torchvision requis pour MobileNet V3")
        
        base_model = mobilenet_v3_small(pretrained=self.pretrained)
        
        self.stem = base_model.features[0]
        features = base_model.features[1:]
        
        self.layer1 = nn.Sequential(*features[0:2])   # C2: 16 channels
        self.layer2 = nn.Sequential(*features[2:5])   # C3: 24 channels
        self.layer3 = nn.Sequential(*features[5:9])   # C4: 48 channels
        self.layer4 = nn.Sequential(*features[9:])    # C5: 576 channels
        
        self._out_channels = [16, 24, 48, 576]
    
    @property
    def out_channels(self) -> List[int]:
        return self._out_channels
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """🔄 Forward pass MobileNet"""
        features = {}
        
        # Stem
        x = self.stem(x)
        
        # Extraction multi-échelle
        x = self.layer1(x)
        features['C2'] = x
        
        x = self.layer2(x)
        features['C3'] = x
        
        x = self.layer3(x)
        features['C4'] = x
        
        x = self.layer4(x)
        features['C5'] = x
        
        return features

# 🚀 EFFICIENTNET BACKBONE
class EfficientNetBackbone(BackboneBase):
    """🚀 Backbone EfficientNet haute performance"""
    
    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        pretrained: bool = True,
        freeze_layers: int = 0
    ):
        super().__init__(pretrained, freeze_layers)
        
        self.model_name = model_name
        
        if not TIMM_AVAILABLE:
            logger.warning("⚠️ timm non disponible, fallback vers implémentation basique")
            self._build_basic_efficientnet()
        else:
            self._build_timm_efficientnet()
        
        if freeze_layers > 0:
            self.freeze_backbone_layers(freeze_layers)
        
        logger.info(f"🚀 EfficientNet {model_name} initialisé")
    
    def _build_timm_efficientnet(self):
        """🏗️ Construction via timm (recommandé)"""
        
        # Chargement modèle timm
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            features_only=True,
            out_indices=[1, 2, 3, 4]  # Indices des features C2, C3, C4, C5
        )
        
        # Récupération des canaux de sortie
        feature_info = self.backbone.feature_info
        self._out_channels = [info['num_chs'] for info in feature_info]
    
    def _build_basic_efficientnet(self):
        """🏗️ Implémentation basique EfficientNet"""
        logger.warning("⚠️ Utilisation implémentation EfficientNet basique")
        
        # Implémentation simplifiée pour B0
        from .efficientnet_basic import EfficientNetBasic
        self.backbone = EfficientNetBasic(
            model_name=self.model_name,
            pretrained=self.pretrained
        )
        
        # Canaux par défaut pour B0
        self._out_channels = [24, 40, 112, 320]
    
    @property
    def out_channels(self) -> List[int]:
        return self._out_channels
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """🔄 Forward pass EfficientNet"""
        
        if hasattr(self.backbone, 'forward_features'):
            # Interface timm
            feature_list = self.backbone.forward_features(x)
            
            features = {
                'C2': feature_list[0],
                'C3': feature_list[1], 
                'C4': feature_list[2],
                'C5': feature_list[3]
            }
        else:
            # Interface basique
            features = self.backbone(x)
        
        return features

# 🏛️ RESNET BACKBONE
class ResNetBackbone(BackboneBase):
    """🏛️ Backbone ResNet robuste et éprouvé"""
    
    def __init__(
        self,
        depth: int = 50,
        pretrained: bool = True,
        freeze_layers: int = 0,
        replace_stride_with_dilation: Optional[List[bool]] = None
    ):
        super().__init__(pretrained, freeze_layers)
        
        self.depth = depth
        
        # Construction selon la profondeur
        if depth == 18:
            self._build_resnet18()
        elif depth == 34:
            self._build_resnet34()
        elif depth == 50:
            self._build_resnet50()
        elif depth == 101:
            self._build_resnet101()
        else:
            raise ValueError(f"Profondeur ResNet non supportée: {depth}")
        
        if freeze_layers > 0:
            self.freeze_backbone_layers(freeze_layers)
        
        logger.info(f"🏛️ ResNet-{depth} initialisé")
    
    def _build_resnet18(self):
        """🏗️ Construction ResNet-18"""
        if not TORCHVISION_AVAILABLE:
            raise ImportError("Torchvision requis pour ResNet")
        
        base_model = resnet18(pretrained=self.pretrained)
        self._extract_resnet_layers(base_model)
        self._out_channels = [64, 128, 256, 512]
    
    def _build_resnet34(self):
        """🏗️ Construction ResNet-34"""
        if not TORCHVISION_AVAILABLE:
            raise ImportError("Torchvision requis pour ResNet")
        
        base_model = resnet34(pretrained=self.pretrained)
        self._extract_resnet_layers(base_model)
        self._out_channels = [64, 128, 256, 512]
    
    def _build_resnet50(self):
        """🏗️ Construction ResNet-50"""
        if not TORCHVISION_AVAILABLE:
            raise ImportError("Torchvision requis pour ResNet")
        
        base_model = resnet50(pretrained=self.pretrained)
        self._extract_resnet_layers(base_model)
        self._out_channels = [256, 512, 1024, 2048]
    
    def _build_resnet101(self):
        """🏗️ Construction ResNet-101"""
        if not TORCHVISION_AVAILABLE:
            raise ImportError("Torchvision requis pour ResNet")
        
        base_model = resnet101(pretrained=self.pretrained)
        self._extract_resnet_layers(base_model)
        self._out_channels = [256, 512, 1024, 2048]
    
    def _extract_resnet_layers(self, base_model):
        """🔧 Extraction des couches ResNet"""
        
        # Stem (conv1 + bn1 + relu + maxpool)
        self.stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )
        
        # Blocs résiduels
        self.layer1 = base_model.layer1  # C2
        self.layer2 = base_model.layer2  # C3
        self.layer3 = base_model.layer3  # C4
        self.layer4 = base_model.layer4  # C5
    
    @property
    def out_channels(self) -> List[int]:
        return self._out_channels
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """🔄 Forward pass ResNet"""
        features = {}
        
        # Stem
        x = self.stem(x)
        
        # Blocs résiduels
        x = self.layer1(x)
        features['C2'] = x
        
        x = self.layer2(x)
        features['C3'] = x
        
        x = self.layer3(x)
        features['C4'] = x
        
        x = self.layer4(x)
        features['C5'] = x
        
        return features

# 🎨 BACKBONE PERSONNALISÉ
class CustomBackbone(BackboneBase):
    """🎨 Backbone personnalisé configurable"""
    
    def __init__(
        self,
        channels: List[int] = [32, 64, 128, 256],
        pretrained: bool = False,
        freeze_layers: int = 0
    ):
        super().__init__(pretrained, freeze_layers)
        
        self._out_channels = channels
        
        # Construction architecture simple
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0]//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0]//2, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Blocs de downsampling
        self.layers = nn.ModuleList()
        
        for i in range(len(channels)):
            in_channels = channels[i-1] if i > 0 else channels[0]
            out_channels = channels[i]
            
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
            self.layers.append(layer)
        
        logger.info(f"🎨 Backbone personnalisé initialisé: {channels}")
    
    @property
    def out_channels(self) -> List[int]:
        return self._out_channels
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """🔄 Forward pass backbone personnalisé"""
        features = {}
        
        x = self.stem(x)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            features[f'C{i+2}'] = x
        
        return features

# 🏭 FACTORY FUNCTION
def get_backbone(
    name: str,
    pretrained: bool = True,
    freeze_layers: int = 0,
    **kwargs
) -> BackboneBase:
    """
    🏭 Factory pour créer un backbone
    
    Args:
        name: Nom du backbone
        pretrained: Utiliser poids pré-entraînés
        freeze_layers: Nombre de couches à geler
        **kwargs: Arguments additionnels
        
    Returns:
        Instance de backbone
    """
    
    name = name.lower()
    
    # MobileNet variants
    if "mobilenet" in name:
        if "v2" in name:
            return MobileNetBackbone("v2", pretrained, freeze_layers, **kwargs)
        elif "v3_large" in name or "v3-large" in name:
            return MobileNetBackbone("v3_large", pretrained, freeze_layers, **kwargs)
        elif "v3_small" in name or "v3-small" in name:
            return MobileNetBackbone("v3_small", pretrained, freeze_layers, **kwargs)
        else:
            return MobileNetBackbone("v3_large", pretrained, freeze_layers, **kwargs)
    
    # EfficientNet variants
    elif "efficientnet" in name:
        return EfficientNetBackbone(name, pretrained, freeze_layers, **kwargs)
    
    # ResNet variants
    elif "resnet" in name:
        if "resnet18" in name:
            return ResNetBackbone(18, pretrained, freeze_layers, **kwargs)
        elif "resnet34" in name:
            return ResNetBackbone(34, pretrained, freeze_layers, **kwargs)
        elif "resnet50" in name:
            return ResNetBackbone(50, pretrained, freeze_layers, **kwargs)
        elif "resnet101" in name:
            return ResNetBackbone(101, pretrained, freeze_layers, **kwargs)
        else:
            return ResNetBackbone(50, pretrained, freeze_layers, **kwargs)
    
    # Custom backbone
    elif "custom" in name:
        return CustomBackbone(pretrained=pretrained, freeze_layers=freeze_layers, **kwargs)
    
    else:
        logger.warning(f"⚠️ Backbone inconnu: {name}, fallback vers MobileNet V3")
        return MobileNetBackbone("v3_large", pretrained, freeze_layers, **kwargs)

# 📋 UTILITAIRES
def list_available_backbones() -> List[str]:
    """📋 Liste les backbones disponibles"""
    backbones = [
        "mobilenet_v2",
        "mobilenet_v3_large", 
        "mobilenet_v3_small",
        "resnet18",
        "resnet34", 
        "resnet50",
        "resnet101",
        "custom"
    ]
    
    if TIMM_AVAILABLE:
        efficientnets = [f"efficientnet-b{i}" for i in range(8)]
        backbones.extend(efficientnets)
    
    return sorted(backbones)

def get_backbone_info(name: str) -> Dict[str, any]:
    """📋 Informations sur un backbone"""
    
    backbone = get_backbone(name, pretrained=False)
    
    return {
        "name": name,
        "out_channels": backbone.out_channels,
        "num_levels": len(backbone.out_channels),
        "feature_info": backbone.get_feature_info()
    }

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "BackboneBase",
    "MobileNetBackbone",
    "EfficientNetBackbone", 
    "ResNetBackbone",
    "CustomBackbone",
    "get_backbone",
    "list_available_backbones",
    "get_backbone_info"
]