"""
🤖 MODELS PACKAGE - ARCHITECTURE DES MODÈLES PYTORCH
===================================================
Package contenant l'architecture complète des modèles de détection d'objets perdus

Ce package implémente:
- Architecture principale du modèle de détection
- Backbones optimisés (MobileNet, EfficientNet, ResNet)
- Feature Pyramid Network (FPN) pour multi-échelle
- Têtes de prédiction (classification + localisation)
- Système d'anchors pour la détection
- Optimisations pour différents cas d'usage

Architecture générale:
Input → Backbone → FPN → Prediction Head → Output
  ↓        ↓        ↓         ↓           ↓
Image → Features → Multi-scale → Detections → Post-process

Modèles supportés:
- Epoch 30: Champion (F1=49.86%, Précision=60.73%)
- Extended: 28 classes d'objets perdus  
- Fast: Optimisé streaming temps réel
- Mobile: Optimisé edge/mobile deployment

Utilisation:
- Entraînement de nouveaux modèles
- Inférence optimisée
- Transfer learning
- Déploiement production
"""

from .model import (
    LostObjectDetectionModel,
    ModelConfig,
    ModelArchitecture,
    create_model
)

from .backbone import (
    BackboneBase,
    MobileNetBackbone,
    EfficientNetBackbone,
    ResNetBackbone,
    get_backbone
)

from .fpn import (
    FeaturePyramidNetwork,
    FPNConfig,
    SimpleFPN,
    BiFPN
)

from .prediction import (
    PredictionHead,
    ClassificationHead,
    RegressionHead,
    PredictionConfig
)

from .anchors import (
    AnchorGenerator,
    AnchorConfig,
    generate_anchors,
    anchor_utils
)

__all__ = [
    # Modèle principal
    "LostObjectDetectionModel",
    "ModelConfig", 
    "ModelArchitecture",
    "create_model",
    
    # Backbones
    "BackboneBase",
    "MobileNetBackbone",
    "EfficientNetBackbone", 
    "ResNetBackbone",
    "get_backbone",
    
    # Feature Pyramid Network
    "FeaturePyramidNetwork",
    "FPNConfig",
    "SimpleFPN",
    "BiFPN",
    
    # Prediction heads
    "PredictionHead",
    "ClassificationHead",
    "RegressionHead", 
    "PredictionConfig",
    
    # Anchors
    "AnchorGenerator",
    "AnchorConfig",
    "generate_anchors",
    "anchor_utils"
]