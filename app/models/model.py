"""
🤖 MODEL - ARCHITECTURE PRINCIPALE DE DÉTECTION D'OBJETS PERDUS
==============================================================
Architecture complète du modèle de détection utilisant PyTorch

Composants:
- LostObjectDetectionModel: Modèle principal intégrant tous les composants
- ModelConfig: Configuration flexible pour différents cas d'usage
- Factory functions pour création automatique de modèles
- Support de différentes architectures (MobileNet, EfficientNet, ResNet)
- Optimisations pour déploiement (quantization, TorchScript, ONNX)

Architecture:
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────────┐
│  Input  │ -> │Backbone │ -> │   FPN   │ -> │ Prediction  │
│ Image   │    │Feature  │    │Multi-   │    │    Head     │
│         │    │Extract  │    │Scale    │    │ Cls + Reg   │
└─────────┘    └─────────┘    └─────────┘    └─────────────┘

Modèles pré-configurés:
- epoch_30: Champion avec MobileNet + FPN
- extended: EfficientNet pour 28 classes
- fast: Ultra-rapide pour streaming
- mobile: Optimisé edge/mobile
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Imports internes
from .backbone import BackboneBase, get_backbone
from .fpn import FeaturePyramidNetwork, FPNConfig
from .prediction import PredictionHead, PredictionConfig
from .anchors import AnchorGenerator, AnchorConfig

logger = logging.getLogger(__name__)

# 📋 ÉNUMÉRATIONS ET TYPES
class ModelArchitecture(str, Enum):
    """🏗️ Architectures de modèles supportées"""
    MOBILENET_SSD = "mobilenet_ssd"
    EFFICIENTDET = "efficientdet"
    YOLO_V5 = "yolo_v5"
    RETINANET = "retinanet"
    FCOS = "fcos"
    CUSTOM = "custom"

class ModelSize(str, Enum):
    """📏 Tailles de modèles"""
    NANO = "nano"           # Ultra-compact
    SMALL = "small"         # Compact
    MEDIUM = "medium"       # Standard
    LARGE = "large"         # Haute performance
    XLARGE = "xlarge"       # Maximum performance

@dataclass
class ModelConfig:
    """⚙️ Configuration complète du modèle"""
    # Architecture de base
    architecture: ModelArchitecture = ModelArchitecture.MOBILENET_SSD
    size: ModelSize = ModelSize.MEDIUM
    
    # Classes et détection
    num_classes: int = 28
    input_size: Tuple[int, int] = (640, 640)
    num_anchors_per_level: int = 9
    
    # Backbone
    backbone_name: str = "mobilenet_v3_large"
    backbone_pretrained: bool = True
    backbone_freeze_layers: int = 0
    
    # FPN
    fpn_channels: int = 256
    fpn_levels: int = 5
    fpn_use_bias: bool = True
    
    # Prediction head
    head_channels: int = 256
    head_num_layers: int = 4
    head_use_bias: bool = True
    head_activation: str = "relu"
    
    # Anchors
    anchor_sizes: List[float] = None
    anchor_ratios: List[float] = None
    anchor_scales: List[float] = None
    
    # Training
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    use_group_norm: bool = False
    
    # Optimisations
    use_separable_conv: bool = False
    use_depthwise_conv: bool = False
    activation_checkpoint: bool = False
    
    def __post_init__(self):
        """Initialisation post-création"""
        if self.anchor_sizes is None:
            self.anchor_sizes = [32, 64, 128, 256, 512]
        if self.anchor_ratios is None:
            self.anchor_ratios = [0.5, 1.0, 2.0]
        if self.anchor_scales is None:
            self.anchor_scales = [1.0, 1.26, 1.587]
    
    @classmethod
    def get_epoch30_config(cls) -> 'ModelConfig':
        """🏆 Configuration du modèle champion Epoch 30"""
        return cls(
            architecture=ModelArchitecture.MOBILENET_SSD,
            size=ModelSize.MEDIUM,
            num_classes=28,
            input_size=(640, 640),
            backbone_name="mobilenet_v3_large",
            backbone_pretrained=True,
            fpn_channels=256,
            head_channels=256,
            head_num_layers=4,
            dropout_rate=0.1
        )
    
    @classmethod
    def get_extended_config(cls) -> 'ModelConfig':
        """🔧 Configuration du modèle étendu"""
        return cls(
            architecture=ModelArchitecture.EFFICIENTDET,
            size=ModelSize.LARGE,
            num_classes=28,
            input_size=(512, 512),
            backbone_name="efficientnet-b2",
            backbone_pretrained=True,
            fpn_channels=384,
            head_channels=384,
            head_num_layers=4,
            dropout_rate=0.2
        )
    
    @classmethod
    def get_fast_config(cls) -> 'ModelConfig':
        """⚡ Configuration du modèle rapide"""
        return cls(
            architecture=ModelArchitecture.YOLO_V5,
            size=ModelSize.SMALL,
            num_classes=28,
            input_size=(416, 416),
            backbone_name="mobilenet_v3_small",
            backbone_pretrained=True,
            fpn_channels=128,
            head_channels=128,
            head_num_layers=2,
            dropout_rate=0.05,
            use_separable_conv=True
        )
    
    @classmethod
    def get_mobile_config(cls) -> 'ModelConfig':
        """📱 Configuration du modèle mobile"""
        return cls(
            architecture=ModelArchitecture.MOBILENET_SSD,
            size=ModelSize.NANO,
            num_classes=28,
            input_size=(320, 320),
            backbone_name="mobilenet_v2",
            backbone_pretrained=True,
            fpn_channels=96,
            head_channels=96,
            head_num_layers=2,
            dropout_rate=0.1,
            use_separable_conv=True,
            use_depthwise_conv=True
        )

# 🤖 MODÈLE PRINCIPAL
class LostObjectDetectionModel(nn.Module):
    """🤖 Modèle principal de détection d'objets perdus"""
    
    def __init__(
        self,
        config: ModelConfig = None,
        backbone: BackboneBase = None,
        fpn: FeaturePyramidNetwork = None,
        prediction_head: PredictionHead = None
    ):
        """
        Initialise le modèle de détection
        
        Args:
            config: Configuration du modèle
            backbone: Backbone personnalisé (optionnel)
            fpn: FPN personnalisé (optionnel)  
            prediction_head: Tête de prédiction personnalisée (optionnel)
        """
        super().__init__()
        
        self.config = config or ModelConfig()
        
        # Construction des composants
        self.backbone = backbone or self._build_backbone()
        self.fpn = fpn or self._build_fpn()
        self.prediction_head = prediction_head or self._build_prediction_head()
        
        # Générateur d'anchors
        self.anchor_generator = self._build_anchor_generator()
        
        # Initialisation des poids
        self._initialize_weights()
        
        logger.info(f"🤖 Modèle initialisé: {self.config.architecture.value}")
    
    def _build_backbone(self) -> BackboneBase:
        """🏗️ Construit le backbone"""
        return get_backbone(
            name=self.config.backbone_name,
            pretrained=self.config.backbone_pretrained,
            freeze_layers=self.config.backbone_freeze_layers
        )
    
    def _build_fpn(self) -> FeaturePyramidNetwork:
        """🏗️ Construit le FPN"""
        fpn_config = FPNConfig(
            in_channels=self.backbone.out_channels,
            out_channels=self.config.fpn_channels,
            num_levels=self.config.fpn_levels,
            use_bias=self.config.fpn_use_bias
        )
        return FeaturePyramidNetwork(fpn_config)
    
    def _build_prediction_head(self) -> PredictionHead:
        """🏗️ Construit la tête de prédiction"""
        pred_config = PredictionConfig(
            in_channels=self.config.fpn_channels,
            num_classes=self.config.num_classes,
            num_anchors=self.config.num_anchors_per_level,
            hidden_channels=self.config.head_channels,
            num_layers=self.config.head_num_layers,
            use_bias=self.config.head_use_bias,
            dropout_rate=self.config.dropout_rate,
            activation=self.config.head_activation
        )
        return PredictionHead(pred_config)
    
    def _build_anchor_generator(self) -> AnchorGenerator:
        """🏗️ Construit le générateur d'anchors"""
        anchor_config = AnchorConfig(
            sizes=self.config.anchor_sizes,
            ratios=self.config.anchor_ratios,
            scales=self.config.anchor_scales,
            num_levels=self.config.fpn_levels
        )
        return AnchorGenerator(anchor_config)
    
    def _initialize_weights(self):
        """⚖️ Initialise les poids du modèle"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        🔄 Forward pass du modèle
        
        Args:
            images: Batch d'images [N, C, H, W]
            targets: Targets pour l'entraînement (optionnel)
            
        Returns:
            - En entraînement: dictionnaire des pertes
            - En inférence: liste des prédictions par image
        """
        
        # 1. Extraction des features via backbone
        features = self.backbone(images)
        
        # 2. Multi-scale features via FPN
        fpn_features = self.fpn(features)
        
        # 3. Prédictions via tête
        predictions = self.prediction_head(fpn_features)
        
        # 4. Génération des anchors
        anchors = self.anchor_generator(images, fpn_features)
        
        # 5. Mode training vs inference
        if self.training and targets is not None:
            return self._compute_losses(predictions, anchors, targets)
        else:
            return self._postprocess_predictions(predictions, anchors, images)
    
    def _compute_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        anchors: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """💥 Calcule les pertes d'entraînement"""
        
        # Import des utilitaires de perte
        from ..utils.losses import DetectionLoss
        
        loss_fn = DetectionLoss(
            num_classes=self.config.num_classes,
            alpha=0.25,  # Focal loss alpha
            gamma=2.0,   # Focal loss gamma
            smooth_l1_beta=0.1
        )
        
        return loss_fn(predictions, anchors, targets)
    
    def _postprocess_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        anchors: List[torch.Tensor],
        images: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """🔄 Post-traite les prédictions pour inférence"""
        
        # Import des utilitaires de post-traitement
        from ..utils.box_utils import decode_boxes, apply_nms
        
        batch_size = images.shape[0]
        results = []
        
        for i in range(batch_size):
            # Décodage des boîtes
            boxes = decode_boxes(
                predictions['boxes'][i],
                anchors,
                self.config.input_size
            )
            
            # Scores et classes
            scores = predictions['scores'][i]
            
            # Application NMS
            keep_indices = apply_nms(
                boxes, scores, 
                score_threshold=0.05,
                nms_threshold=0.5,
                max_detections=100
            )
            
            # Résultats finaux
            result = {
                'boxes': boxes[keep_indices],
                'scores': scores[keep_indices],
                'labels': torch.argmax(scores[keep_indices], dim=1)
            }
            results.append(result)
        
        return results
    
    def get_num_parameters(self) -> int:
        """📊 Retourne le nombre de paramètres"""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self) -> float:
        """📏 Retourne la taille du modèle en MB"""
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def get_model_info(self) -> Dict[str, Any]:
        """📋 Retourne les informations du modèle"""
        return {
            "architecture": self.config.architecture.value,
            "size": self.config.size.value,
            "num_classes": self.config.num_classes,
            "input_size": self.config.input_size,
            "num_parameters": self.get_num_parameters(),
            "model_size_mb": self.get_model_size_mb(),
            "backbone": self.config.backbone_name,
            "fpn_channels": self.config.fpn_channels,
            "head_channels": self.config.head_channels
        }
    
    def export_to_torchscript(self, output_path: str):
        """📦 Exporte le modèle en TorchScript"""
        self.eval()
        
        # Exemple d'entrée pour tracing
        example_input = torch.randn(1, 3, *self.config.input_size)
        
        try:
            # Tentative de tracing
            traced_model = torch.jit.trace(self, example_input)
            traced_model.save(output_path)
            logger.info(f"✅ Modèle exporté en TorchScript: {output_path}")
        except Exception as e:
            logger.warning(f"⚠️ Échec tracing, tentative scripting: {e}")
            try:
                # Fallback vers scripting
                scripted_model = torch.jit.script(self)
                scripted_model.save(output_path)
                logger.info(f"✅ Modèle exporté en TorchScript (script): {output_path}")
            except Exception as e2:
                logger.error(f"❌ Échec export TorchScript: {e2}")
                raise
    
    def export_to_onnx(self, output_path: str):
        """📦 Exporte le modèle en ONNX"""
        self.eval()
        
        example_input = torch.randn(1, 3, *self.config.input_size)
        
        try:
            torch.onnx.export(
                self,
                example_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            logger.info(f"✅ Modèle exporté en ONNX: {output_path}")
        except Exception as e:
            logger.error(f"❌ Échec export ONNX: {e}")
            raise

# 🏭 FONCTIONS FACTORY
def create_model(
    architecture: Union[str, ModelArchitecture],
    num_classes: int = 28,
    pretrained: bool = True,
    **kwargs
) -> LostObjectDetectionModel:
    """
    🏭 Factory pour créer un modèle pré-configuré
    
    Args:
        architecture: Type d'architecture
        num_classes: Nombre de classes
        pretrained: Utiliser backbone pré-entraîné
        **kwargs: Arguments additionnels
        
    Returns:
        Modèle configuré
    """
    
    if isinstance(architecture, str):
        architecture = ModelArchitecture(architecture)
    
    # Sélection de la configuration
    if architecture == ModelArchitecture.MOBILENET_SSD:
        config = ModelConfig.get_epoch30_config()
    elif architecture == ModelArchitecture.EFFICIENTDET:
        config = ModelConfig.get_extended_config()
    elif architecture == ModelArchitecture.YOLO_V5:
        config = ModelConfig.get_fast_config()
    else:
        config = ModelConfig()
    
    # Override des paramètres
    config.num_classes = num_classes
    config.backbone_pretrained = pretrained
    
    # Application des kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return LostObjectDetectionModel(config)

def create_epoch30_model(**kwargs) -> LostObjectDetectionModel:
    """🏆 Crée le modèle champion Epoch 30"""
    config = ModelConfig.get_epoch30_config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return LostObjectDetectionModel(config)

def create_extended_model(**kwargs) -> LostObjectDetectionModel:
    """🔧 Crée le modèle étendu"""
    config = ModelConfig.get_extended_config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return LostObjectDetectionModel(config)

def create_fast_model(**kwargs) -> LostObjectDetectionModel:
    """⚡ Crée le modèle rapide"""
    config = ModelConfig.get_fast_config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return LostObjectDetectionModel(config)

def create_mobile_model(**kwargs) -> LostObjectDetectionModel:
    """📱 Crée le modèle mobile"""
    config = ModelConfig.get_mobile_config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return LostObjectDetectionModel(config)

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "LostObjectDetectionModel",
    "ModelConfig",
    "ModelArchitecture", 
    "ModelSize",
    "create_model",
    "create_epoch30_model",
    "create_extended_model",
    "create_fast_model",
    "create_mobile_model"
]