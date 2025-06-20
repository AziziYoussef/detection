"""
Backbone Networks for Lost Objects Detection
Supports MobileNet, EfficientNet, and ResNet architectures
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BackboneBase(nn.Module):
    """Base class for all backbone networks"""
    
    def __init__(self):
        super().__init__()
        self.feature_channels = []
    
    def get_feature_channels(self) -> List[int]:
        """Return list of feature channels for each level"""
        return self.feature_channels
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning multi-scale features"""
        raise NotImplementedError

class MobileNetV3Backbone(BackboneBase):
    """MobileNetV3 Backbone for efficient object detection"""
    
    def __init__(self, variant: str = 'large', pretrained: bool = True):
        super().__init__()
        
        if variant == 'large':
            backbone = models.mobilenet_v3_large(pretrained=pretrained)
            # Feature channels for Large variant: [24, 40, 112, 960]
            self.feature_channels = [24, 40, 112, 960]
            self.feature_indices = [3, 6, 12, 16]  # Layer indices for feature extraction
        else:
            backbone = models.mobilenet_v3_small(pretrained=pretrained)
            # Feature channels for Small variant: [16, 24, 48, 576]
            self.feature_channels = [16, 24, 48, 576]
            self.feature_indices = [1, 3, 8, 11]
        
        # Extract feature layers
        self.features = backbone.features
        
        # Freeze early layers for stability
        self._freeze_early_layers(num_layers=3)
        
        logger.info(f"MobileNetV3-{variant} backbone initialized, pretrained={pretrained}")
    
    def _freeze_early_layers(self, num_layers: int):
        """Freeze early layers to stabilize training"""
        for i, layer in enumerate(self.features[:num_layers]):
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features"""
        features = []
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_indices:
                features.append(x)
        
        return features

class EfficientNetBackbone(BackboneBase):
    """EfficientNet Backbone for high-performance detection"""
    
    def __init__(self, variant: str = 'b0', pretrained: bool = True):
        super().__init__()
        
        # Import efficientnet (requires timm or torchvision >= 0.11)
        try:
            from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2
            
            backbone_map = {
                'b0': efficientnet_b0,
                'b1': efficientnet_b1, 
                'b2': efficientnet_b2
            }
            
            if variant not in backbone_map:
                raise ValueError(f"Unsupported EfficientNet variant: {variant}")
            
            backbone = backbone_map[variant](pretrained=pretrained)
            
            # Feature channels for different variants
            channels_map = {
                'b0': [24, 40, 112, 1280],
                'b1': [24, 40, 112, 1280],
                'b2': [24, 48, 120, 1408]
            }
            
            self.feature_channels = channels_map[variant]
            
        except ImportError:
            logger.warning("EfficientNet not available, falling back to MobileNetV3")
            # Fallback to MobileNetV3
            fallback = MobileNetV3Backbone('large', pretrained)
            self.features = fallback.features
            self.feature_channels = fallback.feature_channels
            self.feature_indices = fallback.feature_indices
            return
        
        self.features = backbone.features
        
        # Define feature extraction points
        if variant == 'b0':
            self.feature_indices = [2, 4, 6, 8]  # Approximate indices
        else:
            self.feature_indices = [2, 4, 6, 8]  # Adjust for other variants
        
        # Freeze early layers
        self._freeze_early_layers(num_layers=2)
        
        logger.info(f"EfficientNet-{variant} backbone initialized, pretrained={pretrained}")
    
    def _freeze_early_layers(self, num_layers: int):
        """Freeze early layers"""
        for i, layer in enumerate(self.features[:num_layers]):
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features"""
        features = []
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_indices:
                features.append(x)
        
        return features

class ResNetBackbone(BackboneBase):
    """ResNet Backbone for robust detection"""
    
    def __init__(self, variant: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        
        backbone_map = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101
        }
        
        if variant not in backbone_map:
            raise ValueError(f"Unsupported ResNet variant: {variant}")
        
        backbone = backbone_map[variant](pretrained=pretrained)
        
        # Feature channels for ResNet
        if variant in ['resnet18', 'resnet34']:
            self.feature_channels = [64, 128, 256, 512]
        else:  # ResNet50/101
            self.feature_channels = [256, 512, 1024, 2048]
        
        # Extract layers (removing final classifier)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Freeze early layers
        self._freeze_early_layers()
        
        logger.info(f"{variant} backbone initialized, pretrained={pretrained}")
    
    def _freeze_early_layers(self):
        """Freeze early convolutional layers"""
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features"""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Extract features from each residual block
        c2 = self.layer1(x)    # 1/4 resolution
        c3 = self.layer2(c2)   # 1/8 resolution  
        c4 = self.layer3(c3)   # 1/16 resolution
        c5 = self.layer4(c4)   # 1/32 resolution
        
        return [c2, c3, c4, c5]

class LightweightBackbone(BackboneBase):
    """Custom lightweight backbone for edge deployment"""
    
    def __init__(self, width_multiplier: float = 1.0):
        super().__init__()
        
        def make_divisible(v, divisor=8):
            return max(divisor, int(v + divisor / 2) // divisor * divisor)
        
        # Define channel progression
        base_channels = [32, 64, 128, 256]
        self.feature_channels = [
            make_divisible(ch * width_multiplier) for ch in base_channels
        ]
        
        # Build lightweight feature extractor
        self.features = nn.ModuleList([
            self._make_layer(3, self.feature_channels[0], stride=2),      # 1/2
            self._make_layer(self.feature_channels[0], self.feature_channels[1], stride=2),  # 1/4
            self._make_layer(self.feature_channels[1], self.feature_channels[2], stride=2),  # 1/8
            self._make_layer(self.feature_channels[2], self.feature_channels[3], stride=2),  # 1/16
        ])
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Lightweight backbone initialized, width_multiplier={width_multiplier}")
    
    def _make_layer(self, in_channels: int, out_channels: int, stride: int = 1):
        """Create a lightweight convolutional layer"""
        return nn.Sequential(
            # Depthwise separable convolution
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features"""
        features = []
        
        for layer in self.features:
            x = layer(x)
            features.append(x)
        
        return features

class BackboneFactory:
    """Factory class for creating backbone networks"""
    
    @staticmethod
    def create_backbone(
        backbone_name: str, 
        pretrained: bool = True,
        **kwargs
    ) -> BackboneBase:
        """
        Create backbone network
        
        Args:
            backbone_name: Name of backbone architecture
            pretrained: Whether to use pretrained weights
            **kwargs: Additional arguments for backbone
            
        Returns:
            Backbone network instance
        """
        backbone_name = backbone_name.lower()
        
        if backbone_name.startswith('mobilenet_v3'):
            variant = 'large' if 'large' in backbone_name else 'small'
            return MobileNetV3Backbone(variant=variant, pretrained=pretrained)
            
        elif backbone_name.startswith('efficientnet'):
            variant = backbone_name.split('_')[-1] if '_' in backbone_name else 'b0'
            return EfficientNetBackbone(variant=variant, pretrained=pretrained)
            
        elif backbone_name.startswith('resnet'):
            return ResNetBackbone(variant=backbone_name, pretrained=pretrained)
            
        elif backbone_name == 'lightweight':
            width_multiplier = kwargs.get('width_multiplier', 1.0)
            return LightweightBackbone(width_multiplier=width_multiplier)
            
        else:
            logger.warning(f"Unknown backbone: {backbone_name}, using MobileNetV3-Large")
            return MobileNetV3Backbone(variant='large', pretrained=pretrained)
    
    @staticmethod
    def get_available_backbones() -> List[str]:
        """Get list of available backbone architectures"""
        return [
            'mobilenet_v3_large',
            'mobilenet_v3_small', 
            'efficientnet_b0',
            'efficientnet_b1',
            'efficientnet_b2',
            'resnet18',
            'resnet34',
            'resnet50',
            'resnet101',
            'lightweight'
        ]
    
    @staticmethod
    def get_backbone_info(backbone_name: str) -> Dict:
        """Get information about a specific backbone"""
        info_map = {
            'mobilenet_v3_large': {
                'params': '5.4M',
                'flops': '219M',
                'description': 'Efficient mobile architecture, good speed/accuracy balance'
            },
            'mobilenet_v3_small': {
                'params': '2.9M', 
                'flops': '66M',
                'description': 'Ultra-lightweight for edge deployment'
            },
            'efficientnet_b0': {
                'params': '5.3M',
                'flops': '390M', 
                'description': 'Efficient scaling, high accuracy'
            },
            'resnet50': {
                'params': '25.6M',
                'flops': '4.1G',
                'description': 'Robust and proven architecture'
            },
            'lightweight': {
                'params': '1.2M',
                'flops': '45M',
                'description': 'Custom ultra-lightweight for real-time inference'
            }
        }
        
        return info_map.get(backbone_name, {
            'description': 'Custom backbone architecture'
        })