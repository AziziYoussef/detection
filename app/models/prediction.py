"""
Prediction Heads for Lost Objects Detection Model
Contains classification and regression heads for object detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PredictionHead(nn.Module):
    """
    Combined prediction head for object detection
    
    Contains both classification and regression heads with shared feature processing
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 28,
        num_anchors: int = 9,
        num_convs: int = 4,
        use_depthwise: bool = False,
        use_group_norm: bool = True,
        prior_prob: float = 0.01,
        num_groups: int = 32
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_convs = num_convs
        self.use_group_norm = use_group_norm
        self.num_groups = num_groups
        
        # Shared feature processing layers
        self.shared_convs = self._make_shared_layers(
            in_channels, 
            use_depthwise, 
            use_group_norm, 
            num_groups
        )
        
        # Classification head
        self.cls_head = ClassificationHead(
            in_channels=in_channels,
            num_classes=num_classes,
            num_anchors=num_anchors,
            num_convs=num_convs,
            use_depthwise=use_depthwise,
            use_group_norm=use_group_norm,
            prior_prob=prior_prob,
            num_groups=num_groups
        )
        
        # Regression head
        self.reg_head = RegressionHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_convs=num_convs,
            use_depthwise=use_depthwise,
            use_group_norm=use_group_norm,
            num_groups=num_groups
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"PredictionHead initialized: {num_classes} classes, {num_anchors} anchors")
    
    def _make_shared_layers(
        self, 
        in_channels: int, 
        use_depthwise: bool, 
        use_group_norm: bool,
        num_groups: int
    ) -> nn.ModuleList:
        """Create shared convolutional layers"""
        layers = nn.ModuleList()
        
        for i in range(self.num_convs):
            if use_depthwise:
                # Depthwise separable convolution
                conv = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
                    nn.Conv2d(in_channels, in_channels, 1, bias=False)
                )
            else:
                # Standard convolution
                conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
            
            layers.append(conv)
            
            # Normalization
            if use_group_norm:
                layers.append(nn.GroupNorm(num_groups, in_channels))
            else:
                layers.append(nn.BatchNorm2d(in_channels))
            
            # Activation
            layers.append(nn.ReLU(inplace=True))
        
        return layers
    
    def _initialize_weights(self):
        """Initialize prediction head weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through prediction head
        
        Args:
            features: List of feature maps from FPN [P3, P4, P5, P6, P7]
            
        Returns:
            Dictionary containing classification and regression predictions
        """
        cls_outputs = []
        reg_outputs = []
        
        for feature in features:
            # Apply shared convolutions
            shared_feat = feature
            for layer in self.shared_convs:
                shared_feat = layer(shared_feat)
            
            # Classification prediction
            cls_out = self.cls_head(shared_feat)
            cls_outputs.append(cls_out)
            
            # Regression prediction
            reg_out = self.reg_head(shared_feat)
            reg_outputs.append(reg_out)
        
        # Concatenate predictions from all levels
        cls_pred = torch.cat([
            cls_out.permute(0, 2, 3, 1).reshape(cls_out.shape[0], -1, self.num_classes)
            for cls_out in cls_outputs
        ], dim=1)
        
        reg_pred = torch.cat([
            reg_out.permute(0, 2, 3, 1).reshape(reg_out.shape[0], -1, 4)
            for reg_out in reg_outputs
        ], dim=1)
        
        return {
            'classification': cls_pred,
            'regression': reg_pred
        }

class ClassificationHead(nn.Module):
    """
    Classification head for object detection
    Predicts object classes for each anchor
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 28,
        num_anchors: int = 9,
        num_convs: int = 4,
        use_depthwise: bool = False,
        use_group_norm: bool = True,
        prior_prob: float = 0.01,
        num_groups: int = 32
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.prior_prob = prior_prob
        
        # Classification convolutions
        self.cls_convs = nn.ModuleList()
        
        for i in range(num_convs):
            if use_depthwise:
                conv = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
                    nn.Conv2d(in_channels, in_channels, 1, bias=False)
                )
            else:
                conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
            
            self.cls_convs.append(conv)
            
            # Normalization
            if use_group_norm:
                self.cls_convs.append(nn.GroupNorm(num_groups, in_channels))
            else:
                self.cls_convs.append(nn.BatchNorm2d(in_channels))
            
            # Activation
            self.cls_convs.append(nn.ReLU(inplace=True))
        
        # Final classification layer
        self.cls_pred = nn.Conv2d(
            in_channels, 
            num_anchors * num_classes, 
            kernel_size=3, 
            padding=1
        )
        
        # Initialize classification bias for better training stability
        self._initialize_cls_bias()
    
    def _initialize_cls_bias(self):
        """Initialize classification bias with prior probability"""
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head
        
        Args:
            x: Input feature tensor [N, C, H, W]
            
        Returns:
            Classification predictions [N, num_anchors * num_classes, H, W]
        """
        for layer in self.cls_convs:
            x = layer(x)
        
        cls_pred = self.cls_pred(x)
        return cls_pred

class RegressionHead(nn.Module):
    """
    Regression head for object detection
    Predicts bounding box deltas for each anchor
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_anchors: int = 9,
        num_convs: int = 4,
        use_depthwise: bool = False,
        use_group_norm: bool = True,
        num_groups: int = 32
    ):
        super().__init__()
        
        self.num_anchors = num_anchors
        
        # Regression convolutions
        self.reg_convs = nn.ModuleList()
        
        for i in range(num_convs):
            if use_depthwise:
                conv = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
                    nn.Conv2d(in_channels, in_channels, 1, bias=False)
                )
            else:
                conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
            
            self.reg_convs.append(conv)
            
            # Normalization
            if use_group_norm:
                self.reg_convs.append(nn.GroupNorm(num_groups, in_channels))
            else:
                self.reg_convs.append(nn.BatchNorm2d(in_channels))
            
            # Activation
            self.reg_convs.append(nn.ReLU(inplace=True))
        
        # Final regression layer (4 coordinates per anchor: dx, dy, dw, dh)
        self.reg_pred = nn.Conv2d(
            in_channels, 
            num_anchors * 4, 
            kernel_size=3, 
            padding=1
        )
        
        # Initialize regression weights
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through regression head
        
        Args:
            x: Input feature tensor [N, C, H, W]
            
        Returns:
            Regression predictions [N, num_anchors * 4, H, W]
        """
        for layer in self.reg_convs:
            x = layer(x)
        
        reg_pred = self.reg_pred(x)
        return reg_pred

class CenternessPredictionHead(nn.Module):
    """
    Centerness prediction head for FCOS-style detection
    Predicts how close a pixel is to the center of an object
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_convs: int = 4,
        use_group_norm: bool = True,
        num_groups: int = 32
    ):
        super().__init__()
        
        # Centerness convolutions
        self.centerness_convs = nn.ModuleList()
        
        for i in range(num_convs):
            conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
            self.centerness_convs.append(conv)
            
            # Normalization
            if use_group_norm:
                self.centerness_convs.append(nn.GroupNorm(num_groups, in_channels))
            else:
                self.centerness_convs.append(nn.BatchNorm2d(in_channels))
            
            # Activation
            self.centerness_convs.append(nn.ReLU(inplace=True))
        
        # Final centerness layer
        self.centerness_pred = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        # Initialize weights
        nn.init.normal_(self.centerness_pred.weight, std=0.01)
        nn.init.constant_(self.centerness_pred.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through centerness head
        
        Args:
            x: Input feature tensor [N, C, H, W]
            
        Returns:
            Centerness predictions [N, 1, H, W]
        """
        for layer in self.centerness_convs:
            x = layer(x)
        
        centerness = self.centerness_pred(x)
        return torch.sigmoid(centerness)

class QualityPredictionHead(nn.Module):
    """
    Quality prediction head for predicting detection quality
    Combines classification confidence with localization quality
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 28,
        num_anchors: int = 9,
        num_convs: int = 4,
        use_group_norm: bool = True,
        num_groups: int = 32
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Quality convolutions
        self.quality_convs = nn.ModuleList()
        
        for i in range(num_convs):
            conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
            self.quality_convs.append(conv)
            
            # Normalization
            if use_group_norm:
                self.quality_convs.append(nn.GroupNorm(num_groups, in_channels))
            else:
                self.quality_convs.append(nn.BatchNorm2d(in_channels))
            
            # Activation
            self.quality_convs.append(nn.ReLU(inplace=True))
        
        # Quality prediction (IoU estimation)
        self.quality_pred = nn.Conv2d(
            in_channels, 
            num_anchors * num_classes, 
            kernel_size=3, 
            padding=1
        )
        
        # Initialize weights
        nn.init.normal_(self.quality_pred.weight, std=0.01)
        nn.init.constant_(self.quality_pred.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quality head
        
        Args:
            x: Input feature tensor [N, C, H, W]
            
        Returns:
            Quality predictions [N, num_anchors * num_classes, H, W]
        """
        for layer in self.quality_convs:
            x = layer(x)
        
        quality = self.quality_pred(x)
        return torch.sigmoid(quality)

class MultiLevelPredictionHead(nn.Module):
    """
    Multi-level prediction head with feature pyramid integration
    Handles predictions across multiple feature pyramid levels
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 28,
        num_levels: int = 5,
        share_cls_reg: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.num_levels = num_levels
        self.share_cls_reg = share_cls_reg
        
        if share_cls_reg:
            # Shared prediction head across all levels
            self.prediction_head = PredictionHead(
                in_channels=in_channels,
                num_classes=num_classes,
                **kwargs
            )
        else:
            # Separate prediction heads for each level
            self.prediction_heads = nn.ModuleList([
                PredictionHead(
                    in_channels=in_channels,
                    num_classes=num_classes,
                    **kwargs
                )
                for _ in range(num_levels)
            ])
    
    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-level prediction head
        
        Args:
            features: List of feature maps from different levels
            
        Returns:
            Dictionary containing predictions from all levels
        """
        if self.share_cls_reg:
            return self.prediction_head(features)
        else:
            all_cls_preds = []
            all_reg_preds = []
            
            for i, feature in enumerate(features):
                if i < len(self.prediction_heads):
                    pred = self.prediction_heads[i]([feature])
                    all_cls_preds.append(pred['classification'])
                    all_reg_preds.append(pred['regression'])
            
            # Concatenate predictions from all levels
            cls_pred = torch.cat(all_cls_preds, dim=1)
            reg_pred = torch.cat(all_reg_preds, dim=1)
            
            return {
                'classification': cls_pred,
                'regression': reg_pred
            }

class LostObjectSpecificHead(nn.Module):
    """
    Specialized prediction head for lost objects detection
    Includes additional predictions for temporal tracking
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 28,
        num_anchors: int = 9,
        **kwargs
    ):
        super().__init__()
        
        # Standard prediction head
        self.base_head = PredictionHead(
            in_channels=in_channels,
            num_classes=num_classes,
            num_anchors=num_anchors,
            **kwargs
        )
        
        # Additional heads for lost object detection
        self.abandonment_head = nn.Conv2d(
            in_channels, num_anchors, kernel_size=3, padding=1
        )
        
        self.temporal_head = nn.Conv2d(
            in_channels, num_anchors * 2, kernel_size=3, padding=1  # [duration, movement]
        )
        
        # Initialize additional heads
        nn.init.normal_(self.abandonment_head.weight, std=0.01)
        nn.init.constant_(self.abandonment_head.bias, 0)
        nn.init.normal_(self.temporal_head.weight, std=0.01)
        nn.init.constant_(self.temporal_head.bias, 0)
    
    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for lost object specific predictions
        
        Args:
            features: List of feature maps
            
        Returns:
            Dictionary with standard and temporal predictions
        """
        # Get standard predictions
        predictions = self.base_head(features)
        
        # Add temporal predictions for the first feature level
        if features:
            feat = features[0]
            
            # Abandonment probability
            abandonment = torch.sigmoid(self.abandonment_head(feat))
            abandonment = abandonment.permute(0, 2, 3, 1).reshape(
                abandonment.shape[0], -1, 1
            )
            
            # Temporal features
            temporal = self.temporal_head(feat)
            temporal = temporal.permute(0, 2, 3, 1).reshape(
                temporal.shape[0], -1, 2
            )
            
            predictions.update({
                'abandonment': abandonment,
                'temporal': temporal
            })
        
        return predictions

def create_prediction_head(head_type: str = "standard", **kwargs) -> nn.Module:
    """
    Factory function to create different types of prediction heads
    
    Args:
        head_type: Type of prediction head to create
        **kwargs: Additional arguments for the head
        
    Returns:
        Prediction head instance
    """
    if head_type == "standard":
        return PredictionHead(**kwargs)
    elif head_type == "multi_level":
        return MultiLevelPredictionHead(**kwargs)
    elif head_type == "lost_object":
        return LostObjectSpecificHead(**kwargs)
    elif head_type == "centerness":
        return CenternessPredictionHead(**kwargs)
    elif head_type == "quality":
        return QualityPredictionHead(**kwargs)
    else:
        raise ValueError(f"Unknown prediction head type: {head_type}")

# Export main classes
__all__ = [
    'PredictionHead',
    'ClassificationHead', 
    'RegressionHead',
    'CenternessPredictionHead',
    'QualityPredictionHead',
    'MultiLevelPredictionHead',
    'LostObjectSpecificHead',
    'create_prediction_head'
]