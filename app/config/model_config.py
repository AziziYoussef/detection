"""
🤖 CONFIGURATION DES MODÈLES - SETTINGS SPÉCIALISÉS
===================================================
Configuration dédiée aux modèles de détection d'objets perdus

Fonctionnalités:
- Configuration par modèle (Epoch 30, Extended, Fast, Mobile)
- Paramètres d'optimisation GPU/CPU
- Configurations d'entraînement et d'inférence
- Métadonnées des modèles
- Chemins et versions des modèles
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from functools import lru_cache
from pydantic import BaseSettings, Field, validator
from enum import Enum
from dataclasses import dataclass

# 🏷️ ÉNUMÉRATIONS
class ModelType(str, Enum):
    """🤖 Types de modèles disponibles"""
    EPOCH_30 = "epoch_30"          # Champion F1=49.86%
    EXTENDED = "extended"          # 28 classes étendues
    FAST = "fast"                  # Optimisé vitesse
    MOBILE = "mobile"              # Optimisé mobile/edge

class ModelArchitecture(str, Enum):
    """🏗️ Architectures de modèles"""
    MOBILENET_SSD = "mobilenet_ssd"
    EFFICIENTDET = "efficientdet"
    YOLO_V5 = "yolo_v5"
    RETINANET = "retinanet"

class ModelPrecision(str, Enum):
    """🎯 Précisions des modèles"""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"

@dataclass
class ModelMetadata:
    """📋 Métadonnées d'un modèle"""
    name: str
    version: str
    architecture: ModelArchitecture
    num_classes: int
    input_size: Tuple[int, int]
    precision: ModelPrecision
    file_size_mb: float
    accuracy_metrics: Dict[str, float]
    training_info: Dict[str, Any]
    tags: List[str]

# ⚙️ CONFIGURATION DES MODÈLES
class ModelConfiguration(BaseSettings):
    """🤖 Configuration des modèles de détection"""
    
    # 📁 CHEMINS DES MODÈLES
    MODELS_BASE_PATH: str = Field(default="storage/models", env="MODELS_BASE_PATH")
    MODELS_CONFIG_PATH: str = Field(default="storage/models/configs", env="MODELS_CONFIG_PATH")
    MODELS_CACHE_PATH: str = Field(default="storage/cache/models", env="MODELS_CACHE_PATH")
    
    # 🏆 MODÈLE EPOCH 30 (CHAMPION)
    EPOCH_30_PATH: str = Field(default="storage/models/stable_model_epoch_30.pth", env="EPOCH_30_PATH")
    EPOCH_30_CONFIG: str = Field(default="storage/models/config_epoch_30.py", env="EPOCH_30_CONFIG")
    EPOCH_30_ENABLED: bool = Field(default=True, env="EPOCH_30_ENABLED")
    EPOCH_30_PRIORITY: int = Field(default=1, env="EPOCH_30_PRIORITY")  # Priorité maximale
    
    # 🔧 MODÈLE EXTENDED
    EXTENDED_PATH: str = Field(default="storage/models/best_extended_model.pth", env="EXTENDED_PATH")
    EXTENDED_CONFIG: str = Field(default="storage/models/config_extended.py", env="EXTENDED_CONFIG")
    EXTENDED_ENABLED: bool = Field(default=True, env="EXTENDED_ENABLED")
    EXTENDED_PRIORITY: int = Field(default=2, env="EXTENDED_PRIORITY")
    
    # ⚡ MODÈLE FAST
    FAST_PATH: str = Field(default="storage/models/fast_model.pth", env="FAST_PATH")
    FAST_CONFIG: str = Field(default="storage/models/config_fast.py", env="FAST_CONFIG")
    FAST_ENABLED: bool = Field(default=True, env="FAST_ENABLED")
    FAST_PRIORITY: int = Field(default=3, env="FAST_PRIORITY")
    
    # 📱 MODÈLE MOBILE
    MOBILE_PATH: str = Field(default="storage/models/mobile_model.pth", env="MOBILE_PATH")
    MOBILE_CONFIG: str = Field(default="storage/models/config_mobile.py", env="MOBILE_CONFIG")
    MOBILE_ENABLED: bool = Field(default=False, env="MOBILE_ENABLED")  # Désactivé par défaut
    MOBILE_PRIORITY: int = Field(default=4, env="MOBILE_PRIORITY")
    
    # 🎯 PARAMÈTRES D'INFÉRENCE
    DEFAULT_INPUT_SIZE: Tuple[int, int] = Field(default=(640, 640), env="DEFAULT_INPUT_SIZE")
    DEFAULT_BATCH_SIZE: int = Field(default=8, env="DEFAULT_BATCH_SIZE")
    MAX_BATCH_SIZE: int = Field(default=32, env="MAX_BATCH_SIZE")
    
    # 🔧 OPTIMISATIONS
    ENABLE_HALF_PRECISION: bool = Field(default=False, env="ENABLE_HALF_PRECISION")
    ENABLE_TENSORRT: bool = Field(default=False, env="ENABLE_TENSORRT")
    ENABLE_ONNX: bool = Field(default=False, env="ENABLE_ONNX")
    ENABLE_TORCHSCRIPT: bool = Field(default=True, env="ENABLE_TORCHSCRIPT")
    
    # 💾 GESTION MÉMOIRE
    MODEL_CACHE_SIZE: int = Field(default=3, env="MODEL_CACHE_SIZE")
    MEMORY_LIMIT_MB: Optional[int] = Field(default=None, env="MODEL_MEMORY_LIMIT_MB")
    GPU_MEMORY_FRACTION: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")
    
    # 🔄 PRÉCHARGEMENT
    PRELOAD_MODELS: List[str] = Field(default=["epoch_30"], env="PRELOAD_MODELS")
    WARMUP_ITERATIONS: int = Field(default=5, env="WARMUP_ITERATIONS")
    
    # 📊 MÉTRIQUES
    COLLECT_METRICS: bool = Field(default=True, env="COLLECT_METRICS")
    METRICS_INTERVAL: int = Field(default=60, env="METRICS_INTERVAL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    # 🔍 VALIDATORS
    @validator("DEFAULT_BATCH_SIZE")
    def validate_batch_size(cls, v):
        if v < 1 or v > 64:
            raise ValueError("Batch size doit être entre 1 et 64")
        return v
    
    @validator("GPU_MEMORY_FRACTION")
    def validate_gpu_memory_fraction(cls, v):
        if not 0.1 <= v <= 1.0:
            raise ValueError("GPU memory fraction doit être entre 0.1 et 1.0")
        return v
    
    # 🛠️ MÉTHODES UTILITAIRES
    def get_model_path(self, model_type: ModelType) -> Path:
        """📁 Retourne le chemin d'un modèle"""
        path_mapping = {
            ModelType.EPOCH_30: self.EPOCH_30_PATH,
            ModelType.EXTENDED: self.EXTENDED_PATH,
            ModelType.FAST: self.FAST_PATH,
            ModelType.MOBILE: self.MOBILE_PATH
        }
        return Path(path_mapping[model_type])
    
    def get_model_config_path(self, model_type: ModelType) -> Path:
        """📁 Retourne le chemin de config d'un modèle"""
        config_mapping = {
            ModelType.EPOCH_30: self.EPOCH_30_CONFIG,
            ModelType.EXTENDED: self.EXTENDED_CONFIG,
            ModelType.FAST: self.FAST_CONFIG,
            ModelType.MOBILE: self.MOBILE_CONFIG
        }
        return Path(config_mapping[model_type])
    
    def is_model_enabled(self, model_type: ModelType) -> bool:
        """✅ Vérifie si un modèle est activé"""
        enabled_mapping = {
            ModelType.EPOCH_30: self.EPOCH_30_ENABLED,
            ModelType.EXTENDED: self.EXTENDED_ENABLED,
            ModelType.FAST: self.FAST_ENABLED,
            ModelType.MOBILE: self.MOBILE_ENABLED
        }
        return enabled_mapping[model_type]
    
    def get_model_priority(self, model_type: ModelType) -> int:
        """🎯 Retourne la priorité d'un modèle"""
        priority_mapping = {
            ModelType.EPOCH_30: self.EPOCH_30_PRIORITY,
            ModelType.EXTENDED: self.EXTENDED_PRIORITY,
            ModelType.FAST: self.FAST_PRIORITY,
            ModelType.MOBILE: self.MOBILE_PRIORITY
        }
        return priority_mapping[model_type]
    
    def get_enabled_models(self) -> List[ModelType]:
        """📋 Retourne la liste des modèles activés"""
        enabled_models = []
        for model_type in ModelType:
            if self.is_model_enabled(model_type):
                enabled_models.append(model_type)
        return sorted(enabled_models, key=self.get_model_priority)
    
    def get_model_metadata(self, model_type: ModelType) -> ModelMetadata:
        """📋 Retourne les métadonnées d'un modèle"""
        
        # Métadonnées prédéfinies par modèle
        metadata_mapping = {
            ModelType.EPOCH_30: ModelMetadata(
                name="Epoch 30 Champion",
                version="1.0.0",
                architecture=ModelArchitecture.MOBILENET_SSD,
                num_classes=28,
                input_size=(640, 640),
                precision=ModelPrecision.FP32,
                file_size_mb=45.2,
                accuracy_metrics={
                    "f1_score": 0.4986,
                    "precision": 0.6073,
                    "recall": 0.4232,
                    "mAP": 0.4156
                },
                training_info={
                    "epochs": 30,
                    "dataset_size": 8400,
                    "training_time_hours": 12.5,
                    "optimizer": "Adam",
                    "learning_rate": 0.001
                },
                tags=["champion", "balanced", "production"]
            ),
            
            ModelType.EXTENDED: ModelMetadata(
                name="Extended Classes Model",
                version="1.1.0",
                architecture=ModelArchitecture.EFFICIENTDET,
                num_classes=28,
                input_size=(512, 512),
                precision=ModelPrecision.FP32,
                file_size_mb=67.8,
                accuracy_metrics={
                    "f1_score": 0.5234,
                    "precision": 0.6445,
                    "recall": 0.4567,
                    "mAP": 0.4789
                },
                training_info={
                    "epochs": 50,
                    "dataset_size": 12000,
                    "training_time_hours": 24.3,
                    "optimizer": "AdamW",
                    "learning_rate": 0.0005
                },
                tags=["extended", "accuracy", "comprehensive"]
            ),
            
            ModelType.FAST: ModelMetadata(
                name="Fast Streaming Model",
                version="1.0.0",
                architecture=ModelArchitecture.YOLO_V5,
                num_classes=28,
                input_size=(416, 416),
                precision=ModelPrecision.FP16,
                file_size_mb=28.1,
                accuracy_metrics={
                    "f1_score": 0.4523,
                    "precision": 0.5678,
                    "recall": 0.3789,
                    "mAP": 0.3845
                },
                training_info={
                    "epochs": 25,
                    "dataset_size": 6000,
                    "training_time_hours": 6.7,
                    "optimizer": "SGD",
                    "learning_rate": 0.01
                },
                tags=["fast", "streaming", "realtime"]
            ),
            
            ModelType.MOBILE: ModelMetadata(
                name="Mobile Edge Model",
                version="1.0.0",
                architecture=ModelArchitecture.MOBILENET_SSD,
                num_classes=28,
                input_size=(320, 320),
                precision=ModelPrecision.INT8,
                file_size_mb=12.4,
                accuracy_metrics={
                    "f1_score": 0.3987,
                    "precision": 0.4923,
                    "recall": 0.3345,
                    "mAP": 0.3234
                },
                training_info={
                    "epochs": 20,
                    "dataset_size": 4500,
                    "training_time_hours": 3.2,
                    "optimizer": "Adam",
                    "learning_rate": 0.001
                },
                tags=["mobile", "edge", "lightweight"]
            )
        }
        
        return metadata_mapping[model_type]
    
    def get_all_models_info(self) -> Dict[str, Dict[str, Any]]:
        """📊 Retourne toutes les infos des modèles"""
        models_info = {}
        
        for model_type in ModelType:
            metadata = self.get_model_metadata(model_type)
            models_info[model_type.value] = {
                "enabled": self.is_model_enabled(model_type),
                "priority": self.get_model_priority(model_type),
                "path": str(self.get_model_path(model_type)),
                "config_path": str(self.get_model_config_path(model_type)),
                "metadata": metadata.__dict__
            }
        
        return models_info

# 🏭 FACTORY FUNCTION
@lru_cache()
def get_model_config() -> ModelConfiguration:
    """🏭 Factory pour récupérer la configuration des modèles"""
    return ModelConfiguration()