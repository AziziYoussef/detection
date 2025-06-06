"""
⚙️ CONFIGURATION GÉNÉRALE - SETTINGS PRINCIPAUX
==============================================
Configuration centrale du service IA de détection d'objets perdus

Fonctionnalités:
- Chargement depuis variables d'environnement
- Configuration par défaut robuste
- Validation des paramètres
- Support multi-environnements (dev/prod)
- Intégration avec Docker et Kubernetes
"""

import os
import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from functools import lru_cache
from pydantic import BaseSettings, validator, Field
from enum import Enum

logger = logging.getLogger(__name__)

# 📋 ÉNUMÉRATIONS
class Environment(str, Enum):
    """🌍 Environnements d'exécution"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(str, Enum):
    """📝 Niveaux de logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class DeviceType(str, Enum):
    """🖥️ Types de devices"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon

# ⚙️ CLASSE DE CONFIGURATION PRINCIPALE
class Settings(BaseSettings):
    """⚙️ Configuration principale du service"""
    
    # 🏷️ INFORMATIONS DU SERVICE
    SERVICE_NAME: str = "AI Lost Objects Detection Service"
    SERVICE_VERSION: str = "1.0.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # 🌐 CONFIGURATION RÉSEAU
    HOST: str = Field(default="0.0.0.0", env="AI_SERVICE_HOST")
    PORT: int = Field(default=8001, env="AI_SERVICE_PORT")
    WORKERS: int = Field(default=1, env="AI_SERVICE_WORKERS")
    
    # 🔗 URLS DES SERVICES
    BACKEND_URL: str = Field(default="http://localhost:8080", env="BACKEND_URL")
    FRONTEND_URL: str = Field(default="http://localhost:3000", env="FRONTEND_URL")
    
    # 📊 BASE DE DONNÉES (pour analytics)
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    REDIS_URL: Optional[str] = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # 📝 CONFIGURATION LOGGING
    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    LOG_FILE: Optional[str] = Field(default="ai_service.log", env="LOG_FILE")
    LOG_MAX_SIZE: int = Field(default=10 * 1024 * 1024, env="LOG_MAX_SIZE")  # 10MB
    LOG_BACKUP_COUNT: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # 🤖 CONFIGURATION MODÈLES
    MODELS_PATH: str = Field(default="storage/models", env="MODELS_PATH")
    DEVICE: DeviceType = Field(default=DeviceType.AUTO, env="DEVICE")
    BATCH_SIZE: int = Field(default=8, env="BATCH_SIZE")
    HALF_PRECISION: bool = Field(default=False, env="HALF_PRECISION")
    MODEL_CACHE_SIZE: int = Field(default=3, env="MODEL_CACHE_SIZE")
    
    # 🎯 CONFIGURATION DÉTECTION
    DEFAULT_CONFIDENCE_THRESHOLD: float = Field(default=0.5, env="DEFAULT_CONFIDENCE_THRESHOLD")
    DEFAULT_NMS_THRESHOLD: float = Field(default=0.4, env="DEFAULT_NMS_THRESHOLD")
    MAX_DETECTIONS: int = Field(default=100, env="MAX_DETECTIONS")
    
    # 🏷️ CLASSES DE DÉTECTION
    DETECTION_CLASSES: List[str] = [
        "bag", "suitcase", "backpack", "handbag", "briefcase",
        "phone", "smartphone", "tablet", "laptop", "camera",
        "keys", "wallet", "watch", "glasses", "sunglasses",
        "hat", "cap", "scarf", "gloves", "umbrella",
        "book", "notebook", "magazine", "bottle", "cup",
        "headphones", "charger", "toy"
    ]
    
    CLASSES_FR_NAMES: List[str] = [
        "sac", "valise", "sac à dos", "sac à main", "mallette",
        "téléphone", "smartphone", "tablette", "ordinateur portable", "appareil photo",
        "clés", "portefeuille", "montre", "lunettes", "lunettes de soleil",
        "chapeau", "casquette", "écharpe", "gants", "parapluie",
        "livre", "carnet", "magazine", "bouteille", "tasse",
        "écouteurs", "chargeur", "jouet"
    ]
    
    # 📁 CHEMINS DE STOCKAGE
    STORAGE_PATH: str = Field(default="storage", env="STORAGE_PATH")
    TEMP_PATH: str = Field(default="storage/temp", env="TEMP_PATH")
    UPLOADS_PATH: str = Field(default="storage/temp/uploads", env="UPLOADS_PATH")
    RESULTS_PATH: str = Field(default="storage/temp/results", env="RESULTS_PATH")
    CACHE_PATH: str = Field(default="storage/cache", env="CACHE_PATH")
    
    # 📡 CONFIGURATION WEBSOCKET
    WEBSOCKET_MAX_CONNECTIONS: int = Field(default=50, env="WEBSOCKET_MAX_CONNECTIONS")
    WEBSOCKET_PING_INTERVAL: int = Field(default=30, env="WEBSOCKET_PING_INTERVAL")
    WEBSOCKET_PING_TIMEOUT: int = Field(default=60, env="WEBSOCKET_PING_TIMEOUT")
    
    # 🔄 CONFIGURATION CACHE
    CACHE_TTL: int = Field(default=300, env="CACHE_TTL")  # 5 minutes
    CACHE_MAX_SIZE: int = Field(default=1000, env="CACHE_MAX_SIZE")
    RESULTS_CACHE_ENABLED: bool = Field(default=True, env="RESULTS_CACHE_ENABLED")
    
    # 📊 CONFIGURATION MONITORING
    METRICS_ENABLED: bool = Field(default=True, env="METRICS_ENABLED")
    HEALTH_CHECK_INTERVAL: int = Field(default=60, env="HEALTH_CHECK_INTERVAL")
    PERFORMANCE_MONITORING: bool = Field(default=True, env="PERFORMANCE_MONITORING")
    
    # 🔒 CONFIGURATION SÉCURITÉ
    SECRET_KEY: str = Field(default="ai-service-secret-key-change-in-production", env="SECRET_KEY")
    API_KEY_HEADER: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    ALLOWED_ORIGINS: List[str] = Field(default=[], env="ALLOWED_ORIGINS")
    
    # 📤 CONFIGURATION UPLOAD
    MAX_UPLOAD_SIZE: int = Field(default=50 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 50MB
    ALLOWED_EXTENSIONS: List[str] = Field(
        default=["jpg", "jpeg", "png", "bmp", "tiff", "webp", "mp4", "avi", "mov", "mkv"],
        env="ALLOWED_EXTENSIONS"
    )
    
    # ⚡ CONFIGURATION PERFORMANCE
    ENABLE_PROFILING: bool = Field(default=False, env="ENABLE_PROFILING")
    MAX_CONCURRENT_REQUESTS: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    REQUEST_TIMEOUT: int = Field(default=300, env="REQUEST_TIMEOUT")  # 5 minutes
    
    # 🔧 CONFIGURATION AVANCÉE
    TORCH_THREADS: Optional[int] = Field(default=None, env="TORCH_THREADS")
    OPENCV_THREADS: Optional[int] = Field(default=None, env="OPENCV_THREADS")
    MEMORY_LIMIT_MB: Optional[int] = Field(default=None, env="MEMORY_LIMIT_MB")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
    # 🔍 VALIDATORS
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v
    
    @validator("DEVICE")
    def validate_device(cls, v):
        if isinstance(v, str):
            return DeviceType(v.lower())
        return v
    
    @validator("DEFAULT_CONFIDENCE_THRESHOLD")
    def validate_confidence_threshold(cls, v):
        if not 0.1 <= v <= 0.9:
            raise ValueError("confidence_threshold doit être entre 0.1 et 0.9")
        return v
    
    @validator("DEFAULT_NMS_THRESHOLD")
    def validate_nms_threshold(cls, v):
        if not 0.1 <= v <= 0.9:
            raise ValueError("nms_threshold doit être entre 0.1 et 0.9")
        return v
    
    @validator("PORT")
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port doit être entre 1 et 65535")
        return v
    
    @validator("WORKERS")
    def validate_workers(cls, v):
        if v < 1:
            raise ValueError("Nombre de workers doit être >= 1")
        return v
    
    # 🛠️ MÉTHODES UTILITAIRES
    def get_model_path(self, model_name: str) -> Path:
        """📁 Retourne le chemin complet d'un modèle"""
        return Path(self.MODELS_PATH) / f"{model_name}.pth"
    
    def get_storage_path(self, subdir: str = "") -> Path:
        """📁 Retourne un chemin de stockage"""
        base_path = Path(self.STORAGE_PATH)
        if subdir:
            return base_path / subdir
        return base_path
    
    def ensure_directories(self):
        """📁 Crée les répertoires nécessaires"""
        directories = [
            self.STORAGE_PATH,
            self.TEMP_PATH,
            self.UPLOADS_PATH,
            self.RESULTS_PATH,
            self.CACHE_PATH,
            self.MODELS_PATH
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"📁 Répertoire créé/vérifié: {directory}")
    
    def is_production(self) -> bool:
        """🏭 Vérifie si on est en production"""
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """🔧 Vérifie si on est en développement"""
        return self.ENVIRONMENT == Environment.DEVELOPMENT
    
    def get_cors_settings(self) -> Dict[str, Any]:
        """🌐 Retourne les paramètres CORS"""
        default_origins = [
            f"http://localhost:{self.PORT}",
            self.FRONTEND_URL,
            self.BACKEND_URL
        ]
        
        if self.ALLOWED_ORIGINS:
            origins = self.ALLOWED_ORIGINS + default_origins
        else:
            origins = default_origins
        
        return {
            "allow_origins": origins,
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["*"]
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """📝 Retourne la configuration de logging"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.LOG_LEVEL.value,
                    "formatter": "standard",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": self.LOG_LEVEL.value,
                    "formatter": "detailed",
                    "filename": self.LOG_FILE,
                    "maxBytes": self.LOG_MAX_SIZE,
                    "backupCount": self.LOG_BACKUP_COUNT
                }
            },
            "loggers": {
                "": {
                    "handlers": ["console", "file"],
                    "level": self.LOG_LEVEL.value,
                    "propagate": False
                }
            }
        }
    
    def model_post_init(self, __context: Any) -> None:
        """🔧 Post-initialisation"""
        # Créer les répertoires nécessaires
        self.ensure_directories()
        
        # Configuration des threads si spécifié
        if self.TORCH_THREADS:
            import torch
            torch.set_num_threads(self.TORCH_THREADS)
            
        if self.OPENCV_THREADS:
            import cv2
            cv2.setNumThreads(self.OPENCV_THREADS)

# 🏭 FACTORY FUNCTION
@lru_cache()
def get_settings() -> Settings:
    """
    🏭 FACTORY POUR RÉCUPÉRER LA CONFIGURATION
    Utilise LRU cache pour éviter de recharger la config
    """
    return Settings()

# 🛠️ FONCTIONS UTILITAIRES
def get_environment() -> Environment:
    """🌍 Récupère l'environnement actuel"""
    return get_settings().ENVIRONMENT

def is_production() -> bool:
    """🏭 Vérifie si on est en production"""
    return get_settings().is_production()

def is_development() -> bool:
    """🔧 Vérifie si on est en développement"""
    return get_settings().is_development()