# app/config/config.py
import os
from pathlib import Path
from typing import Dict, List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Configuration du service IA"""
    
    # === CONFIGURATION SERVEUR ===
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    WORKERS: int = 1
    
    # === CHEMINS ===
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    STORAGE_DIR: Path = BASE_DIR / "storage"
    MODELS_DIR: Path = STORAGE_DIR / "models"
    TEMP_DIR: Path = STORAGE_DIR / "temp"
    CACHE_DIR: Path = STORAGE_DIR / "cache"
    
    # === CONFIGURATION MODÈLES ===
    DEFAULT_MODEL: str = "stable_model_epoch_30.pth"
    EXTENDED_MODEL: str = "best_extended_model.pth"
    FAST_MODEL: str = "fast_stream_model.pth"
    
    # === PARAMÈTRES DÉTECTION ===
    CONFIDENCE_THRESHOLD: float = 0.5
    NMS_THRESHOLD: float = 0.5
    MAX_DETECTIONS: int = 50
    IMAGE_SIZE: tuple = (320, 320)
    
    # === STREAMING ===
    MAX_CONNECTIONS: int = 10
    STREAM_FPS: int = 15
    BUFFER_SIZE: int = 30
    
    # === OBJETS PERDUS - LOGIQUE MÉTIER ===
    SUSPECT_THRESHOLD_SECONDS: int = 30
    LOST_THRESHOLD_SECONDS: int = 300  # 5 minutes
    CRITICAL_THRESHOLD_SECONDS: int = 1800  # 30 minutes
    OWNER_PROXIMITY_METERS: float = 2.5
    
    # === PERFORMANCE ===
    USE_GPU: bool = False  # Désactivé par défaut pour éviter les problèmes
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 0
    MAX_MEMORY_USAGE: float = 0.8
    
    # === CACHE ===
    CACHE_TTL: int = 3600
    MAX_CACHE_SIZE: int = 100
    
    class Config:
        env_file = ".env"

# Instance globale des paramètres
settings = Settings()

# Configuration des modèles simplifiée
MODEL_CONFIG = {
    'num_classes': 28,
    'image_size': (320, 320),
    'confidence_threshold': 0.5,
    'nms_threshold': 0.5,
    'max_detections': 50,
    
    'classes': [
        'person', 'backpack', 'suitcase', 'handbag', 'tie',
        'umbrella', 'hair drier', 'toothbrush', 'cell phone',
        'laptop', 'keyboard', 'mouse', 'remote', 'tv',
        'clock', 'microwave', 'bottle', 'cup', 'bowl',
        'knife', 'spoon', 'fork', 'wine glass', 'refrigerator',
        'scissors', 'book', 'vase', 'chair'
    ],
    
    'class_names_fr': {
        'person': 'Personne',
        'backpack': 'Sac à dos',
        'suitcase': 'Valise',
        'handbag': 'Sac à main',
        'tie': 'Cravate',
        'hair drier': 'Sèche-cheveux',
        'toothbrush': 'Brosse à dents',
        'cell phone': 'Téléphone',
        'laptop': 'Ordinateur portable',
        'keyboard': 'Clavier',
        'mouse': 'Souris',
        'remote': 'Télécommande',
        'tv': 'Télévision',
        'bottle': 'Bouteille',
        'cup': 'Tasse',
        'bowl': 'Bol',
        'knife': 'Couteau',
        'spoon': 'Cuillère',
        'fork': 'Fourchette',
        'wine glass': 'Verre',
        'scissors': 'Ciseaux',
        'book': 'Livre',
        'clock': 'Horloge',
        'umbrella': 'Parapluie',
        'vase': 'Vase',
        'chair': 'Chaise',
        'microwave': 'Micro-ondes',
        'refrigerator': 'Réfrigérateur'
    }
}

# Configuration simplifiée des objets perdus
LOST_OBJECT_CONFIG = {
    'temporal_thresholds': {
        'surveillance': 30,
        'alert': 300,
        'critical': 1800,
        'escalation': 3600
    },
    
    'spatial_thresholds': {
        'owner_proximity': 2.5,
        'movement_threshold': 0.5,
        'zone_boundary': 10.0
    },
    
    'confidence_thresholds': {
        'object_detection': 0.5,
        'tracking_stability': 0.8,
        'person_association': 0.6
    },
    
    'blacklist_objects': [
        'chair', 'tv', 'refrigerator', 'microwave'
    ],
    
    'priority_objects': [
        'backpack', 'suitcase', 'handbag', 'laptop', 'cell phone'
    ]
}

def ensure_directories():
    """Crée les répertoires nécessaires"""
    directories = [
        settings.STORAGE_DIR,
        settings.MODELS_DIR,
        settings.TEMP_DIR,
        settings.CACHE_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialisation des répertoires
ensure_directories()
