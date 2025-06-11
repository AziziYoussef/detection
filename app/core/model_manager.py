# app/core/model_manager.py
import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import Dict, Optional, List
from datetime import datetime
import psutil
import gc
from collections import OrderedDict
import asyncio

from app.config.config import settings, MODEL_CONFIG
from app.core.detector import ObjectDetector

logger = logging.getLogger(__name__)

class ModelInfo:
    """Informations sur un modèle"""
    def __init__(self, name: str, path: Path, config: dict):
        self.name = name
        self.path = path
        self.config = config
        self.model = None
        self.detector = None
        self.is_loaded = False
        self.load_time = None
        self.memory_usage = 0
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.last_used = None

class LRUCache:
    """Cache LRU pour les modèles"""
    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[ModelInfo]:
        if key in self.cache:
            # Déplacer à la fin (plus récent)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, model_info: ModelInfo):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = model_info
            if len(self.cache) > self.max_size:
                # Supprimer le plus ancien
                oldest_key, oldest_model = self.cache.popitem(last=False)
                self._unload_model(oldest_model)
    
    def _unload_model(self, model_info: ModelInfo):
        """Décharge un modèle de la mémoire"""
        if model_info.is_loaded:
            logger.info(f"Déchargement du modèle {model_info.name}")
            del model_info.model
            del model_info.detector
            model_info.model = None
            model_info.detector = None
            model_info.is_loaded = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class SimpleDetectionModel(nn.Module):
    """Modèle de détection simple (placeholder pour votre modèle réel)"""
    
    def __init__(self, num_classes: int = 28, input_size: int = 320):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Exemple d'architecture simple (à remplacer par votre modèle)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, 10))
        )
        
        # Tête de détection
        self.detection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 10 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes * 6)  # 6 = x, y, w, h, conf, class
        )
    
    def forward(self, x):
        features = self.backbone(x)
        detections = self.detection_head(features)
        
        # Reshape pour format détection
        batch_size = x.size(0)
        detections = detections.view(batch_size, -1, 6)
        
        return detections

class ModelManager:
    """Gestionnaire des modèles de détection"""
    
    def __init__(self):
        self.device = self._setup_device()
        self.models_cache = LRUCache(max_size=3)
        self.available_models = self._discover_models()
        self.default_model = settings.DEFAULT_MODEL
        self.stats = {
            'total_loads': 0,
            'total_inferences': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(f"ModelManager initialisé sur {self.device}")
        logger.info(f"Modèles disponibles: {list(self.available_models.keys())}")
    
    def _setup_device(self) -> torch.device:
        """Configure le device (GPU/CPU)"""
        if settings.USE_GPU and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"GPU détecté: {torch.cuda.get_device_name()}")
            logger.info(f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device('cpu')
            logger.info("Utilisation du CPU")
        
        return device
    
    def _discover_models(self) -> Dict[str, ModelInfo]:
        """Découvre les modèles disponibles"""
        models = {}
        
        # Modèles définis dans la configuration
        model_configs = {
            'stable_epoch_30': {
                'file': settings.DEFAULT_MODEL,
                'description': 'Modèle champion stable (Epoch 30)',
                'performance': 'high',
                'speed': 'medium'
            },
            'extended_28_classes': {
                'file': settings.EXTENDED_MODEL,
                'description': 'Modèle étendu 28 classes',
                'performance': 'very_high',
                'speed': 'medium'
            },
            'fast_stream': {
                'file': settings.FAST_MODEL,
                'description': 'Modèle rapide pour streaming',
                'performance': 'medium',
                'speed': 'very_high'
            }
        }
        
        for name, config in model_configs.items():
            model_path = settings.MODELS_DIR / config['file']
            if model_path.exists():
                models[name] = ModelInfo(name, model_path, config)
                logger.info(f"Modèle trouvé: {name} -> {model_path}")
            else:
                logger.warning(f"Modèle non trouvé: {name} -> {model_path}")
        
        return models
    
    async def initialize(self):
        """Initialise le gestionnaire"""
        # Précharger le modèle par défaut
        if self.default_model.replace('.pth', '') in self.available_models:
            model_name = self.default_model.replace('.pth', '')
            await self.load_model(model_name)
            logger.info(f"Modèle par défaut chargé: {model_name}")
    
    async def load_model(self, model_name: str) -> ModelInfo:
        """Charge un modèle en mémoire"""
        # Vérifier le cache
        model_info = self.models_cache.get(model_name)
        if model_info and model_info.is_loaded:
            self.stats['cache_hits'] += 1
            model_info.last_used = datetime.now()
            logger.debug(f"Modèle {model_name} trouvé dans le cache")
            return model_info
        
        self.stats['cache_misses'] += 1
        
        # Charger le modèle
        if model_name not in self.available_models:
            raise ValueError(f"Modèle non disponible: {model_name}")
        
        model_info = self.available_models[model_name]
        
        logger.info(f"Chargement du modèle {model_name}...")
        start_time = datetime.now()
        
        try:
            # Créer le modèle
            model = SimpleDetectionModel(
                num_classes=MODEL_CONFIG['num_classes'],
                input_size=MODEL_CONFIG['image_size'][0]
            )
            
            # Charger les poids si le fichier existe
            if model_info.path.exists():
                try:
                    checkpoint = torch.load(model_info.path, map_location=self.device)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    logger.info(f"Poids chargés depuis {model_info.path}")
                except Exception as e:
                    logger.warning(f"Impossible de charger les poids: {e}. Utilisation d'un modèle aléatoire.")
            else:
                logger.warning(f"Fichier modèle non trouvé: {model_info.path}. Utilisation d'un modèle aléatoire.")
            
            # Déplacer sur le device et mettre en mode évaluation
            model = model.to(self.device)
            model.eval()
            
            # Créer le détecteur
            detector = ObjectDetector(model, self.device)
            
            # Mettre à jour les informations
            model_info.model = model
            model_info.detector = detector
            model_info.is_loaded = True
            model_info.load_time = datetime.now()
            model_info.last_used = datetime.now()
            model_info.memory_usage = self._get_model_memory_usage(model)
            
            # Ajouter au cache
            self.models_cache.put(model_name, model_info)
            
            load_duration = (datetime.now() - start_time).total_seconds()
            self.stats['total_loads'] += 1
            
            logger.info(f"Modèle {model_name} chargé en {load_duration:.2f}s "
                       f"(Mémoire: {model_info.memory_usage:.1f} MB)")
            
            return model_info
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle {model_name}: {e}")
            raise
    
    async def get_detector(self, model_name: Optional[str] = None) -> ObjectDetector:
        """Récupère un détecteur pour un modèle"""
        if model_name is None:
            model_name = list(self.available_models.keys())[0]  # Premier modèle disponible
        
        model_info = await self.load_model(model_name)
        return model_info.detector
    
    def _get_model_memory_usage(self, model: nn.Module) -> float:
        """Calcule l'usage mémoire d'un modèle en MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / 1024 / 1024  # Conversion en MB
    
    async def get_health_status(self) -> dict:
        """Retourne l'état de santé du gestionnaire"""
        memory = psutil.virtual_memory()
        gpu_memory = {}
        
        if torch.cuda.is_available():
            gpu_memory = {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'cached': torch.cuda.memory_reserved() / 1e9,
                'total': torch.cuda.get_device_properties(0).total_memory / 1e9
            }
        
        loaded_models = [name for name, info in self.models_cache.cache.items() 
                        if info.is_loaded]
        
        return {
            'timestamp': datetime.now(),
            'models_loaded': loaded_models,
            'gpu_available': torch.cuda.is_available(),
            'memory_usage': {
                'ram_percent': memory.percent,
                'ram_available_gb': memory.available / 1e9,
                'gpu': gpu_memory
            },
            'cache_stats': self.stats,
            'device': str(self.device)
        }
    
    async def get_stats(self) -> dict:
        """Retourne les statistiques détaillées"""
        models_stats = {}
        
        for name, info in self.available_models.items():
            models_stats[name] = {
                'is_loaded': info.is_loaded,
                'memory_usage_mb': info.memory_usage,
                'inference_count': info.inference_count,
                'avg_inference_time': (info.total_inference_time / max(1, info.inference_count)),
                'last_used': info.last_used.isoformat() if info.last_used else None,
                'load_time': info.load_time.isoformat() if info.load_time else None
            }
        
        return {
            'general_stats': self.stats,
            'models_performance': models_stats,
            'resource_usage': (await self.get_health_status())['memory_usage'],
            'available_models': list(self.available_models.keys())
        }
    
    async def cleanup(self):
        """Nettoie les ressources"""
        logger.info("Nettoyage du ModelManager...")
        
        # Décharger tous les modèles
        for model_info in self.models_cache.cache.values():
            if model_info.is_loaded:
                self.models_cache._unload_model(model_info)
        
        # Nettoyer la mémoire GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("Nettoyage terminé")
    
    def update_inference_stats(self, model_name: str, inference_time: float):
        """Met à jour les statistiques d'inférence"""
        if model_name in self.available_models:
            info = self.available_models[model_name]
            info.inference_count += 1
            info.total_inference_time += inference_time
            info.last_used = datetime.now()
        
        self.stats['total_inferences'] += 1