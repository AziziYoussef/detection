"""
🧠 CORE PACKAGE - COMPOSANTS PRINCIPAUX DE DÉTECTION
===================================================
Package central contenant tous les composants essentiels pour la détection d'objets perdus

Ce package regroupe:
- Detector: Classe principale de détection utilisant vos modèles PyTorch
- ModelManager: Gestionnaire intelligent des modèles (chargement, cache, optimisation)
- Preprocessing: Prétraitement optimisé des images et vidéos
- Postprocessing: Post-traitement des résultats de détection

Architecture:
- Modèles PyTorch optimisés (Epoch 30: F1=49.86%, Précision=60.73%)
- Support GPU/CPU automatique avec fallback
- Cache intelligent des modèles pour performance
- Pipeline de traitement optimisé pour temps réel

Intégration:
- FastAPI endpoints utilisent ces composants
- WebSocket streaming utilise les optimisations temps réel
- Spring Boot reçoit les résultats structurés
"""

from .detector import ObjectDetector, DetectionEngine
from .model_manager import ModelManager, ModelCache, ModelInfo
from .preprocessing import ImagePreprocessor, VideoPreprocessor, PreprocessingPipeline
from .postprocessing import ResultPostprocessor, DetectionFilter, ResultAggregator

__all__ = [
    # Détection principale
    "ObjectDetector",
    "DetectionEngine",
    
    # Gestion des modèles
    "ModelManager", 
    "ModelCache",
    "ModelInfo",
    
    # Prétraitement
    "ImagePreprocessor",
    "VideoPreprocessor", 
    "PreprocessingPipeline",
    
    # Post-traitement
    "ResultPostprocessor",
    "DetectionFilter",
    "ResultAggregator"
]