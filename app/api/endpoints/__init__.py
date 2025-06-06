"""
🎯 ENDPOINTS PACKAGE - SERVICES DE DÉTECTION SPÉCIALISÉS
=========================================================
Package contenant tous les endpoints spécialisés par type de détection

Services disponibles:
📸 IMAGE_DETECTION - Détection sur images statiques uploadées
🎬 VIDEO_DETECTION - Traitement complet de vidéos uploadées  
📡 STREAM_DETECTION - Détection temps réel via WebSocket
📦 BATCH_DETECTION - Traitement en lot de multiples fichiers

Chaque service utilise vos modèles PyTorch optimisés:
- Epoch 30 (Champion: F1=49.86%, Précision=60.73%)
- Extended (28 classes d'objets perdus)
- Streaming optimisé pour temps réel

Architecture:
- FastAPI routers spécialisés
- Schémas Pydantic pour validation
- Gestion asynchrone des tâches longues
- Intégration Spring Boot + Next.js
"""

# Imports des routers pour faciliter l'utilisation
from .image_detection import router as image_router
from .video_detection import router as video_router
from .stream_detection import router as stream_router
from .batch_detection import router as batch_router

__all__ = [
    "image_router",
    "video_router", 
    "stream_router",
    "batch_router"
]