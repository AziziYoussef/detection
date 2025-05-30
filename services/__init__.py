"""
🎯 SERVICES DE DÉTECTION - MODULE D'INITIALISATION
=================================================
Point d'entrée pour tous les services de détection d'objets perdus

Services disponibles:
- ImageService: Traitement d'images statiques (JPG, PNG, etc.)
- VideoService: Traitement de vidéos uploadées (MP4, AVI, etc.)
- StreamService: Streaming temps réel (WebSocket, webcam)
- BatchService: Traitement batch de multiples fichiers
- ModelService: Gestion centralisée des modèles (déjà implémenté)

Architecture:
- Services indépendants mais interconnectés
- Interface unifiée via ServiceManager
- Gestion centralisée des ressources
- Monitoring et métriques intégrés
"""

from .image_service import ImageService, ImageProcessingConfig
from .video_service import VideoService, VideoProcessingConfig
from .stream_service import StreamService, StreamingConfig
from .batch_service import BatchService, BatchProcessingConfig
from .model_service import ModelService, ModelMetrics, ModelPriority, PerformanceProfile

import logging
from typing import Dict, Any, Optional
from app.core.model_manager import ModelManager

logger = logging.getLogger(__name__)

class ServiceManager:
    """🎯 Gestionnaire centralisé de tous les services de détection"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self._services: Dict[str, Any] = {}
        self._initialized = False
        
        logger.info("🎯 ServiceManager initialisé")
    
    async def initialize(self):
        """🚀 Initialise tous les services"""
        
        if self._initialized:
            logger.warning("⚠️ Services déjà initialisés")
            return
        
        logger.info("🚀 Initialisation des services de détection...")
        
        try:
            # 1. Service de gestion des modèles (central)
            self._services["model"] = ModelService(self.model_manager)
            await self._services["model"].initialize()
            
            # 2. Service d'images statiques
            self._services["image"] = ImageService(self._services["model"])
            await self._services["image"].initialize()
            
            # 3. Service de vidéos
            self._services["video"] = VideoService(self._services["model"])
            await self._services["video"].initialize()
            
            # 4. Service de streaming temps réel
            self._services["stream"] = StreamService(self._services["model"])
            await self._services["stream"].initialize()
            
            # 5. Service de traitement batch
            self._services["batch"] = BatchService(self._services["model"])
            await self._services["batch"].initialize()
            
            self._initialized = True
            logger.info("✅ Tous les services initialisés avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation des services: {e}")
            raise
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """📋 Récupère un service par nom"""
        if not self._initialized:
            raise RuntimeError("Services non initialisés - appelez initialize() d'abord")
        
        return self._services.get(service_name)
    
    @property
    def image_service(self) -> ImageService:
        """📸 Service d'images statiques"""
        return self.get_service("image")
    
    @property
    def video_service(self) -> VideoService:
        """🎬 Service de vidéos"""
        return self.get_service("video")
    
    @property
    def stream_service(self) -> StreamService:
        """📡 Service de streaming"""
        return self.get_service("stream")
    
    @property
    def batch_service(self) -> BatchService:
        """📦 Service batch"""
        return self.get_service("batch")
    
    @property
    def model_service(self) -> ModelService:
        """🤖 Service de modèles"""
        return self.get_service("model")
    
    async def get_global_statistics(self) -> Dict[str, Any]:
        """📊 Statistiques globales de tous les services"""
        
        if not self._initialized:
            return {"error": "Services non initialisés"}
        
        stats = {
            "services_initialized": len(self._services),
            "services_available": list(self._services.keys()),
            "global_metrics": {}
        }
        
        # Collecter les statistiques de chaque service
        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'get_service_statistics'):
                    service_stats = await service.get_service_statistics()
                    stats["global_metrics"][service_name] = service_stats
                elif hasattr(service, 'get_statistics'):
                    service_stats = service.get_statistics()
                    stats["global_metrics"][service_name] = service_stats
            except Exception as e:
                logger.warning(f"⚠️ Erreur collecte stats {service_name}: {e}")
                stats["global_metrics"][service_name] = {"error": str(e)}
        
        return stats
    
    async def cleanup(self):
        """🧹 Nettoyage de tous les services"""
        
        logger.info("🧹 Nettoyage des services...")
        
        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'cleanup'):
                    await service.cleanup()
                    logger.debug(f"✅ Service {service_name} nettoyé")
            except Exception as e:
                logger.warning(f"⚠️ Erreur nettoyage {service_name}: {e}")
        
        self._services.clear()
        self._initialized = False
        
        logger.info("✅ Nettoyage des services terminé")

# 🏭 FACTORY FUNCTIONS
async def create_service_manager(model_manager: ModelManager) -> ServiceManager:
    """🏭 Factory pour créer et initialiser le gestionnaire de services"""
    
    manager = ServiceManager(model_manager)
    await manager.initialize()
    return manager

def get_available_services() -> Dict[str, str]:
    """📋 Liste des services disponibles"""
    
    return {
        "image": "Service de traitement d'images statiques",
        "video": "Service de traitement de vidéos uploadées", 
        "stream": "Service de streaming temps réel",
        "batch": "Service de traitement batch",
        "model": "Service de gestion des modèles"
    }

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    # Services principaux
    "ImageService",
    "VideoService", 
    "StreamService",
    "BatchService",
    "ModelService",
    
    # Configurations
    "ImageProcessingConfig",
    "VideoProcessingConfig",
    "StreamingConfig", 
    "BatchProcessingConfig",
    
    # Gestionnaire
    "ServiceManager",
    "create_service_manager",
    
    # Utilitaires
    "get_available_services",
    
    # Types et énumérations
    "ModelMetrics",
    "ModelPriority",
    "PerformanceProfile"
]