"""
📡 STREAM SERVICE - SERVICE DE STREAMING TEMPS RÉEL
===================================================
Service spécialisé pour la détection d'objets perdus en streaming temps réel

Fonctionnalités:
- WebSocket bidirectionnel pour communication temps réel
- Traitement de flux webcam/caméra en direct
- Optimisations ultra-rapides (< 100ms par frame)
- Qualité adaptative selon la bande passante
- Détection de mouvement pour optimisation
- Buffer intelligent et gestion de latence
- Streaming multi-clients simultanés
- Alertes temps réel pour objets détectés

Optimisations temps réel:
- Modèle "fast" par défaut (< 50ms)
- Résolution adaptative (320p à 720p)
- Skip frames si surcharge
- Compression intelligente des résultats
- Préchargement et cache agressif
"""

import asyncio
import time
import logging
import json
import uuid
import base64
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque
import weakref

import numpy as np
import cv2
from PIL import Image
import websockets
from websockets.exceptions import ConnectionClosed

# Imports internes
from app.schemas.detection import DetectionResult, StreamFrame, StreamSession
from app.config.config import get_settings
from .model_service import ModelService, PerformanceProfile
from .image_service import ImageService, ImageProcessingConfig

logger = logging.getLogger(__name__)

# 📋 ÉNUMÉRATIONS ET TYPES
class StreamQuality(str, Enum):
    """📊 Qualités de streaming"""
    ULTRA_FAST = "ultra_fast"    # 320p, modèle nano
    FAST = "fast"                # 480p, modèle fast
    BALANCED = "balanced"        # 720p, modèle balanced
    HIGH = "high"                # 1080p, modèle standard

class StreamState(str, Enum):
    """🔄 États de streaming"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    PAUSED = "paused"
    ERROR = "error"
    DISCONNECTED = "disconnected"

class FrameSkipStrategy(str, Enum):
    """⏭️ Stratégies de skip de frames"""
    NONE = "none"                # Pas de skip
    ADAPTIVE = "adaptive"        # Basé sur latence
    MOTION_BASED = "motion_based" # Basé sur mouvement
    LOAD_BASED = "load_based"    # Basé sur charge système

@dataclass
class StreamingConfig:
    """⚙️ Configuration du streaming temps réel"""
    
    # 🔧 Paramètres de base
    quality: StreamQuality = StreamQuality.FAST
    target_fps: float = 15.0  # FPS cible pour streaming
    max_latency_ms: float = 200.0  # Latence maximale acceptable
    
    # 📊 Optimisations
    frame_skip_strategy: FrameSkipStrategy = FrameSkipStrategy.ADAPTIVE
    motion_detection: bool = True
    adaptive_quality: bool = True  # Ajuste qualité selon performances
    
    # 🎯 Détection
    confidence_threshold: float = 0.5
    fast_mode: bool = True  # Mode ultra-rapide
    min_detection_size: int = 30  # Taille minimale objets (pixels)
    
    # 📡 WebSocket
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    ping_interval: float = 30.0
    ping_timeout: float = 60.0
    compression_enabled: bool = True
    
    # 💾 Buffer et cache
    frame_buffer_size: int = 5
    result_buffer_size: int = 10
    enable_caching: bool = False  # Pas de cache en streaming
    
    # 📸 Formats d'image
    input_format: str = "jpeg"  # Format frames reçues
    compression_quality: int = 85  # Qualité JPEG

@dataclass
class StreamClient:
    """👤 Client de streaming connecté"""
    client_id: str
    websocket: Any  # WebSocket connection
    config: StreamingConfig
    state: StreamState = StreamState.CONNECTING
    
    # Statistiques
    frames_received: int = 0
    frames_processed: int = 0
    frames_skipped: int = 0
    total_detections: int = 0
    avg_latency_ms: float = 0.0
    connection_time: float = field(default_factory=time.time)
    
    # Buffers
    frame_buffer: deque = field(default_factory=lambda: deque(maxlen=5))
    result_buffer: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Performance tracking
    processing_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update_performance(self, processing_time_ms: float):
        """📊 Met à jour les métriques de performance"""
        self.processing_times.append(processing_time_ms)
        if self.processing_times:
            self.avg_latency_ms = sum(self.processing_times) / len(self.processing_times)

# 📡 SERVICE PRINCIPAL
class StreamService:
    """📡 Service de streaming temps réel"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.settings = get_settings()
        
        # Configuration par défaut
        self.config = StreamingConfig()
        
        # Service d'images optimisé pour streaming
        self.image_service = ImageService(model_service)
        
        # Clients connectés
        self._clients: Dict[str, StreamClient] = {}
        self._client_locks: Dict[str, asyncio.Lock] = {}
        
        # Statistiques globales
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "total_frames_processed": 0,
            "total_detections": 0,
            "avg_processing_time_ms": 0.0,
            "connection_errors": 0,
            "start_time": time.time()
        }
        
        # Détection de mouvement (partagée entre clients)
        self._motion_detectors: Dict[str, Any] = {}
        
        # Thread pool pour traitement parallèle
        self._processing_executor = None
        
        logger.info("📡 StreamService initialisé")
    
    async def initialize(self):
        """🚀 Initialise le service de streaming"""
        logger.info("🚀 Initialisation StreamService...")
        
        # Initialiser le service d'images
        await self.image_service.initialize()
        
        # Initialiser thread pool
        import concurrent.futures
        self._processing_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="stream_processor"
        )
        
        logger.info("✅ StreamService initialisé")
    
    async def connect_client(
        self,
        websocket,
        client_id: str,
        config: Optional[StreamingConfig] = None
    ) -> bool:
        """
        🔌 Connecte un nouveau client de streaming
        
        Args:
            websocket: Connexion WebSocket
            client_id: ID unique du client
            config: Configuration personnalisée
            
        Returns:
            True si connexion réussie
        """
        
        try:
            config = config or self.config
            
            # Créer le client
            client = StreamClient(
                client_id=client_id,
                websocket=websocket,
                config=config,
                state=StreamState.CONNECTING
            )
            
            # Ajouter aux clients connectés
            self._clients[client_id] = client
            self._client_locks[client_id] = asyncio.Lock()
            
            # Créer détecteur de mouvement si nécessaire
            if config.motion_detection:
                self._motion_detectors[client_id] = MotionDetector()
            
            # Marquer comme connecté
            client.state = StreamState.CONNECTED
            
            # Statistiques
            self.stats["total_connections"] += 1
            self.stats["active_connections"] = len(self._clients)
            
            # Message de bienvenue
            await self._send_to_client(client_id, {
                "type": "connection_established",
                "client_id": client_id,
                "config": config.__dict__,
                "timestamp": time.time()
            })
            
            logger.info(f"📡 Client connecté: {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion client {client_id}: {e}")
            self.stats["connection_errors"] += 1
            return False
    
    async def disconnect_client(self, client_id: str):
        """🔌 Déconnecte un client"""
        
        if client_id in self._clients:
            client = self._clients[client_id]
            client.state = StreamState.DISCONNECTED
            
            # Nettoyer les ressources
            del self._clients[client_id]
            
            if client_id in self._client_locks:
                del self._client_locks[client_id]
            
            if client_id in self._motion_detectors:
                del self._motion_detectors[client_id]
            
            # Statistiques
            self.stats["active_connections"] = len(self._clients)
            
            logger.info(f"📡 Client déconnecté: {client_id}")
    
    async def process_frame(
        self,
        client_id: str,
        frame_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        🖼️ Traite une frame de streaming
        
        Args:
            client_id: ID du client
            frame_data: Données de la frame
            
        Returns:
            Résultat de détection ou None
        """
        
        if client_id not in self._clients:
            logger.warning(f"⚠️ Client inconnu: {client_id}")
            return None
        
        client = self._clients[client_id]
        start_time = time.time()
        
        try:
            async with self._client_locks[client_id]:
                client.frames_received += 1
                
                # 1. Décoder la frame
                frame_image = await self._decode_frame(frame_data, client.config)
                if frame_image is None:
                    return None
                
                # 2. Vérifier si on doit skip cette frame
                should_skip = await self._should_skip_frame(client_id, frame_image, client.config)
                if should_skip:
                    client.frames_skipped += 1
                    return {"type": "frame_skipped", "reason": "optimization"}
                
                # 3. Traitement de détection ultra-rapide
                detections = await self._process_frame_fast(
                    frame_image, client.config, f"{client_id}_{client.frames_received}"
                )
                
                # 4. Post-traitement temps réel
                filtered_detections = await self._postprocess_stream_detections(
                    detections, client.config
                )
                
                # 5. Mise à jour des statistiques client
                processing_time_ms = (time.time() - start_time) * 1000
                client.update_performance(processing_time_ms)
                client.frames_processed += 1
                client.total_detections += len(filtered_detections)
                
                # 6. Adaptation qualité si nécessaire
                if client.config.adaptive_quality:
                    await self._adapt_quality(client_id, processing_time_ms)
                
                # 7. Création du résultat
                result = {
                    "type": "detection_result",
                    "client_id": client_id,
                    "frame_id": client.frames_received,
                    "timestamp": time.time(),
                    "detections": [
                        {
                            "class_name": d.class_name,
                            "class_name_fr": d.class_name_fr,
                            "confidence": round(d.confidence, 3),
                            "bbox": {
                                "x1": d.bbox.x1, "y1": d.bbox.y1,
                                "x2": d.bbox.x2, "y2": d.bbox.y2
                            }
                        }
                        for d in filtered_detections
                    ],
                    "processing_time_ms": round(processing_time_ms, 2),
                    "total_objects": len(filtered_detections)
                }
                
                # 8. Mise à jour stats globales
                self.stats["total_frames_processed"] += 1
                self.stats["total_detections"] += len(filtered_detections)
                self._update_global_stats(processing_time_ms)
                
                return result
                
        except Exception as e:
            logger.error(f"❌ Erreur traitement frame {client_id}: {e}")
            return {
                "type": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _decode_frame(
        self,
        frame_data: Dict[str, Any],
        config: StreamingConfig
    ) -> Optional[Image.Image]:
        """🔍 Décode les données de frame reçues"""
        
        try:
            if "image_data" in frame_data:
                # Base64 encoded image
                image_bytes = base64.b64decode(frame_data["image_data"])
                
                if config.input_format.lower() in ["jpeg", "jpg"]:
                    # JPEG directement
                    from io import BytesIO
                    image = Image.open(BytesIO(image_bytes))
                else:
                    # Autres formats
                    image = Image.open(BytesIO(image_bytes))
                
                return image
                
            elif "frame_array" in frame_data:
                # Array NumPy
                frame_array = np.array(frame_data["frame_array"], dtype=np.uint8)
                
                if len(frame_array.shape) == 3:
                    return Image.fromarray(frame_array, 'RGB')
                elif len(frame_array.shape) == 2:
                    return Image.fromarray(frame_array, 'L')
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur décodage frame: {e}")
            return None
    
    async def _should_skip_frame(
        self,
        client_id: str,
        frame: Image.Image,
        config: StreamingConfig
    ) -> bool:
        """⏭️ Détermine si on doit skip cette frame"""
        
        if config.frame_skip_strategy == FrameSkipStrategy.NONE:
            return False
        
        client = self._clients[client_id]
        
        # Skip basé sur latence
        if config.frame_skip_strategy == FrameSkipStrategy.ADAPTIVE:
            if client.avg_latency_ms > config.max_latency_ms:
                return True
        
        # Skip basé sur mouvement
        elif config.frame_skip_strategy == FrameSkipStrategy.MOTION_BASED:
            if client_id in self._motion_detectors:
                motion_detector = self._motion_detectors[client_id]
                has_motion = motion_detector.detect_motion(frame)
                if not has_motion:
                    return True
        
        # Skip basé sur charge
        elif config.frame_skip_strategy == FrameSkipStrategy.LOAD_BASED:
            # Vérifier charge CPU/GPU
            if len(self._clients) > 10:  # Beaucoup de clients
                return client.frames_received % 2 == 0  # Skip 1 frame sur 2
        
        return False
    
    async def _process_frame_fast(
        self,
        image: Image.Image,
        config: StreamingConfig,
        processing_id: str
    ) -> List[DetectionResult]:
        """⚡ Traitement ultra-rapide d'une frame"""
        
        # Configuration optimisée pour vitesse
        image_config = ImageProcessingConfig(
            confidence_threshold=config.confidence_threshold,
            quality="fast",  # Mode le plus rapide
            target_size=(320, 320) if config.quality == StreamQuality.ULTRA_FAST else (480, 480),
            resize_strategy="resize_aspect",
            auto_enhance=False,  # Pas d'amélioration (trop lent)
            save_results=False,
            save_annotated_images=False,
            cache_results=False,  # Pas de cache en streaming
            auto_model_selection=False,
            preferred_model="fast"  # Modèle le plus rapide
        )
        
        # Traitement avec le service d'images
        result = await self.image_service.process_image(
            image_input=image,
            config=image_config,
            processing_id=processing_id
        )
        
        return result.detections
    
    async def _postprocess_stream_detections(
        self,
        detections: List[DetectionResult],
        config: StreamingConfig
    ) -> List[DetectionResult]:
        """🔄 Post-traitement optimisé pour streaming"""
        
        if not detections:
            return []
        
        # Filtrage par taille minimale
        min_area = config.min_detection_size * config.min_detection_size
        filtered = []
        
        for detection in detections:
            bbox_area = (detection.bbox.x2 - detection.bbox.x1) * (detection.bbox.y2 - detection.bbox.y1)
            if bbox_area >= min_area:
                filtered.append(detection)
        
        # Tri par confiance (garder seulement les meilleurs)
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limiter le nombre pour éviter surcharge réseau
        return filtered[:20]  # Max 20 détections par frame
    
    async def _adapt_quality(self, client_id: str, processing_time_ms: float):
        """📊 Adapte la qualité selon les performances"""
        
        client = self._clients[client_id]
        config = client.config
        
        # Seuils de latence pour adaptation
        if processing_time_ms > config.max_latency_ms * 1.5:
            # Trop lent - réduire qualité
            if config.quality == StreamQuality.HIGH:
                config.quality = StreamQuality.BALANCED
            elif config.quality == StreamQuality.BALANCED:
                config.quality = StreamQuality.FAST
            elif config.quality == StreamQuality.FAST:
                config.quality = StreamQuality.ULTRA_FAST
            
            logger.debug(f"🔽 Qualité réduite pour {client_id}: {config.quality.value}")
            
        elif processing_time_ms < config.max_latency_ms * 0.5:
            # Rapide - peut augmenter qualité
            if config.quality == StreamQuality.ULTRA_FAST:
                config.quality = StreamQuality.FAST
            elif config.quality == StreamQuality.FAST:
                config.quality = StreamQuality.BALANCED
            
            logger.debug(f"🔼 Qualité augmentée pour {client_id}: {config.quality.value}")
    
    async def _send_to_client(self, client_id: str, message: Dict[str, Any]):
        """📤 Envoie un message à un client"""
        
        if client_id not in self._clients:
            return
        
        client = self._clients[client_id]
        
        try:
            # Compression si activée
            if client.config.compression_enabled:
                message_str = json.dumps(message, separators=(',', ':'))
            else:
                message_str = json.dumps(message, indent=None)
            
            await client.websocket.send(message_str)
            
        except ConnectionClosed:
            logger.info(f"🔌 Connexion fermée: {client_id}")
            await self.disconnect_client(client_id)
        except Exception as e:
            logger.error(f"❌ Erreur envoi message à {client_id}: {e}")
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """📡 Diffuse un message à tous les clients connectés"""
        
        if not self._clients:
            return
        
        # Envoyer en parallèle à tous les clients
        tasks = [
            self._send_to_client(client_id, message)
            for client_id in self._clients.keys()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """👤 Retourne les informations d'un client"""
        
        if client_id not in self._clients:
            return None
        
        client = self._clients[client_id]
        
        return {
            "client_id": client.client_id,
            "state": client.state.value,
            "connection_time": client.connection_time,
            "uptime_seconds": time.time() - client.connection_time,
            "frames_received": client.frames_received,
            "frames_processed": client.frames_processed,
            "frames_skipped": client.frames_skipped,
            "total_detections": client.total_detections,
            "avg_latency_ms": client.avg_latency_ms,
            "config": client.config.__dict__
        }
    
    def get_all_clients_info(self) -> List[Dict[str, Any]]:
        """👥 Retourne les informations de tous les clients"""
        
        return [
            self.get_client_info(client_id)
            for client_id in self._clients.keys()
        ]
    
    def _update_global_stats(self, processing_time_ms: float):
        """📊 Met à jour les statistiques globales"""
        
        # Moyenne mobile pour temps de traitement
        if self.stats["avg_processing_time_ms"] == 0:
            self.stats["avg_processing_time_ms"] = processing_time_ms
        else:
            # Moyenne mobile avec factor 0.1
            self.stats["avg_processing_time_ms"] = (
                0.9 * self.stats["avg_processing_time_ms"] + 
                0.1 * processing_time_ms
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """📊 Retourne les statistiques du service"""
        
        uptime_hours = (time.time() - self.stats["start_time"]) / 3600
        
        return {
            "service_type": "streaming",
            "uptime_hours": uptime_hours,
            "connections": {
                "total_connections": self.stats["total_connections"],
                "active_connections": self.stats["active_connections"],
                "connection_errors": self.stats["connection_errors"]
            },
            "performance": {
                "total_frames_processed": self.stats["total_frames_processed"],
                "total_detections": self.stats["total_detections"],
                "avg_processing_time_ms": self.stats["avg_processing_time_ms"],
                "frames_per_hour": self.stats["total_frames_processed"] / uptime_hours if uptime_hours > 0 else 0,
                "detections_per_hour": self.stats["total_detections"] / uptime_hours if uptime_hours > 0 else 0
            },
            "clients": self.get_all_clients_info()
        }
    
    async def cleanup(self):
        """🧹 Nettoyage des ressources"""
        logger.info("🧹 Nettoyage StreamService...")
        
        # Déconnecter tous les clients
        client_ids = list(self._clients.keys())
        for client_id in client_ids:
            await self.disconnect_client(client_id)
        
        # Nettoyer détecteurs de mouvement
        self._motion_detectors.clear()
        
        # Fermer thread pool
        if self._processing_executor:
            self._processing_executor.shutdown(wait=True)
        
        # Nettoyer service d'images
        await self.image_service.cleanup()
        
        logger.info("✅ StreamService nettoyé")

# 🏃 DÉTECTEUR DE MOUVEMENT
class MotionDetector:
    """🏃 Détecteur de mouvement pour optimisation streaming"""
    
    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold
        self.prev_frame = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False
        )
    
    def detect_motion(self, image: Image.Image) -> bool:
        """🔍 Détecte le mouvement dans une frame"""
        
        try:
            # Conversion en array OpenCV
            frame = np.array(image)
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Redimensionner pour performance
            frame = cv2.resize(frame, (160, 120))
            
            # Détection via background subtractor
            fg_mask = self.background_subtractor.apply(frame)
            
            # Pourcentage de pixels en mouvement
            motion_ratio = np.count_nonzero(fg_mask) / fg_mask.size
            
            return motion_ratio > self.threshold
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur détection mouvement: {e}")
            return True  # Supposer mouvement en cas d'erreur

# 🌐 GESTIONNAIRE WEBSOCKET
class WebSocketManager:
    """🌐 Gestionnaire des connexions WebSocket pour streaming"""
    
    def __init__(self, stream_service: StreamService):
        self.stream_service = stream_service
        self._connections: Dict[str, Any] = {}
        
    async def connect(self, websocket, client_id: str):
        """🔌 Nouvelle connexion WebSocket"""
        
        self._connections[client_id] = websocket
        
        # Connecter au service de streaming
        success = await self.stream_service.connect_client(
            websocket, client_id
        )
        
        if not success:
            await websocket.close()
            return
        
        logger.info(f"🌐 WebSocket connecté: {client_id}")
    
    async def disconnect(self, client_id: str):
        """🔌 Déconnexion WebSocket"""
        
        if client_id in self._connections:
            del self._connections[client_id]
        
        await self.stream_service.disconnect_client(client_id)
        logger.info(f"🌐 WebSocket déconnecté: {client_id}")
    
    async def handle_message(self, client_id: str, message_data: Dict[str, Any]):
        """📨 Traite un message WebSocket"""
        
        try:
            message_type = message_data.get("type", "unknown")
            
            if message_type == "frame":
                # Frame de streaming à traiter
                result = await self.stream_service.process_frame(
                    client_id, message_data
                )
                
                if result:
                    # Renvoyer le résultat
                    await self._send_to_client(client_id, result)
            
            elif message_type == "config_update":
                # Mise à jour configuration
                await self._update_client_config(client_id, message_data.get("config", {}))
            
            elif message_type == "ping":
                # Ping/Pong pour keep-alive
                await self._send_to_client(client_id, {"type": "pong", "timestamp": time.time()})
            
            else:
                logger.warning(f"⚠️ Type de message inconnu: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ Erreur traitement message WebSocket {client_id}: {e}")
            await self._send_to_client(client_id, {
                "type": "error",
                "error": str(e),
                "timestamp": time.time()
            })
    
    async def _send_to_client(self, client_id: str, message: Dict[str, Any]):
        """📤 Envoie un message via WebSocket"""
        
        if client_id in self._connections:
            try:
                websocket = self._connections[client_id]
                await websocket.send(json.dumps(message))
            except Exception as e:
                logger.error(f"❌ Erreur envoi WebSocket {client_id}: {e}")
                await self.disconnect(client_id)
    
    async def _update_client_config(self, client_id: str, config_updates: Dict[str, Any]):
        """⚙️ Met à jour la configuration d'un client"""
        
        if client_id in self.stream_service._clients:
            client = self.stream_service._clients[client_id]
            
            # Mettre à jour les paramètres autorisés
            allowed_updates = [
                "quality", "confidence_threshold", "adaptive_quality",
                "motion_detection", "compression_enabled"
            ]
            
            for key, value in config_updates.items():
                if key in allowed_updates and hasattr(client.config, key):
                    setattr(client.config, key, value)
            
            # Confirmer la mise à jour
            await self._send_to_client(client_id, {
                "type": "config_updated",
                "config": client.config.__dict__,
                "timestamp": time.time()
            })
    
    async def cleanup(self):
        """🧹 Nettoyage du gestionnaire WebSocket"""
        
        # Fermer toutes les connexions
        for client_id in list(self._connections.keys()):
            await self.disconnect(client_id)
        
        logger.info("✅ WebSocketManager nettoyé")

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "StreamService",
    "StreamingConfig",
    "StreamQuality",
    "StreamState",
    "FrameSkipStrategy",
    "StreamClient",
    "MotionDetector",
    "WebSocketManager"
]