"""
📡 WEBSOCKET STREAM HANDLER - GESTIONNAIRE TEMPS RÉEL
===================================================
Gestionnaire centralisé pour toutes les connexions WebSocket de streaming

Fonctionnalités:
- Gestion de connexions WebSocket multiples simultanées
- Routage des messages par type et client
- Diffusion de résultats de détection en temps réel
- Monitoring et statistiques des connexions
- Gestion automatique des déconnexions et erreurs
- Queue de messages pour optimiser les performances

Architecture:
- WebSocketManager: Gestionnaire principal
- ConnectionManager: Gestion des connexions individuelles  
- StreamHandler: Traitement des streams de détection
- MessageRouter: Routage intelligent des messages
"""

import asyncio
import time
import json
import uuid
import logging
from typing import Dict, List, Optional, Any, Set, Callable, Union
from enum import Enum
from collections import defaultdict, deque
from dataclasses import dataclass, field
import weakref

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import aiofiles

# Services internes
from app.services.model_service import ModelService
from app.schemas.detection import DetectionResult
from app.config.config import get_settings

logger = logging.getLogger(__name__)

# 📋 ÉNUMÉRATIONS ET TYPES
class ConnectionStatus(str, Enum):
    """🔌 Statuts de connexion"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    PAUSED = "paused"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"

class MessageType(str, Enum):
    """📨 Types de messages WebSocket"""
    # Client → Serveur
    CONNECT = "connect"
    FRAME = "frame"
    COMMAND = "command"
    PING = "ping"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    
    # Serveur → Client
    CONNECTION_ACK = "connection_ack"
    DETECTIONS = "detections"
    STATISTICS = "statistics"
    NOTIFICATION = "notification"
    ERROR = "error"
    PONG = "pong"
    STATUS_UPDATE = "status_update"

class StreamQuality(str, Enum):
    """🎚️ Qualités de stream"""
    ULTRA_LOW = "ultra_low"    # 160x120
    LOW = "low"                # 320x240
    MEDIUM = "medium"          # 640x480
    HIGH = "high"              # 1280x720
    ULTRA_HIGH = "ultra_high"  # 1920x1080

# 📋 MODÈLES DE DONNÉES
@dataclass
class WebSocketConnection:
    """🔌 Représentation d'une connexion WebSocket"""
    client_id: str
    websocket: WebSocket
    status: ConnectionStatus = ConnectionStatus.CONNECTING
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    # Configuration client
    model_name: str = "epoch_30"
    confidence_threshold: float = 0.5
    stream_quality: StreamQuality = StreamQuality.MEDIUM
    subscriptions: Set[str] = field(default_factory=set)
    
    # Statistiques
    messages_sent: int = 0
    messages_received: int = 0
    frames_processed: int = 0
    detections_sent: int = 0
    errors_count: int = 0
    
    # Performance
    avg_processing_time: float = 0.0
    current_fps: float = 0.0
    
    # Queue des messages à envoyer
    message_queue: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation"""
        return {
            "client_id": self.client_id,
            "status": self.status.value,
            "connected_at": self.connected_at,
            "last_activity": self.last_activity,
            "model_name": self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "stream_quality": self.stream_quality.value,
            "subscriptions": list(self.subscriptions),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "frames_processed": self.frames_processed,
            "detections_sent": self.detections_sent,
            "errors_count": self.errors_count,
            "avg_processing_time": self.avg_processing_time,
            "current_fps": self.current_fps,
            "queue_size": len(self.message_queue)
        }

class GlobalStats(BaseModel):
    """📊 Statistiques globales WebSocket"""
    total_connections: int = 0
    active_connections: int = 0
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_frames_processed: int = 0
    total_detections_sent: int = 0
    average_fps: float = 0.0
    peak_concurrent_connections: int = 0
    uptime_seconds: float = 0.0
    
    def update_from_connections(self, connections: Dict[str, WebSocketConnection]):
        """Met à jour les stats depuis les connexions"""
        self.active_connections = len([c for c in connections.values() if c.status == ConnectionStatus.CONNECTED])
        self.total_messages_sent = sum(c.messages_sent for c in connections.values())
        self.total_messages_received = sum(c.messages_received for c in connections.values())
        self.total_frames_processed = sum(c.frames_processed for c in connections.values())
        self.total_detections_sent = sum(c.detections_sent for c in connections.values())
        
        # FPS moyen
        active_fps = [c.current_fps for c in connections.values() if c.current_fps > 0]
        self.average_fps = sum(active_fps) / len(active_fps) if active_fps else 0.0

# 🔧 GESTIONNAIRE DE CONNEXIONS
class ConnectionManager:
    """🔌 Gestionnaire des connexions WebSocket individuelles"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.connection_groups: Dict[str, Set[str]] = defaultdict(set)  # Groupes de connexions
        self.heartbeat_interval = 30.0  # Heartbeat toutes les 30s
        self.cleanup_interval = 60.0   # Nettoyage toutes les minutes
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """🚀 Démarre les tâches de maintenance"""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("🔌 ConnectionManager démarré")
    
    async def stop(self):
        """🛑 Arrête les tâches de maintenance"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Fermer toutes les connexions
        for connection in list(self.connections.values()):
            await self.disconnect(connection.client_id)
        
        logger.info("🔌 ConnectionManager arrêté")
    
    async def add_connection(
        self, 
        client_id: str, 
        websocket: WebSocket,
        **config
    ) -> WebSocketConnection:
        """➕ Ajoute une nouvelle connexion"""
        
        # Fermer connexion existante si elle existe
        if client_id in self.connections:
            await self.disconnect(client_id)
        
        # Créer la nouvelle connexion
        connection = WebSocketConnection(
            client_id=client_id,
            websocket=websocket,
            **config
        )
        
        self.connections[client_id] = connection
        connection.status = ConnectionStatus.CONNECTED
        
        logger.info(f"➕ Connexion ajoutée: {client_id}")
        return connection
    
    async def disconnect(self, client_id: str) -> bool:
        """➖ Déconnecte un client"""
        
        if client_id not in self.connections:
            return False
        
        connection = self.connections[client_id]
        connection.status = ConnectionStatus.DISCONNECTING
        
        try:
            # Fermer la WebSocket
            if connection.websocket:
                await connection.websocket.close()
        except Exception as e:
            logger.warning(f"⚠️ Erreur fermeture WebSocket {client_id}: {e}")
        
        # Retirer des groupes
        for group_clients in self.connection_groups.values():
            group_clients.discard(client_id)
        
        # Supprimer la connexion
        del self.connections[client_id]
        
        logger.info(f"➖ Connexion supprimée: {client_id}")
        return True
    
    def get_connection(self, client_id: str) -> Optional[WebSocketConnection]:
        """🔍 Récupère une connexion par ID"""
        return self.connections.get(client_id)
    
    def get_active_connections(self) -> List[WebSocketConnection]:
        """📋 Récupère toutes les connexions actives"""
        return [
            conn for conn in self.connections.values() 
            if conn.status == ConnectionStatus.CONNECTED
        ]
    
    def add_to_group(self, client_id: str, group_name: str):
        """👥 Ajoute un client à un groupe"""
        if client_id in self.connections:
            self.connection_groups[group_name].add(client_id)
    
    def remove_from_group(self, client_id: str, group_name: str):
        """👥 Retire un client d'un groupe"""
        self.connection_groups[group_name].discard(client_id)
    
    def get_group_clients(self, group_name: str) -> List[str]:
        """👥 Récupère les clients d'un groupe"""
        return list(self.connection_groups.get(group_name, set()))
    
    async def _heartbeat_loop(self):
        """💓 Boucle de heartbeat"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await self._send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Erreur heartbeat: {e}")
    
    async def _send_heartbeat(self):
        """💓 Envoie un heartbeat à tous les clients"""
        current_time = time.time()
        disconnected_clients = []
        
        for client_id, connection in self.connections.items():
            try:
                # Vérifier si la connexion est inactive
                if current_time - connection.last_activity > 120:  # 2 minutes
                    disconnected_clients.append(client_id)
                    continue
                
                # Envoyer ping
                await connection.websocket.send_json({
                    "type": MessageType.PING.value,
                    "timestamp": current_time
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur heartbeat client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Nettoyer les connexions mortes
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
    
    async def _cleanup_loop(self):
        """🧹 Boucle de nettoyage"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Erreur cleanup: {e}")
    
    async def _cleanup_connections(self):
        """🧹 Nettoie les connexions obsolètes"""
        current_time = time.time()
        to_cleanup = []
        
        for client_id, connection in self.connections.items():
            # Connexions inactives depuis plus de 5 minutes
            if current_time - connection.last_activity > 300:
                to_cleanup.append(client_id)
            
            # Connexions avec trop d'erreurs
            elif connection.errors_count > 10:
                to_cleanup.append(client_id)
        
        for client_id in to_cleanup:
            logger.info(f"🧹 Nettoyage connexion inactive: {client_id}")
            await self.disconnect(client_id)

# 📨 ROUTEUR DE MESSAGES
class MessageRouter:
    """📨 Routeur intelligent pour les messages WebSocket"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.message_handlers: Dict[MessageType, Callable] = {}
        self._setup_handlers()
    
    def _setup_handlers(self):
        """🔧 Configure les gestionnaires de messages"""
        self.message_handlers = {
            MessageType.CONNECT: self._handle_connect,
            MessageType.FRAME: self._handle_frame,
            MessageType.COMMAND: self._handle_command,
            MessageType.PING: self._handle_ping,
            MessageType.SUBSCRIBE: self._handle_subscribe,
            MessageType.UNSUBSCRIBE: self._handle_unsubscribe,
        }
    
    async def route_message(
        self, 
        connection: WebSocketConnection, 
        message: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """📨 Route un message vers le bon gestionnaire"""
        
        message_type = MessageType(message.get("type", "unknown"))
        handler = self.message_handlers.get(message_type)
        
        if not handler:
            logger.warning(f"⚠️ Type de message inconnu: {message_type}")
            return {
                "type": MessageType.ERROR.value,
                "message": f"Type de message non supporté: {message_type}"
            }
        
        try:
            return await handler(connection, message)
        except Exception as e:
            logger.error(f"❌ Erreur traitement message {message_type}: {e}")
            connection.errors_count += 1
            return {
                "type": MessageType.ERROR.value,
                "message": f"Erreur traitement: {str(e)}"
            }
    
    async def _handle_connect(
        self, 
        connection: WebSocketConnection, 
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """🔌 Gère la connexion initiale"""
        
        # Mise à jour configuration
        config = message.get("config", {})
        if "model_name" in config:
            connection.model_name = config["model_name"]
        if "confidence_threshold" in config:
            connection.confidence_threshold = config["confidence_threshold"]
        if "stream_quality" in config:
            connection.stream_quality = StreamQuality(config["stream_quality"])
        
        return {
            "type": MessageType.CONNECTION_ACK.value,
            "client_id": connection.client_id,
            "status": "connected",
            "server_info": {
                "classes_count": len(get_settings().DETECTION_CLASSES),
                "available_models": ["epoch_30", "extended"],
                "supported_qualities": [q.value for q in StreamQuality]
            }
        }
    
    async def _handle_frame(
        self, 
        connection: WebSocketConnection, 
        message: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """🖼️ Gère les frames de détection"""
        
        frame_data = message.get("data")
        frame_id = message.get("frame_id", f"frame_{int(time.time() * 1000)}")
        
        if not frame_data:
            return {
                "type": MessageType.ERROR.value,
                "message": "Données de frame manquantes"
            }
        
        try:
            # Décodage et détection
            import base64
            import io
            from PIL import Image
            
            # Décodage base64
            image_bytes = base64.b64decode(frame_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Détection via model service
            from app.services.stream_service import StreamDetectionService
            stream_service = StreamDetectionService(self.model_service)
            
            start_time = time.time()
            detections = await stream_service.detect_objects_realtime(
                image=image,
                confidence_threshold=connection.confidence_threshold,
                model_name=connection.model_name,
                quality=connection.stream_quality
            )
            processing_time = (time.time() - start_time) * 1000
            
            # Mise à jour statistiques
            connection.frames_processed += 1
            connection.detections_sent += len(detections)
            connection.avg_processing_time = (
                connection.avg_processing_time * (connection.frames_processed - 1) + processing_time
            ) / connection.frames_processed
            
            # Calcul FPS
            connection.current_fps = 1000 / processing_time if processing_time > 0 else 0
            
            return {
                "type": MessageType.DETECTIONS.value,
                "frame_id": frame_id,
                "detections": [det.dict() for det in detections],
                "processing_time_ms": processing_time,
                "fps": connection.current_fps,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement frame {connection.client_id}: {e}")
            return {
                "type": MessageType.ERROR.value,
                "message": f"Erreur traitement frame: {str(e)}"
            }
    
    async def _handle_command(
        self, 
        connection: WebSocketConnection, 
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """🎮 Gère les commandes"""
        
        action = message.get("action")
        params = message.get("params", {})
        
        if action == "change_threshold":
            threshold = params.get("threshold", 0.5)
            if 0.1 <= threshold <= 0.9:
                connection.confidence_threshold = threshold
                return {
                    "type": MessageType.STATUS_UPDATE.value,
                    "message": f"Seuil changé à {threshold}"
                }
        
        elif action == "change_quality":
            quality = params.get("quality")
            if quality in [q.value for q in StreamQuality]:
                connection.stream_quality = StreamQuality(quality)
                return {
                    "type": MessageType.STATUS_UPDATE.value,
                    "message": f"Qualité changée à {quality}"
                }
        
        elif action == "get_stats":
            return {
                "type": MessageType.STATISTICS.value,
                "stats": connection.to_dict()
            }
        
        return {
            "type": MessageType.ERROR.value,
            "message": f"Commande inconnue: {action}"
        }
    
    async def _handle_ping(
        self, 
        connection: WebSocketConnection, 
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """🏓 Gère les pings"""
        return {
            "type": MessageType.PONG.value,
            "timestamp": time.time()
        }
    
    async def _handle_subscribe(
        self, 
        connection: WebSocketConnection, 
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """📺 Gère les abonnements"""
        
        topics = message.get("topics", [])
        for topic in topics:
            connection.subscriptions.add(topic)
        
        return {
            "type": MessageType.STATUS_UPDATE.value,
            "message": f"Abonné à {len(topics)} topics"
        }
    
    async def _handle_unsubscribe(
        self, 
        connection: WebSocketConnection, 
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """📺 Gère les désabonnements"""
        
        topics = message.get("topics", [])
        for topic in topics:
            connection.subscriptions.discard(topic)
        
        return {
            "type": MessageType.STATUS_UPDATE.value,
            "message": f"Désabonné de {len(topics)} topics"
        }

# 🎯 GESTIONNAIRE PRINCIPAL
class WebSocketManager:
    """🎯 Gestionnaire principal des WebSocket"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.connection_manager = ConnectionManager()
        self.message_router = MessageRouter(model_service)
        self.global_stats = GlobalStats()
        self.start_time = time.time()
        self._running = False
        
        # Queues pour traitement asynchrone
        self.message_queue = asyncio.Queue()
        self.broadcast_queue = asyncio.Queue()
        
        # Tâches de fond
        self._message_processor_task: Optional[asyncio.Task] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """🚀 Démarre le gestionnaire WebSocket"""
        if self._running:
            return
        
        self._running = True
        
        # Démarrer les composants
        await self.connection_manager.start()
        
        # Démarrer les tâches de fond
        self._message_processor_task = asyncio.create_task(self._process_messages())
        self._broadcast_task = asyncio.create_task(self._process_broadcasts())
        self._stats_task = asyncio.create_task(self._update_stats())
        
        logger.info("🚀 WebSocketManager démarré")
    
    async def stop(self):
        """🛑 Arrête le gestionnaire WebSocket"""
        if not self._running:
            return
        
        self._running = False
        
        # Arrêter les tâches
        for task in [self._message_processor_task, self._broadcast_task, self._stats_task]:
            if task:
                task.cancel()
        
        # Arrêter les composants
        await self.connection_manager.stop()
        
        logger.info("🛑 WebSocketManager arrêté")
    
    async def cleanup(self):
        """🧹 Nettoyage final"""
        await self.stop()
    
    async def connect(self, websocket: WebSocket, client_id: str, **config):
        """🔌 Connecte un nouveau client"""
        
        try:
            await websocket.accept()
            
            connection = await self.connection_manager.add_connection(
                client_id=client_id,
                websocket=websocket,
                **config
            )
            
            # Mise à jour stats
            self.global_stats.total_connections += 1
            if self.global_stats.active_connections > self.global_stats.peak_concurrent_connections:
                self.global_stats.peak_concurrent_connections = self.global_stats.active_connections
            
            logger.info(f"🔌 Client connecté: {client_id}")
            
            # Message de bienvenue
            await self.send_to_client(client_id, {
                "type": MessageType.CONNECTION_ACK.value,
                "client_id": client_id,
                "message": "Connexion établie avec succès",
                "server_info": {
                    "version": "1.0.0",
                    "features": ["detection", "streaming", "notifications"]
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion {client_id}: {e}")
            raise
    
    async def disconnect(self, client_id: str):
        """🔌 Déconnecte un client"""
        await self.connection_manager.disconnect(client_id)
    
    async def handle_message(self, client_id: str, message: Dict[str, Any]):
        """📨 Traite un message reçu"""
        
        connection = self.connection_manager.get_connection(client_id)
        if not connection:
            logger.warning(f"⚠️ Message de client inconnu: {client_id}")
            return
        
        connection.last_activity = time.time()
        connection.messages_received += 1
        
        # Ajouter à la queue de traitement
        await self.message_queue.put((connection, message))
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]) -> bool:
        """📤 Envoie un message à un client"""
        
        connection = self.connection_manager.get_connection(client_id)
        if not connection or connection.status != ConnectionStatus.CONNECTED:
            return False
        
        try:
            await connection.websocket.send_json(message)
            connection.messages_sent += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur envoi message {client_id}: {e}")
            await self.disconnect(client_id)
            return False
    
    async def broadcast(self, message: Dict[str, Any], group: Optional[str] = None):
        """📢 Diffuse un message à tous les clients ou un groupe"""
        
        if group:
            client_ids = self.connection_manager.get_group_clients(group)
        else:
            client_ids = list(self.connection_manager.connections.keys())
        
        for client_id in client_ids:
            await self.send_to_client(client_id, message)
    
    async def _process_messages(self):
        """📨 Traite les messages en arrière-plan"""
        
        while self._running:
            try:
                # Récupérer message avec timeout
                connection, message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                # Router le message
                response = await self.message_router.route_message(connection, message)
                
                # Envoyer la réponse si nécessaire
                if response:
                    await self.send_to_client(connection.client_id, response)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"❌ Erreur traitement message: {e}")
    
    async def _process_broadcasts(self):
        """📢 Traite les diffusions en arrière-plan"""
        
        while self._running:
            try:
                # Récupérer broadcast avec timeout
                message, group = await asyncio.wait_for(
                    self.broadcast_queue.get(),
                    timeout=1.0
                )
                
                await self.broadcast(message, group)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"❌ Erreur broadcast: {e}")
    
    async def _update_stats(self):
        """📊 Met à jour les statistiques périodiquement"""
        
        while self._running:
            try:
                await asyncio.sleep(10)  # Toutes les 10 secondes
                
                # Mise à jour des stats globales
                self.global_stats.update_from_connections(
                    self.connection_manager.connections
                )
                self.global_stats.uptime_seconds = time.time() - self.start_time
                
            except Exception as e:
                logger.error(f"❌ Erreur mise à jour stats: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """📊 Récupère les statistiques des connexions"""
        
        return {
            "global_stats": self.global_stats.dict(),
            "active_connections": len(self.connection_manager.get_active_connections()),
            "total_connections": len(self.connection_manager.connections),
            "connections_by_status": self._get_connections_by_status(),
            "peak_concurrent": self.global_stats.peak_concurrent_connections
        }
    
    def _get_connections_by_status(self) -> Dict[str, int]:
        """📊 Compte les connexions par statut"""
        
        status_counts = defaultdict(int)
        for connection in self.connection_manager.connections.values():
            status_counts[connection.status.value] += 1
        
        return dict(status_counts)
    
    def get_current_timestamp(self) -> float:
        """⏰ Récupère le timestamp actuel"""
        return time.time()

# 🎯 GESTIONNAIRE DE STREAM SPÉCIALISÉ
class StreamHandler:
    """🎯 Gestionnaire spécialisé pour les streams de détection"""
    
    def __init__(self, websocket_manager: WebSocketManager):
        self.websocket_manager = websocket_manager
        self.active_streams: Dict[str, Dict[str, Any]] = {}
    
    async def start_stream(self, client_id: str, stream_config: Dict[str, Any]):
        """🚀 Démarre un stream de détection"""
        
        self.active_streams[client_id] = {
            "config": stream_config,
            "started_at": time.time(),
            "frames_processed": 0,
            "detections_sent": 0
        }
        
        await self.websocket_manager.send_to_client(client_id, {
            "type": MessageType.STATUS_UPDATE.value,
            "message": "Stream démarré",
            "stream_id": client_id
        })
    
    async def stop_stream(self, client_id: str):
        """🛑 Arrête un stream de détection"""
        
        if client_id in self.active_streams:
            stream_info = self.active_streams.pop(client_id)
            
            await self.websocket_manager.send_to_client(client_id, {
                "type": MessageType.STATUS_UPDATE.value,
                "message": "Stream arrêté",
                "stream_stats": stream_info
            })

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "WebSocketManager",
    "StreamHandler", 
    "ConnectionManager",
    "MessageRouter",
    "WebSocketConnection",
    "ConnectionStatus",
    "MessageType",
    "StreamQuality"
]