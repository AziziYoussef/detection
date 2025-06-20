"""
📡 WEBSOCKET STREAM HANDLER - GESTIONNAIRE WEBSOCKET TEMPS RÉEL
=============================================================
Gestionnaire centralisé pour les connexions WebSocket de streaming

Fonctionnalités:
- Gestion des connexions multiples
- Routage des messages par type
- Heartbeat et surveillance de santé
- Gestion des erreurs et reconnexions
- Broadcasting et communication client-serveur
- Intégration avec StreamService
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

# Imports internes
from app.services.stream_service import StreamService, StreamClient
from app.schemas.detection import (
    StreamConfig, StreamMessage, FrameMessage, DetectionMessage,
    StatusMessage, ErrorMessage, MessageType, DetectionRequest
)
from app.config.config import Settings

logger = logging.getLogger(__name__)

class ConnectionManager:
    """🔌 Gestionnaire de connexions WebSocket"""
    
    def __init__(self):
        # Connexions actives par client_id
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Métadonnées des connexions
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Statistiques
        self.total_connections = 0
        self.total_messages = 0
        self.start_time = datetime.now()
        
    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """🔌 Accepte une nouvelle connexion"""
        
        try:
            await websocket.accept()
            
            # Fermer l'ancienne connexion si elle existe
            if client_id in self.active_connections:
                await self.disconnect(client_id)
            
            # Enregistrer la nouvelle connexion
            self.active_connections[client_id] = websocket
            self.connection_metadata[client_id] = {
                "connected_at": datetime.now(),
                "messages_received": 0,
                "messages_sent": 0,
                "last_activity": datetime.now()
            }
            
            self.total_connections += 1
            
            logger.info(f"🔌 Connexion WebSocket acceptée: {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion WebSocket {client_id}: {e}")
            return False
    
    async def disconnect(self, client_id: str):
        """🔌 Ferme une connexion"""
        
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.close()
            except:
                pass  # Connexion déjà fermée
            
            del self.active_connections[client_id]
            
            if client_id in self.connection_metadata:
                metadata = self.connection_metadata[client_id]
                session_duration = (datetime.now() - metadata["connected_at"]).total_seconds()
                
                logger.info(
                    f"🔌 Connexion fermée: {client_id} "
                    f"(durée: {session_duration:.1f}s, messages: {metadata['messages_received']})"
                )
                
                del self.connection_metadata[client_id]
    
    async def send_personal_message(self, message: str, client_id: str) -> bool:
        """📤 Envoie un message à un client spécifique"""
        
        if client_id not in self.active_connections:
            return False
        
        try:
            websocket = self.active_connections[client_id]
            await websocket.send_text(message)
            
            # Mise à jour métadonnées
            if client_id in self.connection_metadata:
                self.connection_metadata[client_id]["messages_sent"] += 1
                self.connection_metadata[client_id]["last_activity"] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur envoi message à {client_id}: {e}")
            await self.disconnect(client_id)
            return False
    
    async def broadcast(self, message: str, exclude: Optional[Set[str]] = None):
        """📢 Diffuse un message à tous les clients connectés"""
        
        exclude = exclude or set()
        
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            if client_id in exclude:
                continue
            
            try:
                await websocket.send_text(message)
                
                if client_id in self.connection_metadata:
                    self.connection_metadata[client_id]["messages_sent"] += 1
                    
            except Exception as e:
                logger.error(f"❌ Erreur broadcast à {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Nettoyer les connexions fermées
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
    
    def get_active_clients(self) -> List[str]:
        """📋 Liste des clients actifs"""
        return list(self.active_connections.keys())
    
    def get_connection_count(self) -> int:
        """📊 Nombre de connexions actives"""
        return len(self.active_connections)
    
    def get_statistics(self) -> Dict[str, Any]:
        """📊 Statistiques des connexions"""
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "active_connections": len(self.active_connections),
            "total_connections": self.total_connections,
            "total_messages": self.total_messages,
            "uptime_seconds": uptime,
            "connections_per_minute": self.total_connections / max(uptime / 60, 1),
            "active_clients": list(self.active_connections.keys())
        }

class MessageRouter:
    """📮 Routeur de messages WebSocket"""
    
    def __init__(self, stream_service: StreamService):
        self.stream_service = stream_service
        
        # Handlers par type de message
        self.message_handlers = {
            MessageType.FRAME: self._handle_frame_message,
            MessageType.STATUS: self._handle_status_message,
            MessageType.PING: self._handle_ping_message,
        }
    
    async def route_message(
        self, 
        client_id: str, 
        message_data: Dict[str, Any],
        websocket: WebSocket
    ):
        """🎯 Route un message vers le bon handler"""
        
        try:
            # Extraction du type de message
            message_type = message_data.get("type")
            
            if not message_type:
                await self._send_error(websocket, client_id, "MESSAGE_TYPE_MISSING", 
                                     "Type de message manquant")
                return
            
            # Conversion en MessageType
            try:
                msg_type = MessageType(message_type)
            except ValueError:
                await self._send_error(websocket, client_id, "INVALID_MESSAGE_TYPE", 
                                     f"Type de message invalide: {message_type}")
                return
            
            # Routage vers le handler approprié
            handler = self.message_handlers.get(msg_type)
            if handler:
                await handler(client_id, message_data, websocket)
            else:
                await self._send_error(websocket, client_id, "NO_HANDLER", 
                                     f"Pas de handler pour le type: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ Erreur routage message {client_id}: {e}")
            await self._send_error(websocket, client_id, "ROUTING_ERROR", str(e))
    
    async def _handle_frame_message(
        self, 
        client_id: str, 
        message_data: Dict[str, Any],
        websocket: WebSocket
    ):
        """🖼️ Traite un message de frame"""
        
        try:
            # Validation du message de frame
            frame_message = FrameMessage(**message_data)
            
            # Transmission au service de streaming
            await self.stream_service.handle_frame_message(client_id, {
                "frame_data": frame_message.frame_data,
                "frame_number": frame_message.frame_number,
                "timestamp": frame_message.frame_timestamp.timestamp()
            })
            
        except ValidationError as e:
            await self._send_error(websocket, client_id, "INVALID_FRAME_MESSAGE", str(e))
        except Exception as e:
            logger.error(f"❌ Erreur traitement frame {client_id}: {e}")
            await self._send_error(websocket, client_id, "FRAME_PROCESSING_ERROR", str(e))
    
    async def _handle_status_message(
        self, 
        client_id: str, 
        message_data: Dict[str, Any],
        websocket: WebSocket
    ):
        """📊 Traite un message de statut"""
        
        try:
            status_message = StatusMessage(**message_data)
            
            # Actions selon le statut
            if status_message.status == "ping":
                # Répondre avec pong
                pong_message = StatusMessage(
                    type=MessageType.STATUS,
                    client_id=client_id,
                    status="pong",
                    data={"timestamp": time.time()}
                )
                
                await websocket.send_text(pong_message.json())
                
            elif status_message.status == "config_update":
                # Mise à jour de configuration
                await self._handle_config_update(client_id, status_message.data)
                
        except ValidationError as e:
            await self._send_error(websocket, client_id, "INVALID_STATUS_MESSAGE", str(e))
        except Exception as e:
            logger.error(f"❌ Erreur traitement statut {client_id}: {e}")
    
    async def _handle_ping_message(
        self, 
        client_id: str, 
        message_data: Dict[str, Any],
        websocket: WebSocket
    ):
        """🏓 Traite un ping"""
        
        # Réponse pong simple
        pong_response = {
            "type": "pong",
            "client_id": client_id,
            "timestamp": time.time(),
            "server_time": datetime.now().isoformat()
        }
        
        await websocket.send_text(json.dumps(pong_response))
    
    async def _handle_config_update(self, client_id: str, config_data: Dict[str, Any]):
        """⚙️ Met à jour la configuration d'un client"""
        
        try:
            # Récupération du client dans le service de streaming
            with self.stream_service.client_lock:
                if client_id in self.stream_service.clients:
                    client = self.stream_service.clients[client_id]
                    
                    # Mise à jour des paramètres de détection
                    if "detection_params" in config_data:
                        detection_params = config_data["detection_params"]
                        
                        client.config.detection_params.confidence_threshold = detection_params.get(
                            "confidence_threshold", 
                            client.config.detection_params.confidence_threshold
                        )
                        
                        client.config.detection_params.detection_mode = detection_params.get(
                            "detection_mode",
                            client.config.detection_params.detection_mode
                        )
                    
                    logger.info(f"⚙️ Configuration mise à jour pour {client_id}")
                    
        except Exception as e:
            logger.error(f"❌ Erreur mise à jour config {client_id}: {e}")
    
    async def _send_error(
        self, 
        websocket: WebSocket, 
        client_id: str, 
        error_code: str, 
        error_message: str
    ):
        """❌ Envoie un message d'erreur"""
        
        error_msg = ErrorMessage(
            type=MessageType.ERROR,
            client_id=client_id,
            error_code=error_code,
            error_message=error_message
        )
        
        try:
            await websocket.send_text(error_msg.json())
        except:
            pass  # Connexion fermée

class WebSocketManager:
    """📡 Gestionnaire principal WebSocket"""
    
    def __init__(self, stream_service: StreamService):
        self.stream_service = stream_service
        self.connection_manager = ConnectionManager()
        self.message_router = MessageRouter(stream_service)
        
        # Tâches de background
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Configuration
        self.heartbeat_interval = 30  # secondes
        self.connection_timeout = 300  # 5 minutes
        
        logger.info("📡 WebSocketManager initialisé")
    
    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """🔌 Connecte un nouveau client WebSocket"""
        
        # Validation de l'ID client
        if not client_id or len(client_id) < 3:
            logger.warning(f"⚠️ ID client invalide: {client_id}")
            await websocket.close(code=4000, reason="ID client invalide")
            return False
        
        # Connexion via le gestionnaire
        connected = await self.connection_manager.connect(websocket, client_id)
        
        if connected:
            # Configuration par défaut du streaming
            default_config = StreamConfig(
                client_id=client_id,
                detection_params=DetectionRequest(),
                enable_lost_object_tracking=True
            )
            
            # Enregistrer le client dans le service de streaming
            await self.stream_service.connect_client(websocket, client_id, default_config)
            
            # Démarrer le heartbeat pour ce client
            heartbeat_task = asyncio.create_task(self._client_heartbeat(client_id))
            self.background_tasks.add(heartbeat_task)
            
            return True
        
        return False
    
    async def disconnect(self, client_id: str):
        """🔌 Déconnecte un client"""
        
        # Déconnexion du service de streaming
        await self.stream_service.disconnect_client(client_id)
        
        # Déconnexion WebSocket
        await self.connection_manager.disconnect(client_id)
    
    async def handle_message(self, client_id: str, message_data: Dict[str, Any]):
        """📨 Traite un message reçu"""
        
        # Mise à jour de l'activité
        if client_id in self.connection_manager.connection_metadata:
            metadata = self.connection_manager.connection_metadata[client_id]
            metadata["messages_received"] += 1
            metadata["last_activity"] = datetime.now()
        
        self.connection_manager.total_messages += 1
        
        # Récupération de la connexion WebSocket
        if client_id not in self.connection_manager.active_connections:
            logger.warning(f"⚠️ Message reçu pour client non connecté: {client_id}")
            return
        
        websocket = self.connection_manager.active_connections[client_id]
        
        # Routage du message
        await self.message_router.route_message(client_id, message_data, websocket)
    
    async def _client_heartbeat(self, client_id: str):
        """💓 Heartbeat pour un client spécifique"""
        
        try:
            while client_id in self.connection_manager.active_connections:
                # Envoyer ping
                ping_message = {
                    "type": "ping",
                    "client_id": client_id,
                    "timestamp": time.time(),
                    "server_status": "healthy"
                }
                
                success = await self.connection_manager.send_personal_message(
                    json.dumps(ping_message), 
                    client_id
                )
                
                if not success:
                    break
                
                # Attendre le prochain heartbeat
                await asyncio.sleep(self.heartbeat_interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"❌ Erreur heartbeat {client_id}: {e}")
        
        # Nettoyage
        self.background_tasks.discard(asyncio.current_task())
    
    async def broadcast_status(self, status: str, data: Optional[Dict[str, Any]] = None):
        """📢 Diffuse un statut à tous les clients"""
        
        broadcast_message = {
            "type": "status",
            "status": status,
            "data": data or {},
            "timestamp": time.time(),
            "server_id": "ai-detection-service"
        }
        
        await self.connection_manager.broadcast(json.dumps(broadcast_message))
    
    async def send_alert_to_client(self, client_id: str, alert_data: Dict[str, Any]):
        """🚨 Envoie une alerte à un client spécifique"""
        
        alert_message = {
            "type": "alert",
            "client_id": client_id,
            "alert_data": alert_data,
            "timestamp": time.time(),
            "severity": alert_data.get("severity", "info")
        }
        
        await self.connection_manager.send_personal_message(
            json.dumps(alert_message),
            client_id
        )
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """📊 Statistiques du gestionnaire WebSocket"""
        
        connection_stats = self.connection_manager.get_statistics()
        stream_stats = self.stream_service.get_service_statistics()
        
        return {
            "websocket_manager": {
                **connection_stats,
                "active_background_tasks": len(self.background_tasks),
                "heartbeat_interval": self.heartbeat_interval
            },
            "stream_service": stream_stats
        }
    
    async def cleanup(self):
        """🧹 Nettoyage des ressources"""
        
        logger.info("🧹 Nettoyage WebSocketManager...")
        
        # Arrêter toutes les tâches de background
        for task in self.background_tasks:
            task.cancel()
        
        # Attendre la fin des tâches
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Fermer toutes les connexions
        client_ids = list(self.connection_manager.active_connections.keys())
        for client_id in client_ids:
            await self.disconnect(client_id)
        
        # Nettoyage du service de streaming
        await self.stream_service.cleanup()
        
        logger.info("✅ WebSocketManager nettoyé")

# === EXPORTS ===
__all__ = [
    "WebSocketManager",
    "ConnectionManager",
    "MessageRouter"
]