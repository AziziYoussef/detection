"""
📡 STREAM SERVICE - SERVICE DE STREAMING TEMPS RÉEL
================================================
Service spécialisé pour le traitement temps réel via WebSocket

Fonctionnalités:
- Streaming WebSocket haute performance
- Détection temps réel avec latence minimale
- Tracking continu des objets perdus
- Alertes instantanées
- Gestion multi-clients simultanés
- Optimisations pour mobile/webcam
- Buffer intelligent pour stabilité
"""

import asyncio
import logging
import time
import json
import base64
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from collections import deque
import threading
import uuid
import io

import numpy as np
from PIL import Image
import cv2
import torch

# Imports internes
from app.core.model_manager import ModelManager
from app.core.detector import ObjectDetector, DetectionConfig
from app.core.preprocessing import ImagePreprocessor
from app.core.postprocessing import LostObjectDetector
from app.schemas.detection import (
    DetectionResult, StreamConfig, DetectionMode, ModelType,
    LostObjectState, ObjectStatus, StreamMessage, FrameMessage,
    DetectionMessage, StatusMessage, ErrorMessage, MessageType
)
from app.config.config import Settings

logger = logging.getLogger(__name__)

class StreamClient:
    """👤 Client de streaming individuel"""
    
    def __init__(self, client_id: str, websocket, config: StreamConfig):
        self.client_id = client_id
        self.websocket = websocket
        self.config = config
        
        # État de la connexion
        self.connected_at = datetime.now()
        self.last_frame_time = datetime.now()
        self.last_heartbeat = datetime.now()
        self.is_active = True
        
        # Buffer de frames
        self.frame_buffer: deque = deque(maxlen=5)
        self.processing_buffer: deque = deque(maxlen=10)
        
        # Statistiques
        self.frames_received = 0
        self.frames_processed = 0
        self.detections_sent = 0
        self.average_latency = 0.0
        self.latency_history: deque = deque(maxlen=50)
        
        # Tracking spécialisé pour ce client
        self.lost_object_detector = None
        
        # Alertes en attente
        self.pending_alerts: List[Dict[str, Any]] = []
        
        logger.info(f"📡 Nouveau client streaming: {client_id}")
    
    @property
    def session_duration(self) -> float:
        """⏱️ Durée de la session en secondes"""
        return (datetime.now() - self.connected_at).total_seconds()
    
    @property
    def is_healthy(self) -> bool:
        """🏥 Vérifie si la connexion est saine"""
        now = datetime.now()
        
        # Vérifications de santé
        no_recent_frame = (now - self.last_frame_time).total_seconds() > 30
        no_heartbeat = (now - self.last_heartbeat).total_seconds() > 60
        
        return self.is_active and not no_recent_frame and not no_heartbeat
    
    async def send_message(self, message: StreamMessage):
        """📤 Envoie un message au client"""
        
        if not self.is_active:
            return
        
        try:
            message_data = message.dict()
            await self.websocket.send_text(json.dumps(message_data))
            
        except Exception as e:
            logger.error(f"❌ Erreur envoi message à {self.client_id}: {e}")
            self.is_active = False
    
    def add_frame(self, frame_data: str, frame_number: int):
        """➕ Ajoute une frame au buffer"""
        
        self.frames_received += 1
        self.last_frame_time = datetime.now()
        
        frame_info = {
            "data": frame_data,
            "number": frame_number,
            "timestamp": time.time(),
            "received_at": datetime.now()
        }
        
        self.frame_buffer.append(frame_info)
    
    def get_next_frame(self) -> Optional[Dict[str, Any]]:
        """⏭️ Récupère la prochaine frame à traiter"""
        
        if self.frame_buffer:
            return self.frame_buffer.popleft()
        return None
    
    def update_latency(self, processing_time: float):
        """📊 Met à jour les métriques de latence"""
        
        self.latency_history.append(processing_time)
        self.average_latency = sum(self.latency_history) / len(self.latency_history)
    
    def get_statistics(self) -> Dict[str, Any]:
        """📊 Statistiques du client"""
        
        return {
            "client_id": self.client_id,
            "session_duration": self.session_duration,
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "detections_sent": self.detections_sent,
            "average_latency_ms": self.average_latency * 1000,
            "buffer_size": len(self.frame_buffer),
            "is_healthy": self.is_healthy,
            "pending_alerts": len(self.pending_alerts)
        }

class FrameProcessor:
    """🖼️ Processeur optimisé pour frames temps réel"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.preprocessor = ImagePreprocessor()
        
        # Configuration optimisée pour streaming
        self.max_frame_size = (640, 480)  # Résolution max pour temps réel
        self.jpeg_quality = 70
        
    async def decode_frame(self, frame_data: str) -> Optional[Image.Image]:
        """🔓 Décode une frame base64"""
        
        try:
            # Suppression du préfixe data URL si présent
            if frame_data.startswith('data:'):
                frame_data = frame_data.split(',', 1)[1]
            
            # Décodage base64
            image_bytes = base64.b64decode(frame_data)
            
            # Chargement de l'image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Conversion RGB si nécessaire
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Redimensionnement si trop grande
            if (image.width > self.max_frame_size[0] or 
                image.height > self.max_frame_size[1]):
                
                image.thumbnail(self.max_frame_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"❌ Erreur décodage frame: {e}")
            return None
    
    async def encode_frame(self, image: Image.Image) -> str:
        """🔐 Encode une frame en base64"""
        
        try:
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=self.jpeg_quality)
            
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded}"
            
        except Exception as e:
            logger.error(f"❌ Erreur encodage frame: {e}")
            return ""

class AlertManager:
    """🚨 Gestionnaire d'alertes temps réel"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Configuration des alertes
        self.alert_cooldowns = {
            'suspect': 30,    # 30 secondes entre alertes suspect
            'lost': 60,       # 1 minute entre alertes perdu
            'critical': 300   # 5 minutes entre alertes critique
        }
        
        # Historique des alertes par client
        self.client_alerts: Dict[str, Dict[str, datetime]] = {}
        
    async def check_and_send_alerts(
        self,
        client: StreamClient,
        lost_objects: List[LostObjectState]
    ):
        """🔔 Vérifie et envoie les alertes nécessaires"""
        
        if client.client_id not in self.client_alerts:
            self.client_alerts[client.client_id] = {}
        
        client_alert_history = self.client_alerts[client.client_id]
        now = datetime.now()
        
        for lost_object in lost_objects:
            # Ne traiter que les objets avec statut d'alerte
            if lost_object.status in [ObjectStatus.SUSPECT, ObjectStatus.LOST, ObjectStatus.CRITICAL]:
                
                alert_key = f"{lost_object.object_id}_{lost_object.status.value}"
                last_alert_time = client_alert_history.get(alert_key)
                
                # Vérifier le cooldown
                cooldown = self.alert_cooldowns.get(lost_object.status.value, 60)
                
                if (last_alert_time is None or 
                    (now - last_alert_time).total_seconds() > cooldown):
                    
                    # Envoyer l'alerte
                    await self._send_alert(client, lost_object)
                    client_alert_history[alert_key] = now
    
    async def _send_alert(self, client: StreamClient, lost_object: LostObjectState):
        """📢 Envoie une alerte spécifique"""
        
        alert_data = {
            "type": "lost_object_alert",
            "object_id": lost_object.object_id,
            "status": lost_object.status.value,
            "class_name": lost_object.detection_result.class_name_fr,
            "duration": lost_object.stationary_duration,
            "confidence": lost_object.detection_result.confidence,
            "position": {
                "x": lost_object.detection_result.bbox.center[0],
                "y": lost_object.detection_result.bbox.center[1]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Message d'alerte
        alert_message = StatusMessage(
            type=MessageType.STATUS,
            client_id=client.client_id,
            status="alert",
            data=alert_data
        )
        
        await client.send_message(alert_message)
        
        # Ajouter aux alertes en attente
        client.pending_alerts.append(alert_data)
        
        logger.warning(
            f"🚨 Alerte envoyée à {client.client_id}: "
            f"{lost_object.detection_result.class_name} {lost_object.status.value}"
        )

class StreamService:
    """📡 Service principal de streaming temps réel"""
    
    def __init__(self, model_manager: ModelManager, settings: Settings):
        self.model_manager = model_manager
        self.settings = settings
        
        # Gestionnaires
        self.frame_processor = FrameProcessor(settings)
        self.alert_manager = AlertManager(settings)
        
        # Clients connectés
        self.clients: Dict[str, StreamClient] = {}
        self.client_lock = threading.RLock()
        
        # Pool de détecteurs (réutilisation)
        self.detector_pool: List[ObjectDetector] = []
        self.detector_lock = asyncio.Lock()
        
        # Tâches de background
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Statistiques globales
        self.total_connections = 0
        self.total_frames_processed = 0
        self.start_time = datetime.now()
        
        logger.info("📡 StreamService initialisé")
    
    async def initialize(self):
        """🚀 Initialise le service"""
        
        # Pré-charger des détecteurs pour le pool
        await self._initialize_detector_pool()
        
        # Démarrer les tâches de background
        await self._start_background_tasks()
        
        logger.info("✅ StreamService initialisé")
    
    async def _initialize_detector_pool(self):
        """🔧 Initialise le pool de détecteurs"""
        
        # Configuration optimisée pour streaming
        config = DetectionConfig(
            confidence_threshold=0.5,
            nms_threshold=0.4,
            max_detections=20,  # Limité pour performance
            detection_mode=DetectionMode.FAST,
            model_type=ModelType.FAST,  # Modèle rapide
            half_precision=True  # FP16 pour vitesse
        )
        
        # Créer quelques détecteurs
        for i in range(min(3, self.settings.WORKERS)):
            detector = ObjectDetector(self.model_manager, config)
            self.detector_pool.append(detector)
        
        logger.info(f"🔧 Pool de détecteurs initialisé: {len(self.detector_pool)} détecteurs")
    
    async def _start_background_tasks(self):
        """🔄 Démarre les tâches de background"""
        
        # Tâche de nettoyage des clients inactifs
        cleanup_task = asyncio.create_task(self._cleanup_inactive_clients())
        self.background_tasks.add(cleanup_task)
        
        # Tâche de monitoring
        monitor_task = asyncio.create_task(self._monitor_service_health())
        self.background_tasks.add(monitor_task)
        
        # Tâche de traitement des frames
        processor_task = asyncio.create_task(self._frame_processing_loop())
        self.background_tasks.add(processor_task)
    
    async def connect_client(self, websocket, client_id: str, config: StreamConfig) -> StreamClient:
        """🔌 Connecte un nouveau client"""
        
        with self.client_lock:
            if client_id in self.clients:
                # Déconnecter l'ancien client avec ce même ID
                await self.disconnect_client(client_id)
            
            # Créer le nouveau client
            client = StreamClient(client_id, websocket, config)
            
            # Initialiser le tracker d'objets perdus pour ce client
            if config.enable_lost_object_tracking:
                client.lost_object_detector = LostObjectDetector(self.settings)
            
            self.clients[client_id] = client
            self.total_connections += 1
            
            logger.info(f"🔌 Client connecté: {client_id} (total: {len(self.clients)})")
            
            # Message de bienvenue
            welcome_message = StatusMessage(
                type=MessageType.STATUS,
                client_id=client_id,
                status="connected",
                data={"message": "Connexion établie", "client_id": client_id}
            )
            
            await client.send_message(welcome_message)
            
            return client
    
    async def disconnect_client(self, client_id: str):
        """🔌 Déconnecte un client"""
        
        with self.client_lock:
            if client_id in self.clients:
                client = self.clients[client_id]
                client.is_active = False
                
                try:
                    # Message de déconnexion
                    goodbye_message = StatusMessage(
                        type=MessageType.STATUS,
                        client_id=client_id,
                        status="disconnected",
                        data={"message": "Connexion fermée"}
                    )
                    await client.send_message(goodbye_message)
                    
                except:
                    pass  # Connexion déjà fermée
                
                del self.clients[client_id]
                
                logger.info(f"🔌 Client déconnecté: {client_id} (restant: {len(self.clients)})")
    
    async def handle_frame_message(self, client_id: str, message_data: Dict[str, Any]):
        """🖼️ Traite un message de frame"""
        
        with self.client_lock:
            if client_id not in self.clients:
                return
            
            client = self.clients[client_id]
        
        # Extraire les données de la frame
        frame_data = message_data.get("frame_data", "")
        frame_number = message_data.get("frame_number", 0)
        
        if not frame_data:
            logger.warning(f"⚠️ Frame vide reçue de {client_id}")
            return
        
        # Ajouter au buffer du client
        client.add_frame(frame_data, frame_number)
    
    async def _frame_processing_loop(self):
        """🔄 Boucle principale de traitement des frames"""
        
        while True:
            try:
                # Traiter les frames de tous les clients
                processing_tasks = []
                
                with self.client_lock:
                    active_clients = list(self.clients.values())
                
                for client in active_clients:
                    if client.is_active and len(client.frame_buffer) > 0:
                        # Créer une tâche de traitement pour ce client
                        task = asyncio.create_task(self._process_client_frame(client))
                        processing_tasks.append(task)
                
                # Traiter en parallèle
                if processing_tasks:
                    await asyncio.gather(*processing_tasks, return_exceptions=True)
                
                # Petit délai pour éviter la surcharge CPU
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle de traitement: {e}")
                await asyncio.sleep(1)
    
    async def _process_client_frame(self, client: StreamClient):
        """🔄 Traite une frame d'un client"""
        
        # Récupérer la frame
        frame_info = client.get_next_frame()
        if not frame_info:
            return
        
        start_time = time.time()
        
        try:
            # Décodage de la frame
            image = await self.frame_processor.decode_frame(frame_info["data"])
            if not image:
                return
            
            # Obtenir un détecteur du pool
            detector = await self._get_detector_from_pool()
            if not detector:
                logger.warning("⚠️ Aucun détecteur disponible dans le pool")
                return
            
            try:
                # Détection
                detections = await detector.detect_objects(
                    image=image,
                    model_name=client.config.detection_params.model_name.value,
                    confidence_threshold=client.config.detection_params.confidence_threshold,
                    detection_id=f"{client.client_id}_frame_{frame_info['number']}"
                )
                
                # Analyse objets perdus
                lost_objects = []
                if (client.config.enable_lost_object_tracking and 
                    client.lost_object_detector):
                    
                    lost_objects = client.lost_object_detector.analyze_detections(
                        detections,
                        datetime.now()
                    )
                
                # Envoyer les résultats
                await self._send_detection_results(client, detections, lost_objects, frame_info)
                
                # Vérifier et envoyer les alertes
                if lost_objects:
                    await self.alert_manager.check_and_send_alerts(client, lost_objects)
                
                # Métriques
                processing_time = time.time() - start_time
                client.update_latency(processing_time)
                client.frames_processed += 1
                self.total_frames_processed += 1
                
            finally:
                # Remettre le détecteur dans le pool
                await self._return_detector_to_pool(detector)
                
        except Exception as e:
            logger.error(f"❌ Erreur traitement frame client {client.client_id}: {e}")
            
            # Envoyer message d'erreur
            error_message = ErrorMessage(
                type=MessageType.ERROR,
                client_id=client.client_id,
                error_code="PROCESSING_ERROR",
                error_message=str(e)
            )
            await client.send_message(error_message)
    
    async def _get_detector_from_pool(self) -> Optional[ObjectDetector]:
        """🔧 Récupère un détecteur du pool"""
        
        async with self.detector_lock:
            if self.detector_pool:
                return self.detector_pool.pop()
            return None
    
    async def _return_detector_to_pool(self, detector: ObjectDetector):
        """🔧 Remet un détecteur dans le pool"""
        
        async with self.detector_lock:
            self.detector_pool.append(detector)
    
    async def _send_detection_results(
        self,
        client: StreamClient,
        detections: List[DetectionResult],
        lost_objects: List[LostObjectState],
        frame_info: Dict[str, Any]
    ):
        """📤 Envoie les résultats de détection au client"""
        
        # Calcul de la latence totale
        total_latency = time.time() - frame_info["timestamp"]
        
        # Message de détection
        detection_message = DetectionMessage(
            type=MessageType.DETECTION,
            client_id=client.client_id,
            detections=detections,
            lost_objects=lost_objects,
            processing_time_ms=total_latency * 1000,
            data={
                "frame_number": frame_info["number"],
                "total_latency_ms": total_latency * 1000,
                "objects_count": len(detections),
                "lost_objects_count": len(lost_objects)
            }
        )
        
        await client.send_message(detection_message)
        client.detections_sent += 1
    
    async def _cleanup_inactive_clients(self):
        """🧹 Nettoie les clients inactifs"""
        
        while True:
            try:
                with self.client_lock:
                    inactive_clients = [
                        client_id for client_id, client in self.clients.items()
                        if not client.is_healthy
                    ]
                
                for client_id in inactive_clients:
                    await self.disconnect_client(client_id)
                
                await asyncio.sleep(30)  # Vérification toutes les 30 secondes
                
            except Exception as e:
                logger.error(f"❌ Erreur nettoyage clients: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_service_health(self):
        """🏥 Monitore la santé du service"""
        
        while True:
            try:
                with self.client_lock:
                    client_count = len(self.clients)
                    healthy_clients = sum(1 for c in self.clients.values() if c.is_healthy)
                
                # Log périodique des statistiques
                if client_count > 0:
                    logger.info(
                        f"📊 StreamService: {healthy_clients}/{client_count} clients actifs, "
                        f"{self.total_frames_processed} frames traitées"
                    )
                
                await asyncio.sleep(60)  # Monitoring toutes les minutes
                
            except Exception as e:
                logger.error(f"❌ Erreur monitoring: {e}")
                await asyncio.sleep(60)
    
    async def send_ping_to_all_clients(self):
        """📡 Envoie un ping à tous les clients"""
        
        with self.client_lock:
            clients = list(self.clients.values())
        
        for client in clients:
            if client.is_active:
                ping_message = StatusMessage(
                    type=MessageType.STATUS,
                    client_id=client.client_id,
                    status="ping",
                    data={"timestamp": time.time()}
                )
                
                await client.send_message(ping_message)
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """📊 Statistiques globales du service"""
        
        with self.client_lock:
            active_clients = len(self.clients)
            client_stats = [client.get_statistics() for client in self.clients.values()]
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "service_uptime_seconds": uptime,
            "total_connections": self.total_connections,
            "active_clients": active_clients,
            "total_frames_processed": self.total_frames_processed,
            "frames_per_second": self.total_frames_processed / max(uptime, 1),
            "detector_pool_size": len(self.detector_pool),
            "client_statistics": client_stats
        }
    
    async def cleanup(self):
        """🧹 Nettoyage des ressources"""
        
        # Arrêter les tâches de background
        for task in self.background_tasks:
            task.cancel()
        
        # Déconnecter tous les clients
        with self.client_lock:
            client_ids = list(self.clients.keys())
        
        for client_id in client_ids:
            await self.disconnect_client(client_id)
        
        # Nettoyer le pool de détecteurs
        async with self.detector_lock:
            for detector in self.detector_pool:
                await detector.cleanup()
            self.detector_pool.clear()
        
        logger.info("🧹 StreamService nettoyé")

# === EXPORTS ===
__all__ = [
    "StreamService",
    "StreamClient",
    "FrameProcessor",
    "AlertManager"
]