"""
📡 SERVICE DE STREAMING TEMPS RÉEL
==================================
Endpoint spécialisé pour la détection en temps réel via WebSocket

Fonctionnalités:
- WebSocket pour réception frames webcam/caméra en temps réel
- Détection ultra-rapide avec vos modèles optimisés
- Gestion connexions multiples simultanées
- Latence minimale (<100ms par frame)
- Adaptation automatique qualité/vitesse selon charge
- Statistiques temps réel (FPS, détections/sec)

Intégration:
- Next.js: Interface webcam → WebSocket → Résultats temps réel
- Spring Boot: Monitoring connexions et statistiques
- GPU optimisé pour inférence ultra-rapide
"""

import asyncio
import time
import json
import uuid
import logging
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import base64
import io

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

# Services internes
from app.services.stream_service import StreamDetectionService
from app.schemas.detection import DetectionResult, StreamFrameResult
from app.config.config import get_settings

logger = logging.getLogger(__name__)

# 🛣️ CRÉATION DU ROUTER
router = APIRouter()

# 📋 ÉNUMÉRATIONS ET SCHÉMAS
class StreamQuality(str, Enum):
    """🎚️ Qualités de streaming"""
    ULTRA_LOW = "ultra_low"    # 160x120 - Ultra rapide
    LOW = "low"                # 320x240 - Rapide  
    MEDIUM = "medium"          # 640x480 - Équilibré
    HIGH = "high"              # 1280x720 - Haute qualité

class StreamMode(str, Enum):
    """🎮 Modes de streaming"""
    REALTIME = "realtime"      # Temps réel max (skip frames si nécessaire)
    QUALITY = "quality"        # Toutes les frames (peut être plus lent)
    ADAPTIVE = "adaptive"      # Adaptation automatique selon charge

class StreamClient(BaseModel):
    """👤 Client de streaming connecté"""
    client_id: str
    websocket: WebSocket
    connected_at: float
    last_frame_time: float
    frames_processed: int
    detections_count: int
    current_fps: float
    stream_quality: StreamQuality
    stream_mode: StreamMode
    model_name: str
    confidence_threshold: float

class StreamStats(BaseModel):
    """📊 Statistiques de streaming"""
    total_clients: int
    active_connections: int
    frames_per_second: float
    detections_per_second: float
    average_latency_ms: float
    gpu_usage_percent: float
    memory_usage_mb: float

# 🗃️ GESTION DES CLIENTS CONNECTÉS
connected_clients: Dict[str, StreamClient] = {}
stream_stats = StreamStats(
    total_clients=0,
    active_connections=0,
    frames_per_second=0.0,
    detections_per_second=0.0,
    average_latency_ms=0.0,
    gpu_usage_percent=0.0,
    memory_usage_mb=0.0
)

# 🔧 DÉPENDANCES
async def get_stream_service() -> StreamDetectionService:
    """Récupère le service de streaming"""
    try:
        from main import get_model_service
        model_service = get_model_service()
        return StreamDetectionService(model_service)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service streaming non disponible: {e}")

# 📡 ENDPOINT WEBSOCKET PRINCIPAL
@router.websocket("/live")
async def websocket_stream_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = Query(default=None),
    model_name: str = Query(default="epoch_30"),
    confidence_threshold: float = Query(default=0.5, ge=0.1, le=0.9),
    stream_quality: StreamQuality = Query(default=StreamQuality.MEDIUM),
    stream_mode: StreamMode = Query(default=StreamMode.ADAPTIVE)
):
    """
    📡 WebSocket pour streaming temps réel
    
    **Protocole de communication:**
    
    **Client → Serveur:**
    ```json
    {
        "type": "frame",
        "data": "base64_encoded_image",
        "timestamp": 1234567890.123,
        "frame_id": "unique_frame_id"
    }
    ```
    
    **Serveur → Client:**
    ```json
    {
        "type": "detections",
        "frame_id": "unique_frame_id", 
        "detections": [...],
        "processing_time_ms": 45.2,
        "fps": 18.5,
        "timestamp": 1234567890.456
    }
    ```
    
    **Commandes de contrôle:**
    ```json
    {
        "type": "command",
        "action": "pause|resume|change_quality|change_threshold",
        "params": {...}
    }
    ```
    """
    
    # Génération ID client si non fourni
    if not client_id:
        client_id = f"client_{uuid.uuid4().hex[:8]}"
    
    logger.info(f"📡 Nouvelle connexion stream: {client_id}")
    
    # Obtenir le service de streaming
    stream_service = await get_stream_service()
    
    try:
        # Accepter la connexion WebSocket
        await websocket.accept()
        
        # Créer le client
        client = StreamClient(
            client_id=client_id,
            websocket=websocket,
            connected_at=time.time(),
            last_frame_time=time.time(),
            frames_processed=0,
            detections_count=0,
            current_fps=0.0,
            stream_quality=stream_quality,
            stream_mode=stream_mode,
            model_name=model_name,
            confidence_threshold=confidence_threshold
        )
        
        # Enregistrer le client
        connected_clients[client_id] = client
        stream_stats.total_clients += 1
        stream_stats.active_connections = len(connected_clients)
        
        # Message de bienvenue
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "message": f"Connexion établie - Modèle: {model_name}",
            "server_info": {
                "classes_count": len(get_settings().DETECTION_CLASSES),
                "classes": get_settings().CLASSES_FR_NAMES,
                "model_performance": "F1=49.86%, Précision=60.73%"
            }
        })
        
        # Boucle principale de traitement
        frame_buffer = []
        last_stats_update = time.time()
        
        while True:
            try:
                # Réception du message
                message = await websocket.receive_json()
                
                # Traitement selon le type de message
                if message.get("type") == "frame":
                    await handle_frame_message(
                        client, message, stream_service, frame_buffer
                    )
                    
                elif message.get("type") == "command":
                    await handle_command_message(client, message)
                    
                elif message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": time.time()
                    })
                
                # Mise à jour statistiques périodique
                if time.time() - last_stats_update > 5.0:  # Toutes les 5 secondes
                    await update_stream_statistics()
                    await send_stats_to_client(client)
                    last_stats_update = time.time()
                    
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ Timeout client {client_id}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"📡 Déconnexion normale: {client_id}")
        
    except Exception as e:
        logger.error(f"❌ Erreur WebSocket {client_id}: {e}", exc_info=True)
        
    finally:
        # Nettoyage
        if client_id in connected_clients:
            del connected_clients[client_id]
            stream_stats.active_connections = len(connected_clients)
            
        logger.info(f"📡 Client {client_id} déconnecté")

async def handle_frame_message(
    client: StreamClient,
    message: Dict[str, Any],
    stream_service: StreamDetectionService,
    frame_buffer: List[Dict]
):
    """🖼️ Traite un message de frame reçu"""
    
    start_time = time.time()
    
    try:
        # Extraction des données de la frame
        frame_data = message.get("data")
        frame_id = message.get("frame_id", f"frame_{int(time.time() * 1000)}")
        client_timestamp = message.get("timestamp", time.time())
        
        if not frame_data:
            await client.websocket.send_json({
                "type": "error",
                "message": "Données de frame manquantes",
                "frame_id": frame_id
            })
            return
        
        # Décodage base64 → PIL Image
        try:
            image_bytes = base64.b64decode(frame_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Conversion RGB si nécessaire
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
        except Exception as e:
            await client.websocket.send_json({
                "type": "error",
                "message": f"Erreur décodage image: {e}",
                "frame_id": frame_id
            })
            return
        
        # Gestion du mode adaptatif
        if client.stream_mode == StreamMode.ADAPTIVE:
            # Skip frames si on accumule du retard
            if len(frame_buffer) > 3:
                frame_buffer.clear()  # Vider le buffer
                logger.debug(f"🔄 Buffer vidé pour {client.client_id} (mode adaptatif)")
        
        # Mode temps réel strict
        elif client.stream_mode == StreamMode.REALTIME:
            # Traitement immédiat, skip si occupation
            if len(frame_buffer) > 1:
                frame_buffer.pop(0)  # Retirer la plus ancienne
        
        # Ajout au buffer
        frame_buffer.append({
            "image": image,
            "frame_id": frame_id,
            "timestamp": client_timestamp,
            "client_id": client.client_id
        })
        
        # Traitement de la frame
        if frame_buffer:
            frame_to_process = frame_buffer.pop(0)
            
            # Détection avec le modèle
            detections = await stream_service.detect_objects_realtime(
                image=frame_to_process["image"],
                confidence_threshold=client.confidence_threshold,
                model_name=client.model_name,
                quality=client.stream_quality
            )
            
            # Mise à jour statistiques client
            client.frames_processed += 1
            client.detections_count += len(detections)
            client.last_frame_time = time.time()
            
            # Calcul FPS
            processing_time = (time.time() - start_time) * 1000
            time_diff = time.time() - client_timestamp
            client.current_fps = 1.0 / time_diff if time_diff > 0 else 0.0
            
            # Envoi des résultats
            await client.websocket.send_json({
                "type": "detections",
                "frame_id": frame_to_process["frame_id"],
                "detections": [det.dict() for det in detections],
                "processing_time_ms": processing_time,
                "fps": client.current_fps,
                "timestamp": time.time(),
                "stats": {
                    "frames_processed": client.frames_processed,
                    "total_detections": client.detections_count,
                    "buffer_size": len(frame_buffer)
                }
            })
            
        # Limite de frames par seconde (si mode quality)
        if client.stream_mode == StreamMode.QUALITY:
            await asyncio.sleep(0.033)  # ~30 FPS max
            
    except Exception as e:
        logger.error(f"❌ Erreur traitement frame {client.client_id}: {e}")
        
        await client.websocket.send_json({
            "type": "error",
            "message": f"Erreur traitement: {e}",
            "frame_id": frame_id
        })

async def handle_command_message(client: StreamClient, message: Dict[str, Any]):
    """🎮 Traite un message de commande"""
    
    action = message.get("action")
    params = message.get("params", {})
    
    try:
        if action == "change_threshold":
            new_threshold = params.get("threshold", 0.5)
            if 0.1 <= new_threshold <= 0.9:
                client.confidence_threshold = new_threshold
                await client.websocket.send_json({
                    "type": "command_result",
                    "action": action,
                    "status": "success",
                    "message": f"Seuil changé à {new_threshold}"
                })
            else:
                raise ValueError("Seuil invalide")
                
        elif action == "change_quality":
            new_quality = params.get("quality")
            if new_quality in [q.value for q in StreamQuality]:
                client.stream_quality = StreamQuality(new_quality)
                await client.websocket.send_json({
                    "type": "command_result",
                    "action": action,
                    "status": "success",
                    "message": f"Qualité changée à {new_quality}"
                })
            else:
                raise ValueError("Qualité invalide")
                
        elif action == "change_mode":
            new_mode = params.get("mode")
            if new_mode in [m.value for m in StreamMode]:
                client.stream_mode = StreamMode(new_mode)
                await client.websocket.send_json({
                    "type": "command_result",
                    "action": action,
                    "status": "success",
                    "message": f"Mode changé à {new_mode}"
                })
            else:
                raise ValueError("Mode invalide")
                
        elif action == "get_stats":
            await send_detailed_stats_to_client(client)
            
        else:
            raise ValueError(f"Action inconnue: {action}")
            
    except Exception as e:
        await client.websocket.send_json({
            "type": "command_result",
            "action": action,
            "status": "error",
            "message": str(e)
        })

async def update_stream_statistics():
    """📊 Met à jour les statistiques globales de streaming"""
    
    if not connected_clients:
        return
    
    # Calcul moyennes
    total_fps = sum(client.current_fps for client in connected_clients.values())
    avg_fps = total_fps / len(connected_clients) if connected_clients else 0.0
    
    total_detections = sum(client.detections_count for client in connected_clients.values())
    
    # Mise à jour des stats globales
    stream_stats.frames_per_second = avg_fps
    stream_stats.detections_per_second = total_detections / max(1, time.time() - min(
        client.connected_at for client in connected_clients.values()
    ))
    
    # Utilisation GPU/mémoire (approximative)
    try:
        import torch
        if torch.cuda.is_available():
            stream_stats.gpu_usage_percent = torch.cuda.utilization()
            stream_stats.memory_usage_mb = torch.cuda.memory_allocated() / 1024 / 1024
    except:
        pass

async def send_stats_to_client(client: StreamClient):
    """📊 Envoie les statistiques à un client"""
    
    await client.websocket.send_json({
        "type": "stats",
        "client_stats": {
            "frames_processed": client.frames_processed,
            "detections_count": client.detections_count,
            "current_fps": client.current_fps,
            "connected_duration": time.time() - client.connected_at
        },
        "server_stats": stream_stats.dict()
    })

async def send_detailed_stats_to_client(client: StreamClient):
    """📊 Envoie les statistiques détaillées à un client"""
    
    await client.websocket.send_json({
        "type": "detailed_stats",
        "client_info": client.dict(exclude={"websocket"}),
        "server_info": stream_stats.dict(),
        "all_clients": [
            {
                "client_id": c.client_id,
                "connected_at": c.connected_at,
                "frames_processed": c.frames_processed,
                "current_fps": c.current_fps
            }
            for c in connected_clients.values()
        ]
    })

# 📊 ENDPOINT REST - STATISTIQUES STREAMING
@router.get(
    "/statistics",
    summary="📊 Statistiques streaming",
    description="""
    ## 📊 Statistiques globales du streaming temps réel
    
    **Informations incluses:**
    - Nombre de clients connectés
    - FPS moyen et détections/seconde
    - Latence moyenne et utilisation ressources
    - Détails par client connecté
    
    **Utilisé par:** Spring Boot pour monitoring, Next.js pour dashboard admin
    """
)
async def get_streaming_statistics():
    """📊 Récupère les statistiques de streaming"""
    
    await update_stream_statistics()
    
    return {
        "success": True,
        "statistics": stream_stats.dict(),
        "connected_clients": [
            {
                "client_id": client.client_id,
                "connected_at": client.connected_at,
                "frames_processed": client.frames_processed,
                "detections_count": client.detections_count,
                "current_fps": client.current_fps,
                "stream_quality": client.stream_quality,
                "stream_mode": client.stream_mode,
                "model_name": client.model_name
            }
            for client in connected_clients.values()
        ],
        "timestamp": time.time()
    }

# 🔧 ENDPOINT REST - CONFIGURATION STREAMING
@router.get(
    "/config",
    summary="🔧 Configuration streaming",
    description="Configuration et limites du service de streaming"
)
async def get_streaming_config():
    """🔧 Récupère la configuration du streaming"""
    
    settings = get_settings()
    
    return {
        "success": True,
        "config": {
            "max_concurrent_clients": settings.MAX_STREAM_CLIENTS,
            "supported_qualities": [q.value for q in StreamQuality],
            "supported_modes": [m.value for m in StreamMode],
            "default_model": "epoch_30",
            "available_models": ["epoch_30", "extended", "fast"],
            "confidence_range": {"min": 0.1, "max": 0.9},
            "max_fps": 30,
            "buffer_size_limit": 5
        }
    }

# 🛑 ENDPOINT REST - DÉCONNEXION CLIENT
@router.delete(
    "/disconnect/{client_id}",
    summary="🛑 Déconnecter client",
    description="Force la déconnexion d'un client (admin uniquement)"
)
async def disconnect_client(client_id: str):
    """🛑 Force la déconnexion d'un client"""
    
    if client_id not in connected_clients:
        raise HTTPException(status_code=404, detail="Client non trouvé")
    
    client = connected_clients[client_id]
    
    try:
        await client.websocket.close()
        del connected_clients[client_id]
        stream_stats.active_connections = len(connected_clients)
        
        return {
            "success": True,
            "message": f"Client {client_id} déconnecté",
            "client_id": client_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur déconnexion: {e}")

# 📝 INFORMATIONS D'EXPORT
__all__ = ["router"]