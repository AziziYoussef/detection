# app/api/endpoints/stream_detection.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import json
import asyncio
import time
import logging
from typing import Dict, Optional, Set
from datetime import datetime
import uuid
import base64
from collections import deque

from app.schemas.detection import StreamFrame, StreamStatus, StreamConfig, LostObjectAlert
from app.utils.image_utils import decode_base64_to_image, encode_image_to_base64, validate_image
from app.core.model_manager import ModelManager

router = APIRouter()
logger = logging.getLogger(__name__)

class ConnectionManager:
    """Gestionnaire des connexions WebSocket"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.client_configs: Dict[str, StreamConfig] = {}
        self.client_stats: Dict[str, dict] = {}
        self.detection_buffers: Dict[str, deque] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accepte une nouvelle connexion"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_stats[client_id] = {
            'connected_since': datetime.now(),
            'frames_processed': 0,
            'alerts_generated': 0,
            'last_frame': None,
            'avg_fps': 0.0,
            'processing_times': deque(maxlen=30)
        }
        self.detection_buffers[client_id] = deque(maxlen=100)
        logger.info(f"🔗 Client {client_id} connecté au streaming")
    
    def disconnect(self, client_id: str):
        """Supprime une connexion"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.client_configs:
            del self.client_configs[client_id]
        if client_id in self.client_stats:
            del self.client_stats[client_id]
        if client_id in self.detection_buffers:
            del self.detection_buffers[client_id]
        logger.info(f"🔌 Client {client_id} déconnecté")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Envoie un message à un client spécifique"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Erreur envoi message à {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict):
        """Diffuse un message à tous les clients connectés"""
        disconnected = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                disconnected.append(client_id)
        
        # Nettoyer les connexions fermées
        for client_id in disconnected:
            self.disconnect(client_id)
    
    def get_active_clients(self) -> Set[str]:
        """Retourne la liste des clients actifs"""
        return set(self.active_connections.keys())
    
    def update_stats(self, client_id: str, processing_time: float):
        """Met à jour les statistiques d'un client"""
        if client_id in self.client_stats:
            stats = self.client_stats[client_id]
            stats['frames_processed'] += 1
            stats['last_frame'] = datetime.now()
            stats['processing_times'].append(processing_time)
            
            # Calcul FPS moyen
            if len(stats['processing_times']) >= 2:
                avg_time = sum(stats['processing_times']) / len(stats['processing_times'])
                stats['avg_fps'] = 1000 / avg_time if avg_time > 0 else 0

# Instance globale du gestionnaire
manager = ConnectionManager()

async def get_model_manager(request: Request) -> ModelManager:
    """Récupère le gestionnaire de modèles"""
    return request.app.state.model_manager

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    🎥 WebSocket pour streaming de détection en temps réel
    
    **Protocol:**
    - Connexion: Envoi automatique du statut
    - Envoi image: `{"type": "frame", "data": "base64_image", "config": {...}}`
    - Configuration: `{"type": "config", "config": {...}}`
    - Status: `{"type": "status"}` -> retourne les stats
    
    **Réponses:**
    - Detection: `{"type": "detection", "result": {...}}`
    - Alert: `{"type": "alert", "alert": {...}}`
    - Status: `{"type": "status", "data": {...}}`
    - Error: `{"type": "error", "message": "..."}`
    """
    
    # Validation de l'ID client
    if not client_id or len(client_id) < 3:
        await websocket.close(code=1008, reason="ID client invalide")
        return
    
    try:
        # Connexion
        await manager.connect(websocket, client_id)
        
        # Configuration par défaut
        default_config = StreamConfig()
        manager.client_configs[client_id] = default_config
        
        # Message de bienvenue
        await manager.send_personal_message({
            "type": "connected",
            "client_id": client_id,
            "message": f"🎥 Stream connecté pour {client_id}",
            "config": default_config.dict(),
            "timestamp": datetime.now().isoformat()
        }, client_id)
        
        # Récupération du model manager depuis l'app
        # Note: Ici on devrait passer le request, mais WebSocket ne l'a pas
        # On utilisera le modèle par défaut
        model_name = "stable_epoch_30"
        
        # Boucle de traitement des messages
        while True:
            try:
                # Attendre un message avec timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)
                
                message_type = message.get("type")
                
                if message_type == "frame":
                    # Traitement d'une frame
                    await process_frame(websocket, client_id, message, model_name)
                
                elif message_type == "config":
                    # Mise à jour de la configuration
                    await update_config(client_id, message.get("config", {}))
                
                elif message_type == "status":
                    # Demande de statut
                    await send_status(client_id)
                
                elif message_type == "ping":
                    # Ping/Pong pour maintenir la connexion
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }, client_id)
                
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Type de message inconnu: {message_type}"
                    }, client_id)
                    
            except asyncio.TimeoutError:
                # Ping automatique pour vérifier la connexion
                await manager.send_personal_message({
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                }, client_id)
                
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Format JSON invalide"
                }, client_id)
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} déconnecté normalement")
    except Exception as e:
        logger.error(f"Erreur WebSocket pour {client_id}: {e}")
    finally:
        manager.disconnect(client_id)

async def process_frame(websocket: WebSocket, client_id: str, message: dict, model_name: str):
    """Traite une frame reçue"""
    start_time = time.time()
    
    try:
        # Extraction des données
        image_data = message.get("data")
        frame_config = message.get("config", {})
        
        if not image_data:
            await manager.send_personal_message({
                "type": "error",
                "message": "Aucune donnée image fournie"
            }, client_id)
            return
        
        # Décodage de l'image
        try:
            image = decode_base64_to_image(image_data)
        except Exception as e:
            await manager.send_personal_message({
                "type": "error",
                "message": f"Erreur décodage image: {str(e)}"
            }, client_id)
            return
        
        if not validate_image(image):
            await manager.send_personal_message({
                "type": "error",
                "message": "Image invalide"
            }, client_id)
            return
        
        # Configuration de détection
        config = manager.client_configs[client_id]
        detection_config = {
            "confidence_threshold": frame_config.get("confidence_threshold", 0.5),
            "nms_threshold": frame_config.get("nms_threshold", 0.5),
            "max_detections": frame_config.get("max_detections", 30),  # Réduit pour le streaming
            "enable_tracking": True,
            "enable_lost_detection": True
        }
        
        # Simulation de détection (à remplacer par vraie détection)
        # Note: Ici on devrait utiliser le vrai model manager
        objects, persons, alerts = await simulate_detection(image, detection_config)
        
        # Création de la réponse
        processing_time = (time.time() - start_time) * 1000
        
        frame_result = {
            "frame_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "processing_time": processing_time,
            "objects": [obj.dict() for obj in objects],
            "persons": [person.dict() for person in persons],
            "total_objects": len(objects),
            "lost_objects": sum(1 for obj in objects if obj.status in ['lost', 'critical']),
            "suspect_objects": sum(1 for obj in objects if obj.status in ['suspect', 'surveillance'])
        }
        
        # Envoi du résultat
        await manager.send_personal_message({
            "type": "detection",
            "result": frame_result
        }, client_id)
        
        # Envoi des alertes si présentes
        for alert in alerts:
            await manager.send_personal_message({
                "type": "alert",
                "alert": alert.dict()
            }, client_id)
            manager.client_stats[client_id]['alerts_generated'] += 1
        
        # Mise à jour des stats
        manager.update_stats(client_id, processing_time)
        
        # Stockage dans le buffer pour analyse
        manager.detection_buffers[client_id].append({
            "timestamp": datetime.now(),
            "objects_count": len(objects),
            "processing_time": processing_time
        })
        
    except Exception as e:
        logger.error(f"Erreur traitement frame pour {client_id}: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": f"Erreur traitement: {str(e)}"
        }, client_id)

async def simulate_detection(image: np.ndarray, config: dict):
    """Simulation de détection (à remplacer par vraie détection)"""
    # Cette fonction est un placeholder
    # Dans la vraie implémentation, on utiliserait le ModelManager
    from app.schemas.detection import ObjectDetection, PersonDetection, BoundingBox, ObjectStatus
    
    # Simulation d'objets détectés
    objects = []
    persons = []
    alerts = []
    
    # Simulation simple
    h, w = image.shape[:2]
    
    # Exemple d'objet détecté
    if np.random.random() > 0.7:  # 30% de chance
        obj = ObjectDetection(
            object_id=str(uuid.uuid4()),
            class_name="backpack",
            class_name_fr="Sac à dos",
            confidence=0.75,
            confidence_level="high",
            bounding_box=BoundingBox(
                x=float(np.random.randint(0, w-100)),
                y=float(np.random.randint(0, h-100)),
                width=100.0,
                height=80.0
            ),
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            status=ObjectStatus.NORMAL
        )
        objects.append(obj)
    
    return objects, persons, alerts

async def update_config(client_id: str, new_config: dict):
    """Met à jour la configuration d'un client"""
    if client_id in manager.client_configs:
        current_config = manager.client_configs[client_id]
        
        # Mise à jour des champs fournis
        for key, value in new_config.items():
            if hasattr(current_config, key):
                setattr(current_config, key, value)
        
        await manager.send_personal_message({
            "type": "config_updated",
            "config": current_config.dict(),
            "message": "Configuration mise à jour"
        }, client_id)

async def send_status(client_id: str):
    """Envoie le statut d'un client"""
    if client_id in manager.client_stats:
        stats = manager.client_stats[client_id]
        
        status = StreamStatus(
            client_id=client_id,
            is_active=True,
            fps=stats['avg_fps'],
            frames_processed=stats['frames_processed'],
            alerts_generated=stats['alerts_generated'],
            connected_since=stats['connected_since'],
            last_frame=stats['last_frame']
        )
        
        await manager.send_personal_message({
            "type": "status",
            "data": status.dict()
        }, client_id)

@router.get("/status")
async def get_stream_status():
    """
    📊 Récupère le statut global du streaming
    
    Retourne les informations sur tous les clients connectés
    """
    active_clients = manager.get_active_clients()
    
    clients_status = []
    for client_id in active_clients:
        if client_id in manager.client_stats:
            stats = manager.client_stats[client_id]
            clients_status.append({
                "client_id": client_id,
                "connected_since": stats['connected_since'].isoformat(),
                "frames_processed": stats['frames_processed'],
                "alerts_generated": stats['alerts_generated'],
                "avg_fps": stats['avg_fps'],
                "last_frame": stats['last_frame'].isoformat() if stats['last_frame'] else None
            })
    
    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "total_connections": len(active_clients),
        "clients": clients_status
    }

@router.post("/broadcast")
async def broadcast_message(message: dict):
    """
    📢 Diffuse un message à tous les clients connectés
    
    Utile pour les alertes globales ou les maintenances
    """
    try:
        broadcast_msg = {
            "type": "broadcast",
            "timestamp": datetime.now().isoformat(),
            "message": message.get("message", "Message de diffusion"),
            "data": message.get("data", {})
        }
        
        await manager.broadcast(broadcast_msg)
        
        return {
            "success": True,
            "message": "Message diffusé",
            "clients_count": len(manager.get_active_clients())
        }
        
    except Exception as e:
        logger.error(f"Erreur diffusion: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur diffusion: {str(e)}")

@router.get("/demo")
async def stream_demo():
    """
    🎬 Page de démonstration du streaming
    
    Interface simple pour tester le WebSocket
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🎥 Demo Stream IA</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
            .controls { margin: 20px 0; }
            .status { background: #e8f5e8; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .alert { background: #ffe8e8; padding: 10px; margin: 10px 0; border-radius: 5px; }
            button { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
            .btn-primary { background: #007bff; color: white; }
            .btn-success { background: #28a745; color: white; }
            .btn-danger { background: #dc3545; color: white; }
            #video { border: 2px solid #ddd; border-radius: 10px; }
            #results { height: 300px; overflow-y: auto; background: #f8f9fa; padding: 10px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 Démonstration Streaming IA</h1>
            
            <div class="controls">
                <button id="connectBtn" class="btn-primary">🔗 Connecter</button>
                <button id="disconnectBtn" class="btn-danger" disabled>🔌 Déconnecter</button>
                <button id="startVideo" class="btn-success" disabled>📹 Démarrer Caméra</button>
                <button id="stopVideo" class="btn-danger" disabled>⏹ Arrêter Caméra</button>
            </div>
            
            <div class="status">
                <strong>État:</strong> <span id="status">Déconnecté</span>
            </div>
            
            <div style="display: flex; gap: 20px;">
                <div style="flex: 1;">
                    <h3>📹 Vidéo</h3>
                    <video id="video" width="400" height="300" autoplay muted></video>
                    <canvas id="canvas" style="display: none;"></canvas>
                </div>
                
                <div style="flex: 1;">
                    <h3>🔍 Résultats de Détection</h3>
                    <div id="results"></div>
                </div>
            </div>
        </div>
        
        <script>
            const clientId = 'demo_' + Math.random().toString(36).substr(2, 9);
            let ws = null;
            let video = null;
            let canvas = null;
            let ctx = null;
            let stream = null;
            let intervalId = null;
            
            document.addEventListener('DOMContentLoaded', function() {
                video = document.getElementById('video');
                canvas = document.getElementById('canvas');
                ctx = canvas.getContext('2d');
                
                document.getElementById('connectBtn').onclick = connect;
                document.getElementById('disconnectBtn').onclick = disconnect;
                document.getElementById('startVideo').onclick = startVideo;
                document.getElementById('stopVideo').onclick = stopVideo;
            });
            
            function connect() {
                const wsUrl = `ws://localhost:8000/api/v1/stream/ws/${clientId}`;
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function() {
                    updateStatus('🟢 Connecté', 'green');
                    document.getElementById('connectBtn').disabled = true;
                    document.getElementById('disconnectBtn').disabled = false;
                    document.getElementById('startVideo').disabled = false;
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                };
                
                ws.onclose = function() {
                    updateStatus('🔴 Déconnecté', 'red');
                    document.getElementById('connectBtn').disabled = false;
                    document.getElementById('disconnectBtn').disabled = true;
                    document.getElementById('startVideo').disabled = true;
                    document.getElementById('stopVideo').disabled = true;
                };
                
                ws.onerror = function(error) {
                    console.error('Erreur WebSocket:', error);
                    updateStatus('❌ Erreur de connexion', 'red');
                };
            }
            
            function disconnect() {
                if (ws) {
                    ws.close();
                }
                stopVideo();
            }
            
            function startVideo() {
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(function(s) {
                        stream = s;
                        video.srcObject = stream;
                        document.getElementById('startVideo').disabled = true;
                        document.getElementById('stopVideo').disabled = false;
                        
                        // Démarrer l'envoi de frames
                        intervalId = setInterval(captureAndSend, 500); // 2 FPS
                    })
                    .catch(function(err) {
                        console.error('Erreur caméra:', err);
                        alert('Impossible d\\'accéder à la caméra');
                    });
            }
            
            function stopVideo() {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                    video.srcObject = null;
                }
                
                if (intervalId) {
                    clearInterval(intervalId);
                    intervalId = null;
                }
                
                document.getElementById('startVideo').disabled = false;
                document.getElementById('stopVideo').disabled = true;
            }
            
            function captureAndSend() {
                if (!ws || ws.readyState !== WebSocket.OPEN || !video.videoWidth) {
                    return;
                }
                
                // Capturer la frame
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0);
                
                // Convertir en base64
                const imageData = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
                
                // Envoyer au serveur
                const message = {
                    type: 'frame',
                    data: imageData,
                    config: {
                        confidence_threshold: 0.5,
                        max_detections: 20
                    }
                };
                
                ws.send(JSON.stringify(message));
            }
            
            function handleMessage(data) {
                const results = document.getElementById('results');
                const timestamp = new Date().toLocaleTimeString();
                
                if (data.type === 'connected') {
                    addResult(`✅ ${timestamp}: ${data.message}`, 'info');
                }
                else if (data.type === 'detection') {
                    const result = data.result;
                    addResult(`🔍 ${timestamp}: ${result.total_objects} objets, ${result.lost_objects} perdus (${result.processing_time.toFixed(1)}ms)`, 'success');
                }
                else if (data.type === 'alert') {
                    const alert = data.alert;
                    addResult(`🚨 ${timestamp}: ALERTE - ${alert.message}`, 'alert');
                }
                else if (data.type === 'error') {
                    addResult(`❌ ${timestamp}: ${data.message}`, 'error');
                }
            }
            
            function addResult(text, type) {
                const results = document.getElementById('results');
                const div = document.createElement('div');
                div.textContent = text;
                div.style.margin = '5px 0';
                div.style.padding = '5px';
                div.style.borderRadius = '3px';
                
                if (type === 'alert') {
                    div.style.backgroundColor = '#ffe8e8';
                    div.style.color = '#d63384';
                } else if (type === 'success') {
                    div.style.backgroundColor = '#e8f5e8';
                    div.style.color = '#198754';
                } else if (type === 'error') {
                    div.style.backgroundColor = '#f8d7da';
                    div.style.color = '#dc3545';
                } else {
                    div.style.backgroundColor = '#e7f3ff';
                    div.style.color = '#0066cc';
                }
                
                results.appendChild(div);
                results.scrollTop = results.scrollHeight;
                
                // Garder seulement les 50 derniers messages
                while (results.children.length > 50) {
                    results.removeChild(results.firstChild);
                }
            }
            
            function updateStatus(text, color) {
                const status = document.getElementById('status');
                status.textContent = text;
                status.style.color = color;
            }
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)