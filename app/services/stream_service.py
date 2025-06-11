# app/services/stream_service.py
import asyncio
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from collections import deque
import threading
import queue
import time

from app.schemas.detection import StreamFrame, StreamConfig, LostObjectAlert
from app.core.detector import ObjectDetector
from app.utils.image_utils import validate_image, encode_image_to_base64

logger = logging.getLogger(__name__)

class StreamBuffer:
    """Buffer circulaire pour les frames de streaming"""
    
    def __init__(self, max_size: int = 30):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def add_frame(self, frame_data: dict):
        """Ajoute une frame au buffer"""
        with self.lock:
            self.buffer.append(frame_data)
    
    def get_recent_frames(self, count: int = 10) -> List[dict]:
        """Récupère les N dernières frames"""
        with self.lock:
            return list(self.buffer)[-count:]
    
    def get_frame_history(self, seconds: int = 60) -> List[dict]:
        """Récupère l'historique des N dernières secondes"""
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        with self.lock:
            return [frame for frame in self.buffer 
                   if frame.get('timestamp', datetime.min) > cutoff_time]

class StreamAnalyzer:
    """Analyseur de tendances pour les streams"""
    
    def __init__(self):
        self.object_history = deque(maxlen=1000)
        self.alert_history = deque(maxlen=200)
        self.performance_history = deque(maxlen=100)
    
    def add_detection_result(self, objects: List, persons: List, alerts: List, processing_time: float):
        """Ajoute un résultat de détection à l'analyse"""
        timestamp = datetime.now()
        
        # Historique des objets
        self.object_history.append({
            'timestamp': timestamp,
            'objects_count': len(objects),
            'persons_count': len(persons),
            'lost_objects': sum(1 for obj in objects if obj.status.value in ['lost', 'critical']),
            'suspect_objects': sum(1 for obj in objects if obj.status.value in ['suspect', 'surveillance'])
        })
        
        # Historique des alertes
        for alert in alerts:
            self.alert_history.append({
                'timestamp': timestamp,
                'alert_level': alert.alert_level,
                'object_class': alert.object_detection.class_name,
                'message': alert.message
            })
        
        # Performance
        self.performance_history.append({
            'timestamp': timestamp,
            'processing_time': processing_time
        })
    
    def get_trends(self, minutes: int = 5) -> dict:
        """Analyse les tendances des N dernières minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        # Filtrer les données récentes
        recent_objects = [h for h in self.object_history if h['timestamp'] > cutoff]
        recent_alerts = [h for h in self.alert_history if h['timestamp'] > cutoff]
        recent_performance = [h for h in self.performance_history if h['timestamp'] > cutoff]
        
        if not recent_objects:
            return {"status": "no_data", "message": "Pas assez de données pour l'analyse"}
        
        # Calculs de tendances
        avg_objects = sum(h['objects_count'] for h in recent_objects) / len(recent_objects)
        avg_lost = sum(h['lost_objects'] for h in recent_objects) / len(recent_objects)
        avg_processing = sum(h['processing_time'] for h in recent_performance) / len(recent_performance) if recent_performance else 0
        
        # Détection de pics
        object_counts = [h['objects_count'] for h in recent_objects]
        has_object_spike = max(object_counts) > avg_objects * 2 if object_counts else False
        
        alert_count = len(recent_alerts)
        alert_rate = alert_count / minutes  # alertes par minute
        
        # Classification de l'activité
        activity_level = "normal"
        if alert_rate > 2:
            activity_level = "high"
        elif alert_rate > 0.5:
            activity_level = "moderate"
        elif avg_objects > 5:
            activity_level = "busy"
        
        return {
            "period_minutes": minutes,
            "activity_level": activity_level,
            "statistics": {
                "avg_objects_per_frame": avg_objects,
                "avg_lost_objects": avg_lost,
                "total_alerts": alert_count,
                "alert_rate_per_minute": alert_rate,
                "avg_processing_time_ms": avg_processing,
                "frames_analyzed": len(recent_objects)
            },
            "flags": {
                "object_spike_detected": has_object_spike,
                "high_alert_rate": alert_rate > 1,
                "performance_issue": avg_processing > 500
            }
        }

class StreamProcessor:
    """Processeur de stream temps réel"""
    
    def __init__(self, detector: ObjectDetector, config: StreamConfig):
        self.detector = detector
        self.config = config
        self.buffer = StreamBuffer(config.buffer_size)
        self.analyzer = StreamAnalyzer()
        
        # État du processeur
        self.is_running = False
        self.stats = {
            'frames_processed': 0,
            'alerts_generated': 0,
            'start_time': None,
            'last_frame_time': None
        }
        
        # Queue pour le traitement asynchrone
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_callbacks: List[Callable] = []
        
        # Thread de traitement
        self.processing_thread = None
    
    def add_result_callback(self, callback: Callable):
        """Ajoute un callback pour les résultats"""
        self.result_callbacks.append(callback)
    
    def start(self):
        """Démarre le processeur"""
        if self.is_running:
            return
        
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        # Lancement du thread de traitement
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("StreamProcessor démarré")
    
    def stop(self):
        """Arrête le processeur"""
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        
        logger.info("StreamProcessor arrêté")
    
    def process_frame(self, frame: np.ndarray, frame_metadata: dict = None) -> bool:
        """
        Ajoute une frame à traiter
        
        Returns:
            True si la frame a été ajoutée, False si la queue est pleine
        """
        if not self.is_running:
            return False
        
        try:
            frame_data = {
                'frame': frame,
                'metadata': frame_metadata or {},
                'timestamp': datetime.now()
            }
            
            self.frame_queue.put_nowait(frame_data)
            return True
            
        except queue.Full:
            logger.warning("Queue de traitement pleine, frame ignorée")
            return False
    
    def _processing_loop(self):
        """Boucle principale de traitement"""
        while self.is_running:
            try:
                # Attendre une frame avec timeout
                frame_data = self.frame_queue.get(timeout=1.0)
                
                # Traitement de la frame
                result = self._process_single_frame(frame_data)
                
                # Callbacks
                for callback in self.result_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Erreur callback: {e}")
                
                # Mise à jour des stats
                self.stats['frames_processed'] += 1
                self.stats['last_frame_time'] = datetime.now()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Erreur traitement frame: {e}")
    
    def _process_single_frame(self, frame_data: dict) -> dict:
        """Traite une frame individuelle"""
        start_time = time.time()
        frame = frame_data['frame']
        metadata = frame_data['metadata']
        timestamp = frame_data['timestamp']
        
        try:
            # Validation
            if not validate_image(frame):
                raise ValueError("Frame invalide")
            
            # Détection
            objects, persons, alerts = self.detector.detect(
                frame,
                confidence_threshold=metadata.get('confidence_threshold', 0.5),
                enable_tracking=True,
                enable_lost_detection=True
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Ajout aux analyses
            self.analyzer.add_detection_result(objects, persons, alerts, processing_time)
            
            # Création du résultat
            result = {
                'success': True,
                'timestamp': timestamp,
                'processing_time': processing_time,
                'frame_size': frame.shape,
                'objects': [obj.dict() for obj in objects],
                'persons': [person.dict() for person in persons],
                'alerts': [alert.dict() for alert in alerts],
                'statistics': {
                    'objects_count': len(objects),
                    'persons_count': len(persons),
                    'alerts_count': len(alerts),
                    'lost_objects': sum(1 for obj in objects if obj.status.value in ['lost', 'critical']),
                    'suspect_objects': sum(1 for obj in objects if obj.status.value in ['suspect', 'surveillance'])
                }
            }
            
            # Ajout au buffer
            self.buffer.add_frame(result)
            
            # Mise à jour stats alertes
            if alerts:
                self.stats['alerts_generated'] += len(alerts)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur traitement frame: {e}")
            return {
                'success': False,
                'timestamp': timestamp,
                'error': str(e),
                'processing_time': (time.time() - start_time) * 1000
            }
    
    def get_stats(self) -> dict:
        """Récupère les statistiques du processeur"""
        stats = self.stats.copy()
        
        # Calculs additionnels
        if stats['start_time']:
            uptime = (datetime.now() - stats['start_time']).total_seconds()
            stats['uptime_seconds'] = uptime
            
            if stats['frames_processed'] > 0:
                stats['avg_fps'] = stats['frames_processed'] / uptime
        
        # Tendances récentes
        stats['trends'] = self.analyzer.get_trends(minutes=5)
        
        return stats
    
    def get_recent_frames(self, count: int = 10) -> List[dict]:
        """Récupère les frames récentes"""
        return self.buffer.get_recent_frames(count)

class StreamManager:
    """Gestionnaire global des streams"""
    
    def __init__(self):
        self.active_streams: Dict[str, StreamProcessor] = {}
        self.stream_configs: Dict[str, StreamConfig] = {}
    
    def create_stream(self, stream_id: str, detector: ObjectDetector, 
                     config: StreamConfig = None) -> StreamProcessor:
        """Crée un nouveau stream"""
        if stream_id in self.active_streams:
            raise ValueError(f"Stream {stream_id} déjà actif")
        
        config = config or StreamConfig()
        processor = StreamProcessor(detector, config)
        
        self.active_streams[stream_id] = processor
        self.stream_configs[stream_id] = config
        
        logger.info(f"Stream {stream_id} créé")
        return processor
    
    def get_stream(self, stream_id: str) -> Optional[StreamProcessor]:
        """Récupère un stream existant"""
        return self.active_streams.get(stream_id)
    
    def stop_stream(self, stream_id: str):
        """Arrête et supprime un stream"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id].stop()
            del self.active_streams[stream_id]
            
        if stream_id in self.stream_configs:
            del self.stream_configs[stream_id]
        
        logger.info(f"Stream {stream_id} arrêté")
    
    def list_streams(self) -> Dict[str, dict]:
        """Liste tous les streams actifs"""
        streams_info = {}
        
        for stream_id, processor in self.active_streams.items():
            streams_info[stream_id] = {
                'stream_id': stream_id,
                'is_running': processor.is_running,
                'config': self.stream_configs.get(stream_id, {}).dict() if stream_id in self.stream_configs else {},
                'stats': processor.get_stats()
            }
        
        return streams_info
    
    def get_global_stats(self) -> dict:
        """Statistiques globales de tous les streams"""
        total_streams = len(self.active_streams)
        active_streams = sum(1 for p in self.active_streams.values() if p.is_running)
        
        total_frames = sum(p.stats['frames_processed'] for p in self.active_streams.values())
        total_alerts = sum(p.stats['alerts_generated'] for p in self.active_streams.values())
        
        return {
            'total_streams': total_streams,
            'active_streams': active_streams,
            'total_frames_processed': total_frames,
            'total_alerts_generated': total_alerts,
            'streams': self.list_streams()
        }
    
    def cleanup_inactive_streams(self, max_inactive_minutes: int = 30):
        """Nettoie les streams inactifs"""
        cutoff_time = datetime.now() - timedelta(minutes=max_inactive_minutes)
        to_remove = []
        
        for stream_id, processor in self.active_streams.items():
            last_frame = processor.stats.get('last_frame_time')
            if last_frame and last_frame < cutoff_time:
                to_remove.append(stream_id)
        
        for stream_id in to_remove:
            logger.info(f"Nettoyage stream inactif: {stream_id}")
            self.stop_stream(stream_id)
        
        return len(to_remove)

# Instance globale du gestionnaire
stream_manager = StreamManager()