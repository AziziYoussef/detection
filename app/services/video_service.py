"""
🎬 VIDEO SERVICE - SERVICE DE TRAITEMENT VIDÉO
============================================
Service spécialisé pour l'analyse vidéo frame par frame

Fonctionnalités:
- Upload et validation de fichiers vidéo
- Traitement frame par frame avec détection d'objets
- Tracking temporel avancé des objets perdus
- Génération de timeline des événements
- Export vidéo annotée avec détections
- Optimisation pour traitement asynchrone
- Support formats multiples (MP4, AVI, MOV, etc.)
"""

import asyncio
import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from pathlib import Path
import uuid
from datetime import datetime, timedelta
import json

import cv2
import numpy as np
from PIL import Image
import torch

# Imports internes
from app.core.model_manager import ModelManager
from app.core.detector import ObjectDetector, DetectionConfig
from app.core.preprocessing import ImagePreprocessor
from app.core.postprocessing import ResultPostprocessor, LostObjectDetector
from app.schemas.detection import (
    DetectionResult, DetectionRequest, DetectionResponse,
    LostObjectState, DetectionMode, ModelType, ObjectStatus
)
from app.config.config import Settings

logger = logging.getLogger(__name__)

class VideoValidationError(Exception):
    """❌ Erreur de validation vidéo"""
    pass

class VideoInfo:
    """📹 Informations sur une vidéo"""
    
    def __init__(self, video_path: str):
        self.path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            raise VideoValidationError(f"Impossible d'ouvrir la vidéo: {video_path}")
        
        # Propriétés de la vidéo
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        # Format et codec
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
    def get_info_dict(self) -> Dict[str, Any]:
        """📋 Retourne les informations sous forme de dictionnaire"""
        return {
            "filename": self.path.name,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration,
            "codec": self.codec,
            "size_mb": self.path.stat().st_size / (1024 * 1024) if self.path.exists() else 0
        }
    
    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

class VideoFrameExtractor:
    """🎞️ Extracteur de frames optimisé"""
    
    def __init__(self, video_path: str, target_fps: Optional[float] = None):
        self.video_info = VideoInfo(video_path)
        self.target_fps = target_fps or min(self.video_info.fps, 5.0)  # Max 5 FPS par défaut
        
        # Calcul de l'intervalle de frames
        self.frame_interval = max(1, int(self.video_info.fps / self.target_fps))
        
        logger.info(
            f"🎞️ Extracteur configuré: {self.video_info.fps} FPS → {self.target_fps} FPS "
            f"(1 frame / {self.frame_interval})"
        )
    
    async def extract_frames(self) -> AsyncGenerator[Tuple[int, np.ndarray, float], None]:
        """🎞️ Générateur asynchrone de frames"""
        
        cap = cv2.VideoCapture(str(self.video_info.path))
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Ne traiter qu'un frame sur N selon l'intervalle
                if frame_number % self.frame_interval == 0:
                    # Timestamp de la frame
                    timestamp = frame_number / self.video_info.fps
                    
                    # Conversion BGR → RGB pour PIL/PyTorch
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    yield frame_number, frame_rgb, timestamp
                    
                    # Yield control pour permettre à d'autres coroutines de s'exécuter
                    await asyncio.sleep(0)
                
                frame_number += 1
                
        finally:
            cap.release()
    
    async def extract_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """⏰ Extrait une frame à un timestamp spécifique"""
        
        cap = cv2.VideoCapture(str(self.video_info.path))
        
        try:
            # Aller au timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            
            ret, frame = cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return None
            
        finally:
            cap.release()

class VideoProcessor:
    """🎬 Processeur principal de vidéo"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        self.max_video_size = 500 * 1024 * 1024  # 500MB
        self.max_duration = 1800  # 30 minutes
    
    async def validate_video(self, video_path: str) -> VideoInfo:
        """✅ Valide un fichier vidéo"""
        
        video_path = Path(video_path)
        
        # Vérification existence
        if not video_path.exists():
            raise VideoValidationError(f"Fichier vidéo non trouvé: {video_path}")
        
        # Vérification extension
        if video_path.suffix.lower() not in self.supported_formats:
            raise VideoValidationError(
                f"Format non supporté: {video_path.suffix}. "
                f"Formats supportés: {self.supported_formats}"
            )
        
        # Vérification taille
        file_size = video_path.stat().st_size
        if file_size > self.max_video_size:
            raise VideoValidationError(
                f"Fichier trop volumineux: {file_size / 1024 / 1024:.1f}MB "
                f"(max: {self.max_video_size / 1024 / 1024:.1f}MB)"
            )
        
        # Chargement et validation des propriétés
        try:
            video_info = VideoInfo(str(video_path))
            
            # Vérification durée
            if video_info.duration > self.max_duration:
                raise VideoValidationError(
                    f"Vidéo trop longue: {video_info.duration:.1f}s "
                    f"(max: {self.max_duration}s)"
                )
            
            # Vérification résolution
            if video_info.width * video_info.height > 1920 * 1080:
                logger.warning(
                    f"⚠️ Haute résolution détectée: {video_info.width}x{video_info.height}"
                )
            
            return video_info
            
        except Exception as e:
            raise VideoValidationError(f"Erreur lecture vidéo: {e}")

class VideoTimeline:
    """📊 Timeline des événements vidéo"""
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.lost_objects_timeline: Dict[str, List[Dict[str, Any]]] = {}
        self.detection_summary: Dict[str, Any] = {}
    
    def add_frame_detections(
        self, 
        frame_number: int, 
        timestamp: float,
        detections: List[DetectionResult],
        lost_objects: List[LostObjectState]
    ):
        """➕ Ajoute les détections d'une frame"""
        
        # Événement de frame
        frame_event = {
            "type": "frame_processed",
            "frame_number": frame_number,
            "timestamp": timestamp,
            "detections_count": len(detections),
            "lost_objects_count": len(lost_objects),
            "objects_by_class": self._count_by_class(detections)
        }
        self.events.append(frame_event)
        
        # Événements objets perdus
        for lost_object in lost_objects:
            object_id = lost_object.object_id
            
            if object_id not in self.lost_objects_timeline:
                self.lost_objects_timeline[object_id] = []
            
            # Événement selon le statut
            if lost_object.status != ObjectStatus.NORMAL:
                event = {
                    "timestamp": timestamp,
                    "frame_number": frame_number,
                    "status": lost_object.status.value,
                    "duration": lost_object.stationary_duration,
                    "class_name": lost_object.detection_result.class_name,
                    "confidence": lost_object.detection_result.confidence,
                    "position": {
                        "x": lost_object.detection_result.bbox.center[0],
                        "y": lost_object.detection_result.bbox.center[1]
                    }
                }
                self.lost_objects_timeline[object_id].append(event)
                
                # Événement global si nouveau statut
                if (not self.events or 
                    self.events[-1].get("object_id") != object_id or
                    self.events[-1].get("status") != lost_object.status.value):
                    
                    global_event = {
                        "type": "lost_object_status_change",
                        "object_id": object_id,
                        "frame_number": frame_number,
                        "timestamp": timestamp,
                        "status": lost_object.status.value,
                        "class_name": lost_object.detection_result.class_name
                    }
                    self.events.append(global_event)
    
    def _count_by_class(self, detections: List[DetectionResult]) -> Dict[str, int]:
        """📊 Compte les détections par classe"""
        counts = {}
        for detection in detections:
            counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
        return counts
    
    def generate_summary(self) -> Dict[str, Any]:
        """📋 Génère un résumé de la timeline"""
        
        total_frames = len([e for e in self.events if e["type"] == "frame_processed"])
        total_detections = sum(e.get("detections_count", 0) for e in self.events)
        
        # Objets perdus par statut
        lost_objects_by_status = {}
        for timeline in self.lost_objects_timeline.values():
            if timeline:
                final_status = timeline[-1]["status"]
                lost_objects_by_status[final_status] = lost_objects_by_status.get(final_status, 0) + 1
        
        # Classes les plus détectées
        all_classes = {}
        for event in self.events:
            if event["type"] == "frame_processed":
                for class_name, count in event.get("objects_by_class", {}).items():
                    all_classes[class_name] = all_classes.get(class_name, 0) + count
        
        most_detected_classes = sorted(all_classes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_frames_processed": total_frames,
            "total_detections": total_detections,
            "unique_lost_objects": len(self.lost_objects_timeline),
            "lost_objects_by_status": lost_objects_by_status,
            "most_detected_classes": most_detected_classes,
            "timeline_events": len(self.events)
        }
    
    def export_to_json(self, output_path: str):
        """💾 Exporte la timeline en JSON"""
        
        timeline_data = {
            "summary": self.generate_summary(),
            "events": self.events,
            "lost_objects_timeline": self.lost_objects_timeline,
            "generated_at": datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(timeline_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Timeline exportée: {output_path}")

class VideoAnnotator:
    """🎨 Annotateur de vidéo avec détections"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.output_dir = settings.TEMP_DIR / "video_results"
        self.output_dir.mkdir(exist_ok=True)
    
    async def create_annotated_video(
        self,
        original_video_path: str,
        detections_by_frame: Dict[int, Tuple[List[DetectionResult], List[LostObjectState]]],
        output_filename: Optional[str] = None
    ) -> str:
        """🎬 Crée une vidéo annotée avec les détections"""
        
        video_info = VideoInfo(original_video_path)
        
        # Nom de fichier de sortie
        if not output_filename:
            output_filename = f"annotated_{int(time.time())}.mp4"
        
        output_path = self.output_dir / output_filename
        
        # Configuration de l'encodeur
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            video_info.fps,
            (video_info.width, video_info.height)
        )
        
        try:
            cap = cv2.VideoCapture(original_video_path)
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Annotation si des détections existent pour cette frame
                if frame_number in detections_by_frame:
                    detections, lost_objects = detections_by_frame[frame_number]
                    frame = self._annotate_frame(frame, detections, lost_objects)
                
                writer.write(frame)
                frame_number += 1
                
                # Yield pour éviter le blocage
                if frame_number % 30 == 0:
                    await asyncio.sleep(0)
            
            cap.release()
            writer.release()
            
            logger.info(f"🎬 Vidéo annotée créée: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Erreur création vidéo annotée: {e}")
            if writer:
                writer.release()
            raise
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        lost_objects: List[LostObjectState]
    ) -> np.ndarray:
        """🖊️ Annote une frame avec les détections"""
        
        # Copie pour annotation
        annotated_frame = frame.copy()
        
        # Couleurs
        colors = {
            'normal': (0, 255, 0),      # Vert
            'suspect': (0, 165, 255),   # Orange
            'lost': (0, 0, 255),        # Rouge
            'critical': (0, 0, 139)     # Rouge foncé
        }
        
        # Dessiner les détections normales
        for detection in detections:
            bbox = detection.bbox
            color = colors['normal']
            
            # Rectangle
            cv2.rectangle(
                annotated_frame,
                (bbox.x1, bbox.y1),
                (bbox.x2, bbox.y2),
                color,
                2
            )
            
            # Label
            label = f"{detection.class_name} {detection.confidence:.2f}"
            cv2.putText(
                annotated_frame,
                label,
                (bbox.x1, bbox.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
        
        # Dessiner les objets perdus (par-dessus)
        for lost_object in lost_objects:
            if lost_object.status == ObjectStatus.NORMAL:
                continue
            
            bbox = lost_object.detection_result.bbox
            color = colors.get(lost_object.status.value, colors['lost'])
            
            # Rectangle épais
            cv2.rectangle(
                annotated_frame,
                (bbox.x1 - 2, bbox.y1 - 2),
                (bbox.x2 + 2, bbox.y2 + 2),
                color,
                4
            )
            
            # Label spécial
            status_text = {
                'suspect': 'SUSPECT',
                'lost': 'PERDU',
                'critical': 'CRITIQUE'
            }.get(lost_object.status.value, 'INCONNU')
            
            label = f"{status_text} ({lost_object.stationary_duration}s)"
            
            # Fond pour le texte
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            
            cv2.rectangle(
                annotated_frame,
                (bbox.x1, bbox.y2 + 5),
                (bbox.x1 + text_width, bbox.y2 + 5 + text_height + 10),
                color,
                -1
            )
            
            cv2.putText(
                annotated_frame,
                label,
                (bbox.x1, bbox.y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
        
        return annotated_frame

class VideoService:
    """🎬 Service principal de traitement vidéo"""
    
    def __init__(self, model_manager: ModelManager, settings: Settings):
        self.model_manager = model_manager
        self.settings = settings
        
        # Composants
        self.video_processor = VideoProcessor(settings)
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = ResultPostprocessor(settings)
        self.annotator = VideoAnnotator(settings)
        
        # Tracking pour vidéos
        self._video_trackers: Dict[str, LostObjectDetector] = {}
        
        # Statistiques
        self.videos_processed = 0
        self.total_frames_processed = 0
        self.processing_times: List[float] = []
        
        logger.info("🎬 VideoService initialisé")
    
    async def process_video(
        self,
        video_path: str,
        request: DetectionRequest,
        progress_callback: Optional[callable] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        🎯 Traite une vidéo complète
        
        Args:
            video_path: Chemin vers le fichier vidéo
            request: Paramètres de détection
            progress_callback: Callback pour progression (optionnel)
            request_id: ID unique de la requête
            
        Returns:
            Résultats complets avec timeline et fichiers générés
        """
        
        start_time = time.time()
        request_id = request_id or str(uuid.uuid4())
        
        try:
            # 1. Validation de la vidéo
            video_info = await self.video_processor.validate_video(video_path)
            logger.info(f"🎬 Traitement vidéo {request_id}: {video_info.get_info_dict()}")
            
            # 2. Configuration du détecteur
            detector_config = DetectionConfig(
                confidence_threshold=request.confidence_threshold,
                nms_threshold=request.nms_threshold,
                max_detections=request.max_detections,
                detection_mode=request.detection_mode,
                model_type=request.model_name
            )
            
            detector = ObjectDetector(self.model_manager, detector_config)
            
            # 3. Tracker spécialisé pour cette vidéo
            if request.enable_lost_detection:
                video_tracker = LostObjectDetector(self.settings)
                self._video_trackers[request_id] = video_tracker
            
            # 4. Traitement frame par frame
            frame_extractor = VideoFrameExtractor(
                video_path, 
                target_fps=5.0  # 5 FPS pour l'analyse
            )
            
            timeline = VideoTimeline()
            detections_by_frame = {}
            frames_processed = 0
            
            async for frame_number, frame_array, timestamp in frame_extractor.extract_frames():
                # Conversion en PIL Image
                pil_frame = Image.fromarray(frame_array)
                
                # Détection sur cette frame
                detections = await detector.detect_objects(
                    image=pil_frame,
                    model_name=request.model_name.value,
                    confidence_threshold=request.confidence_threshold,
                    detection_id=f"{request_id}_frame_{frame_number}"
                )
                
                # Analyse objets perdus
                lost_objects = []
                if request.enable_lost_detection and request_id in self._video_trackers:
                    lost_objects = self._video_trackers[request_id].analyze_detections(
                        detections,
                        datetime.fromtimestamp(start_time + timestamp)
                    )
                
                # Mise à jour timeline
                timeline.add_frame_detections(frame_number, timestamp, detections, lost_objects)
                detections_by_frame[frame_number] = (detections, lost_objects)
                
                frames_processed += 1
                self.total_frames_processed += 1
                
                # Callback de progression
                if progress_callback:
                    progress = frames_processed / (video_info.frame_count / frame_extractor.frame_interval)
                    await progress_callback(min(progress, 1.0), frames_processed, timestamp)
                
                # Yield périodique
                if frames_processed % 10 == 0:
                    await asyncio.sleep(0.01)
            
            # 5. Génération des fichiers de sortie
            output_files = {}
            
            # Timeline JSON
            timeline_path = self.settings.TEMP_DIR / f"timeline_{request_id}.json"
            timeline.export_to_json(str(timeline_path))
            output_files['timeline'] = str(timeline_path)
            
            # Vidéo annotée (si demandée)
            if request.return_cropped_objects:
                annotated_video_path = await self.annotator.create_annotated_video(
                    video_path,
                    detections_by_frame,
                    f"annotated_{request_id}.mp4"
                )
                output_files['annotated_video'] = annotated_video_path
            
            # 6. Résumé final
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.videos_processed += 1
            
            summary = timeline.generate_summary()
            
            result = {
                "success": True,
                "request_id": request_id,
                "video_info": video_info.get_info_dict(),
                "processing_summary": {
                    "frames_processed": frames_processed,
                    "processing_time_seconds": processing_time,
                    "avg_time_per_frame_ms": (processing_time / frames_processed) * 1000 if frames_processed > 0 else 0,
                    **summary
                },
                "output_files": output_files,
                "model_used": request.model_name.value
            }
            
            logger.info(
                f"🎬 Vidéo traitée {request_id}: {frames_processed} frames "
                f"en {processing_time:.1f}s ({summary['unique_lost_objects']} objets perdus)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement vidéo {request_id}: {e}")
            return {
                "success": False,
                "request_id": request_id,
                "error": str(e),
                "processing_time_seconds": time.time() - start_time
            }
        
        finally:
            # Nettoyage du tracker
            if request_id in self._video_trackers:
                del self._video_trackers[request_id]
    
    async def get_video_preview(
        self,
        video_path: str,
        timestamps: List[float]
    ) -> List[str]:
        """👁️ Génère des images de prévisualisation"""
        
        frame_extractor = VideoFrameExtractor(video_path)
        preview_paths = []
        
        for i, timestamp in enumerate(timestamps):
            frame = await frame_extractor.extract_frame_at_time(timestamp)
            
            if frame is not None:
                # Sauvegarde de l'image
                preview_filename = f"preview_{int(timestamp)}s_{i}.jpg"
                preview_path = self.settings.TEMP_DIR / preview_filename
                
                pil_image = Image.fromarray(frame)
                pil_image.save(preview_path, "JPEG", quality=85)
                
                preview_paths.append(str(preview_path))
        
        return preview_paths
    
    def get_supported_formats(self) -> List[str]:
        """📋 Formats vidéo supportés"""
        return list(self.video_processor.supported_formats)
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """📊 Statistiques du service"""
        
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            "videos_processed": self.videos_processed,
            "total_frames_processed": self.total_frames_processed,
            "average_processing_time_seconds": avg_processing_time,
            "active_video_trackers": len(self._video_trackers),
            "supported_formats": self.get_supported_formats()
        }
    
    async def cleanup(self):
        """🧹 Nettoyage des ressources"""
        self._video_trackers.clear()
        logger.info("🧹 VideoService nettoyé")

# === EXPORTS ===
__all__ = [
    "VideoService",
    "VideoProcessor", 
    "VideoTimeline",
    "VideoAnnotator",
    "VideoInfo",
    "VideoValidationError"
]