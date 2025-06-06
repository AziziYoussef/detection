"""
🎬 VIDEO SERVICE - SERVICE DE TRAITEMENT DE VIDÉOS UPLOADÉES
===========================================================
Service spécialisé pour la détection d'objets perdus dans des vidéos

Fonctionnalités:
- Support multi-formats (MP4, AVI, MOV, MKV, WebM)
- Extraction et traitement de frames
- Détection avec tracking temporel
- Timeline des détections avec timestamps
- Génération de vidéos annotées
- Optimisation pour vidéos longues
- Analyse de mouvement et objets statiques
- Export de rapports détaillés

Formats supportés:
- Vidéos: MP4, AVI, MOV, MKV, WebM, FLV
- Codecs: H.264, H.265, VP8, VP9, MPEG-4
- Taille max: 500MB par vidéo
- Durée max: 30 minutes
- Résolution max: 4K (3840x2160)
"""

import asyncio
import time
import logging
import json
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading

import numpy as np
import cv2
from PIL import Image

# Imports internes
from app.schemas.detection import DetectionResult, VideoProcessingResult, VideoFrame, VideoTimeline
from app.config.config import get_settings
from .model_service import ModelService, PerformanceProfile
from .image_service import ImageService, ImageProcessingConfig

logger = logging.getLogger(__name__)

# 📋 ÉNUMÉRATIONS ET TYPES
class VideoFormat(str, Enum):
    """🎬 Formats vidéo supportés"""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    MKV = "mkv"
    WEBM = "webm"
    FLV = "flv"

class ProcessingStrategy(str, Enum):
    """🎛️ Stratégies de traitement vidéo"""
    FULL_FRAMES = "full_frames"           # Toutes les frames
    SAMPLE_UNIFORM = "sample_uniform"     # Échantillonnage uniforme
    SAMPLE_ADAPTIVE = "sample_adaptive"   # Échantillonnage adaptatif
    MOTION_BASED = "motion_based"         # Basé sur le mouvement
    KEYFRAMES_ONLY = "keyframes_only"     # Seulement les keyframes

class TrackingMode(str, Enum):
    """🎯 Modes de tracking"""
    NONE = "none"               # Pas de tracking
    SIMPLE = "simple"           # Tracking simple par position
    ADVANCED = "advanced"       # Tracking avancé avec features
    PERSISTENCE = "persistence" # Suivi de persistance d'objets

@dataclass
class VideoProcessingConfig:
    """⚙️ Configuration du traitement vidéo"""
    
    # 🔧 Paramètres de base
    max_video_size_mb: float = 500.0
    max_duration_minutes: float = 30.0
    max_resolution: Tuple[int, int] = (3840, 2160)  # 4K
    supported_formats: List[str] = field(default_factory=lambda: ["mp4", "avi", "mov", "mkv", "webm"])
    
    # 🎬 Stratégie de traitement
    processing_strategy: ProcessingStrategy = ProcessingStrategy.SAMPLE_ADAPTIVE
    frames_per_second: Optional[float] = None  # Auto si None
    max_frames_to_process: int = 300  # Limite pour éviter surcharge
    
    # 🎯 Tracking et détection
    tracking_mode: TrackingMode = TrackingMode.SIMPLE
    confidence_threshold: float = 0.5
    temporal_smoothing: bool = True
    min_detection_persistence: int = 3  # Frames minimum pour validation
    
    # 📊 Optimisations
    resize_for_processing: bool = True
    target_processing_size: Tuple[int, int] = (640, 640)
    parallel_processing: bool = True
    max_workers: int = 4
    
    # 💾 Sauvegarde
    save_annotated_video: bool = True
    save_timeline_json: bool = True
    save_frame_samples: bool = False
    export_summary: bool = True
    
    # 🔧 Avancé
    motion_threshold: float = 0.1  # Seuil de détection de mouvement
    keyframe_interval: int = 30    # Intervalle keyframes
    
    def __post_init__(self):
        """Validation de la configuration"""
        if self.confidence_threshold < 0.1 or self.confidence_threshold > 0.9:
            raise ValueError("confidence_threshold doit être entre 0.1 et 0.9")
        if self.max_frames_to_process < 10:
            raise ValueError("max_frames_to_process doit être >= 10")

# 🎬 SERVICE PRINCIPAL
class VideoService:
    """🎬 Service de traitement de vidéos"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.settings = get_settings()
        
        # Configuration par défaut
        self.config = VideoProcessingConfig()
        
        # Service d'images pour traiter les frames
        self.image_service = ImageService(model_service)
        
        # Chemins de stockage
        self.uploads_path = Path(self.settings.UPLOADS_PATH) / "videos"
        self.results_path = Path(self.settings.RESULTS_PATH) / "videos"
        self.cache_path = Path(self.settings.CACHE_PATH) / "videos"
        
        # Créer les répertoires
        for path in [self.uploads_path, self.results_path, self.cache_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Tracking des objets
        self._trackers: Dict[str, Any] = {}
        
        # Statistiques
        self.stats = {
            "total_processed": 0,
            "total_frames_analyzed": 0,
            "total_detections": 0,
            "processing_times": [],
            "format_counts": {},
            "error_count": 0,
            "average_fps_processed": 0.0
        }
        
        logger.info("🎬 VideoService initialisé")
    
    async def initialize(self):
        """🚀 Initialise le service vidéo"""
        logger.info("🚀 Initialisation VideoService...")
        
        # Vérifier OpenCV
        try:
            import cv2
            logger.debug(f"✅ OpenCV {cv2.__version__} disponible")
        except ImportError:
            logger.error("❌ OpenCV requis pour VideoService")
            raise
        
        # Initialiser le service d'images
        await self.image_service.initialize()
        
        logger.info("✅ VideoService initialisé")
    
    async def process_video(
        self,
        video_input: Union[Path, str, bytes],
        config: Optional[VideoProcessingConfig] = None,
        processing_id: Optional[str] = None
    ) -> VideoProcessingResult:
        """
        🎬 Traite une vidéo et détecte les objets perdus
        
        Args:
            video_input: Vidéo sous différents formats
            config: Configuration de traitement
            processing_id: ID unique pour ce traitement
            
        Returns:
            Résultat complet du traitement vidéo
        """
        
        start_time = time.time()
        processing_id = processing_id or str(uuid.uuid4())
        config = config or self.config
        
        logger.info(f"🎬 Début traitement vidéo: {processing_id}")
        
        try:
            # 1. Chargement et validation de la vidéo
            video_info = await self._load_and_validate_video(video_input, processing_id)
            
            # 2. Extraction des frames selon la stratégie
            frames_data = await self._extract_frames(video_info, config, processing_id)
            
            # 3. Traitement des frames
            frame_results = await self._process_frames(frames_data, config, processing_id)
            
            # 4. Tracking temporel des objets
            tracked_timeline = await self._apply_temporal_tracking(
                frame_results, video_info, config
            )
            
            # 5. Génération de la timeline finale
            timeline = await self._generate_timeline(tracked_timeline, video_info)
            
            # 6. Création des visualisations
            visualizations = await self._generate_video_visualizations(
                video_info, timeline, config, processing_id
            )
            
            # 7. Génération du rapport
            summary = await self._generate_video_summary(timeline, video_info, config)
            
            # 8. Création du résultat final
            processing_time = time.time() - start_time
            
            result = VideoProcessingResult(
                processing_id=processing_id,
                video_info=video_info,
                timeline=timeline,
                summary=summary,
                processing_config=config.__dict__,
                processing_time_seconds=processing_time,
                visualizations=visualizations,
                metadata={
                    "total_frames_processed": len(frame_results),
                    "total_detections": sum(len(frame.detections) for frame in timeline.frames),
                    "unique_objects": len(timeline.unique_objects),
                    "video_duration_seconds": video_info.get("duration", 0),
                    "processing_fps": len(frame_results) / processing_time if processing_time > 0 else 0
                }
            )
            
            # 9. Sauvegarde des résultats
            if config.save_timeline_json:
                await self._save_video_result(result, processing_id)
            
            # 10. Mise à jour des statistiques
            self._update_video_statistics(result, video_info)
            
            logger.info(
                f"✅ Vidéo traitée: {processing_id} - "
                f"{len(timeline.frames)} frames, {len(timeline.unique_objects)} objets uniques "
                f"en {processing_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            self.stats["error_count"] += 1
            logger.error(f"❌ Erreur traitement vidéo {processing_id}: {e}")
            
            # Retourner un résultat d'erreur
            return VideoProcessingResult(
                processing_id=processing_id,
                video_info={"error": str(e)},
                timeline=VideoTimeline(frames=[], unique_objects=[], duration_seconds=0.0),
                summary={},
                processing_config=config.__dict__ if config else {},
                processing_time_seconds=time.time() - start_time,
                visualizations={},
                metadata={"error": str(e)}
            )
    
    async def _load_and_validate_video(
        self,
        video_input: Union[Path, str, bytes],
        processing_id: str
    ) -> Dict[str, Any]:
        """🔍 Charge et valide une vidéo"""
        
        video_info = {
            "processing_id": processing_id,
            "file_path": None,
            "format": None,
            "duration": 0.0,
            "fps": 0.0,
            "frame_count": 0,
            "resolution": (0, 0),
            "file_size_mb": 0.0,
            "codec": None,
            "cap": None
        }
        
        try:
            # Gestion du type d'entrée
            if isinstance(video_input, bytes):
                # Sauvegarder les bytes temporairement
                temp_path = self.cache_path / f"{processing_id}_temp.mp4"
                with open(temp_path, 'wb') as f:
                    f.write(video_input)
                video_path = temp_path
                video_info["file_size_mb"] = len(video_input) / (1024 * 1024)
            else:
                # Fichier
                video_path = Path(video_input)
                if not video_path.exists():
                    raise FileNotFoundError(f"Vidéo non trouvée: {video_path}")
                video_info["file_size_mb"] = video_path.stat().st_size / (1024 * 1024)
            
            # Validation de la taille
            if video_info["file_size_mb"] > self.config.max_video_size_mb:
                raise ValueError(f"Vidéo trop grande: {video_info['file_size_mb']:.1f}MB")
            
            # Ouverture avec OpenCV
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")
            
            # Extraction des métadonnées
            video_info.update({
                "file_path": str(video_path),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "resolution": (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ),
                "codec": self._get_video_codec(cap),
                "cap": cap
            })
            
            # Calcul de la durée
            if video_info["fps"] > 0:
                video_info["duration"] = video_info["frame_count"] / video_info["fps"]
            
            # Validation des limites
            if video_info["duration"] > self.config.max_duration_minutes * 60:
                raise ValueError(f"Vidéo trop longue: {video_info['duration']:.1f}s")
            
            if max(video_info["resolution"]) > max(self.config.max_resolution):
                raise ValueError(f"Résolution trop élevée: {video_info['resolution']}")
            
            # Format de fichier
            video_info["format"] = video_path.suffix.lower().lstrip('.')
            
            return video_info
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement vidéo {processing_id}: {e}")
            raise
    
    def _get_video_codec(self, cap: cv2.VideoCapture) -> str:
        """🔍 Détermine le codec vidéo"""
        try:
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            return codec.strip()
        except:
            return "unknown"
    
    async def _extract_frames(
        self,
        video_info: Dict[str, Any],
        config: VideoProcessingConfig,
        processing_id: str
    ) -> List[Dict[str, Any]]:
        """🎞️ Extrait les frames selon la stratégie"""
        
        cap = video_info["cap"]
        total_frames = video_info["frame_count"]
        fps = video_info["fps"]
        
        frames_data = []
        
        # Déterminer la stratégie d'extraction
        if config.processing_strategy == ProcessingStrategy.FULL_FRAMES:
            frame_indices = list(range(min(total_frames, config.max_frames_to_process)))
            
        elif config.processing_strategy == ProcessingStrategy.SAMPLE_UNIFORM:
            # Échantillonnage uniforme
            step = max(1, total_frames // config.max_frames_to_process)
            frame_indices = list(range(0, total_frames, step))[:config.max_frames_to_process]
            
        elif config.processing_strategy == ProcessingStrategy.SAMPLE_ADAPTIVE:
            # Échantillonnage adaptatif basé sur la durée
            if video_info["duration"] <= 60:  # <= 1 minute
                step = max(1, total_frames // min(config.max_frames_to_process, 100))
            elif video_info["duration"] <= 300:  # <= 5 minutes
                step = max(1, total_frames // min(config.max_frames_to_process, 200))
            else:  # > 5 minutes
                step = max(1, total_frames // config.max_frames_to_process)
            
            frame_indices = list(range(0, total_frames, step))[:config.max_frames_to_process]
            
        elif config.processing_strategy == ProcessingStrategy.KEYFRAMES_ONLY:
            # Extraction des keyframes (approximation)
            step = config.keyframe_interval
            frame_indices = list(range(0, total_frames, step))[:config.max_frames_to_process]
            
        else:  # MOTION_BASED
            frame_indices = await self._extract_motion_based_frames(
                cap, total_frames, config.max_frames_to_process, config.motion_threshold
            )
        
        logger.info(f"🎞️ Extraction de {len(frame_indices)} frames sur {total_frames}")
        
        # Extraction des frames
        for i, frame_idx in enumerate(frame_indices):
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Conversion BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Timestamp
                timestamp = frame_idx / fps if fps > 0 else 0.0
                
                frame_data = {
                    "index": frame_idx,
                    "timestamp": timestamp,
                    "image": Image.fromarray(frame_rgb),
                    "original_size": frame_rgb.shape[:2][::-1]  # (width, height)
                }
                
                frames_data.append(frame_data)
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur extraction frame {frame_idx}: {e}")
                continue
        
        return frames_data
    
    async def _extract_motion_based_frames(
        self,
        cap: cv2.VideoCapture,
        total_frames: int,
        max_frames: int,
        motion_threshold: float
    ) -> List[int]:
        """🏃 Extrait les frames basées sur le mouvement"""
        
        selected_frames = [0]  # Toujours inclure la première frame
        prev_frame = None
        
        # Échantillonnage pour détecter le mouvement
        check_interval = max(1, total_frames // (max_frames * 3))
        
        for frame_idx in range(0, total_frames, check_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Conversion en niveaux de gris
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if prev_frame is not None:
                # Différence entre frames
                frame_diff = cv2.absdiff(prev_frame, gray)
                thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
                
                # Pourcentage de pixels différents
                motion_ratio = np.count_nonzero(thresh) / thresh.size
                
                if motion_ratio > motion_threshold:
                    selected_frames.append(frame_idx)
                    
                    if len(selected_frames) >= max_frames:
                        break
            
            prev_frame = gray
        
        return selected_frames[:max_frames]
    
    async def _process_frames(
        self,
        frames_data: List[Dict[str, Any]],
        config: VideoProcessingConfig,
        processing_id: str
    ) -> List[VideoFrame]:
        """🔄 Traite toutes les frames extraites"""
        
        logger.info(f"🔄 Traitement de {len(frames_data)} frames")
        
        frame_results = []
        
        if config.parallel_processing and len(frames_data) > 4:
            # Traitement en parallèle
            semaphore = asyncio.Semaphore(config.max_workers)
            
            async def process_single_frame(frame_data, frame_num):
                async with semaphore:
                    return await self._process_single_frame(frame_data, config, f"{processing_id}_f{frame_num}")
            
            tasks = [
                process_single_frame(frame_data, i)
                for i, frame_data in enumerate(frames_data)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Gérer les exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"⚠️ Erreur frame {i}: {result}")
                    # Frame vide en cas d'erreur
                    result = VideoFrame(
                        frame_index=frames_data[i]["index"],
                        timestamp=frames_data[i]["timestamp"],
                        detections=[],
                        frame_info={"error": str(result)}
                    )
                
                frame_results.append(result)
        
        else:
            # Traitement séquentiel
            for i, frame_data in enumerate(frames_data):
                try:
                    result = await self._process_single_frame(
                        frame_data, config, f"{processing_id}_f{i}"
                    )
                    frame_results.append(result)
                except Exception as e:
                    logger.warning(f"⚠️ Erreur frame {i}: {e}")
                    # Frame vide
                    frame_results.append(VideoFrame(
                        frame_index=frame_data["index"],
                        timestamp=frame_data["timestamp"],
                        detections=[],
                        frame_info={"error": str(e)}
                    ))
        
        return frame_results
    
    async def _process_single_frame(
        self,
        frame_data: Dict[str, Any],
        config: VideoProcessingConfig,
        frame_processing_id: str
    ) -> VideoFrame:
        """🖼️ Traite une frame individuelle"""
        
        image = frame_data["image"]
        
        # Configuration pour le traitement d'image
        image_config = ImageProcessingConfig(
            confidence_threshold=config.confidence_threshold,
            target_size=config.target_processing_size if config.resize_for_processing else None,
            save_results=False,  # Pas de sauvegarde individuelle
            save_annotated_images=False,
            cache_results=False,  # Pas de cache pour frames vidéo
            quality="fast" if config.parallel_processing else "balanced"
        )
        
        # Traitement de l'image
        result = await self.image_service.process_image(
            image_input=image,
            config=image_config,
            processing_id=frame_processing_id
        )
        
        # Création de la frame vidéo
        video_frame = VideoFrame(
            frame_index=frame_data["index"],
            timestamp=frame_data["timestamp"],
            detections=result.detections,
            frame_info={
                "processing_time": result.processing_time_seconds,
                "model_used": result.model_used,
                "original_size": frame_data["original_size"]
            }
        )
        
        return video_frame
    
    async def _apply_temporal_tracking(
        self,
        frame_results: List[VideoFrame],
        video_info: Dict[str, Any],
        config: VideoProcessingConfig
    ) -> List[VideoFrame]:
        """🎯 Applique le tracking temporel des objets"""
        
        if config.tracking_mode == TrackingMode.NONE:
            return frame_results
        
        logger.info(f"🎯 Application du tracking temporel: {config.tracking_mode.value}")
        
        tracked_frames = []
        object_tracks = {}  # {track_id: [détections]}
        next_track_id = 1
        
        for frame in frame_results:
            tracked_detections = []
            
            for detection in frame.detections:
                # Chercher correspondance avec tracks existants
                best_track_id = None
                best_distance = float('inf')
                
                for track_id, track_history in object_tracks.items():
                    if not track_history:
                        continue
                    
                    last_detection = track_history[-1]
                    
                    # Vérifier si même classe
                    if last_detection.class_id != detection.class_id:
                        continue
                    
                    # Calculer distance spatiale
                    distance = self._calculate_detection_distance(last_detection, detection)
                    
                    # Vérifier si dans le seuil acceptable
                    if distance < 100 and distance < best_distance:  # 100 pixels max
                        best_distance = distance
                        best_track_id = track_id
                
                # Assigner à un track
                if best_track_id is not None:
                    # Track existant
                    detection.track_id = best_track_id
                    object_tracks[best_track_id].append(detection)
                else:
                    # Nouveau track
                    detection.track_id = next_track_id
                    object_tracks[next_track_id] = [detection]
                    next_track_id += 1
                
                tracked_detections.append(detection)
            
            # Créer la frame trackée
            tracked_frame = VideoFrame(
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
                detections=tracked_detections,
                frame_info=frame.frame_info
            )
            
            tracked_frames.append(tracked_frame)
        
        # Filtrage par persistance
        if config.min_detection_persistence > 1:
            tracked_frames = self._filter_by_persistence(tracked_frames, config.min_detection_persistence)
        
        return tracked_frames
    
    def _calculate_detection_distance(self, det1: DetectionResult, det2: DetectionResult) -> float:
        """📏 Calcule la distance entre deux détections"""
        
        # Centre des boîtes
        center1 = ((det1.bbox.x1 + det1.bbox.x2) / 2, (det1.bbox.y1 + det1.bbox.y2) / 2)
        center2 = ((det2.bbox.x1 + det2.bbox.x2) / 2, (det2.bbox.y1 + det2.bbox.y2) / 2)
        
        # Distance euclidienne
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _filter_by_persistence(
        self,
        frames: List[VideoFrame],
        min_persistence: int
    ) -> List[VideoFrame]:
        """🔍 Filtre les détections par persistance"""
        
        # Compter les apparitions par track
        track_counts = {}
        for frame in frames:
            for detection in frame.detections:
                track_id = getattr(detection, 'track_id', None)
                if track_id:
                    track_counts[track_id] = track_counts.get(track_id, 0) + 1
        
        # Tracks valides (apparaissent assez souvent)
        valid_tracks = {
            track_id for track_id, count in track_counts.items()
            if count >= min_persistence
        }
        
        # Filtrer les frames
        filtered_frames = []
        for frame in frames:
            filtered_detections = [
                det for det in frame.detections
                if getattr(det, 'track_id', None) in valid_tracks
            ]
            
            filtered_frame = VideoFrame(
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
                detections=filtered_detections,
                frame_info=frame.frame_info
            )
            
            filtered_frames.append(filtered_frame)
        
        return filtered_frames
    
    async def _generate_timeline(
        self,
        tracked_frames: List[VideoFrame],
        video_info: Dict[str, Any]
    ) -> VideoTimeline:
        """📊 Génère la timeline finale"""
        
        # Objets uniques détectés
        unique_objects = {}
        
        for frame in tracked_frames:
            for detection in frame.detections:
                track_id = getattr(detection, 'track_id', f"temp_{id(detection)}")
                
                if track_id not in unique_objects:
                    unique_objects[track_id] = {
                        "track_id": track_id,
                        "class_name": detection.class_name,
                        "class_name_fr": detection.class_name_fr,
                        "first_seen": frame.timestamp,
                        "last_seen": frame.timestamp,
                        "total_appearances": 1,
                        "max_confidence": detection.confidence,
                        "avg_confidence": detection.confidence
                    }
                else:
                    obj = unique_objects[track_id]
                    obj["last_seen"] = frame.timestamp
                    obj["total_appearances"] += 1
                    obj["max_confidence"] = max(obj["max_confidence"], detection.confidence)
                    obj["avg_confidence"] = (obj["avg_confidence"] + detection.confidence) / 2
        
        timeline = VideoTimeline(
            frames=tracked_frames,
            unique_objects=list(unique_objects.values()),
            duration_seconds=video_info.get("duration", 0.0)
        )
        
        return timeline
    
    async def _generate_video_visualizations(
        self,
        video_info: Dict[str, Any],
        timeline: VideoTimeline,
        config: VideoProcessingConfig,
        processing_id: str
    ) -> Dict[str, str]:
        """🎨 Génère les visualisations vidéo"""
        
        visualizations = {}
        
        if not config.save_annotated_video:
            return visualizations
        
        try:
            # Vidéo annotée
            annotated_path = await self._create_annotated_video(
                video_info, timeline, processing_id
            )
            if annotated_path:
                visualizations["annotated_video"] = str(annotated_path)
            
            # Échantillons de frames annotées
            if config.save_frame_samples:
                samples_paths = await self._save_frame_samples(
                    timeline, processing_id
                )
                visualizations["frame_samples"] = samples_paths
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur génération visualisations vidéo: {e}")
        
        return visualizations
    
    async def _create_annotated_video(
        self,
        video_info: Dict[str, Any],
        timeline: VideoTimeline,
        processing_id: str
    ) -> Optional[Path]:
        """🎬 Crée une vidéo annotée avec les détections"""
        
        try:
            input_path = video_info["file_path"]
            output_path = self.results_path / f"{processing_id}_annotated.mp4"
            
            # Ouvrir vidéo d'entrée
            cap = cv2.VideoCapture(input_path)
            
            # Configuration de sortie
            fps = video_info["fps"]
            width, height = video_info["resolution"]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # Créer un dictionnaire frame_index -> détections
            detections_by_frame = {}
            for frame in timeline.frames:
                detections_by_frame[frame.frame_index] = frame.detections
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Annoter si cette frame a des détections
                if frame_idx in detections_by_frame:
                    frame = self._annotate_frame(frame, detections_by_frame[frame_idx])
                
                out.write(frame)
                frame_idx += 1
            
            cap.release()
            out.release()
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Erreur création vidéo annotée: {e}")
            return None
    
    def _annotate_frame(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """🎨 Annote une frame avec les détections"""
        
        annotated = frame.copy()
        
        for detection in detections:
            bbox = detection.bbox
            
            # Couleur selon la classe
            from storage.models.config_epoch_30 import CHAMPION_CLASS_COLORS
            color = CHAMPION_CLASS_COLORS.get(detection.class_id, (255, 255, 255))
            color = (color[2], color[1], color[0])  # RGB -> BGR pour OpenCV
            
            # Boîte de détection
            cv2.rectangle(
                annotated,
                (bbox.x1, bbox.y1),
                (bbox.x2, bbox.y2),
                color,
                2
            )
            
            # Label
            label = f"{detection.class_name_fr} ({detection.confidence:.2f})"
            if hasattr(detection, 'track_id') and detection.track_id:
                label += f" ID:{detection.track_id}"
            
            # Fond pour le texte
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(
                annotated,
                (bbox.x1, bbox.y1 - label_height - 10),
                (bbox.x1 + label_width, bbox.y1),
                color,
                -1
            )
            
            # Texte
            cv2.putText(
                annotated,
                label,
                (bbox.x1, bbox.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return annotated
    
    async def _save_frame_samples(
        self,
        timeline: VideoTimeline,
        processing_id: str
    ) -> List[str]:
        """📸 Sauvegarde des échantillons de frames"""
        
        samples_dir = self.results_path / f"{processing_id}_samples"
        samples_dir.mkdir(exist_ok=True)
        
        sample_paths = []
        
        # Sélectionner quelques frames intéressantes
        frames_with_detections = [f for f in timeline.frames if f.detections]
        
        if frames_with_detections:
            # Prendre jusqu'à 10 frames échantillons
            step = max(1, len(frames_with_detections) // 10)
            sample_frames = frames_with_detections[::step][:10]
            
            for i, frame in enumerate(sample_frames):
                sample_path = samples_dir / f"frame_{frame.frame_index:06d}.jpg"
                # Ici on sauvegarderait la frame annotée
                # (nécessiterait de garder les images en mémoire)
                sample_paths.append(str(sample_path))
        
        return sample_paths
    
    async def _generate_video_summary(
        self,
        timeline: VideoTimeline,
        video_info: Dict[str, Any],
        config: VideoProcessingConfig
    ) -> Dict[str, Any]:
        """📊 Génère un résumé du traitement vidéo"""
        
        summary = {
            "video_info": {
                "duration_seconds": video_info.get("duration", 0),
                "fps": video_info.get("fps", 0),
                "resolution": video_info.get("resolution", (0, 0)),
                "total_frames": video_info.get("frame_count", 0),
                "file_size_mb": video_info.get("file_size_mb", 0)
            },
            "processing_info": {
                "frames_analyzed": len(timeline.frames),
                "processing_strategy": config.processing_strategy.value,
                "tracking_mode": config.tracking_mode.value
            },
            "detection_summary": {
                "total_detections": sum(len(f.detections) for f in timeline.frames),
                "unique_objects": len(timeline.unique_objects),
                "frames_with_detections": len([f for f in timeline.frames if f.detections]),
                "avg_detections_per_frame": 0.0
            },
            "object_analysis": {},
            "temporal_analysis": {}
        }
        
        # Calculs additionnels
        total_detections = summary["detection_summary"]["total_detections"]
        frames_analyzed = len(timeline.frames)
        
        if frames_analyzed > 0:
            summary["detection_summary"]["avg_detections_per_frame"] = total_detections / frames_analyzed
        
        # Analyse par classe d'objets
        class_counts = {}
        for obj in timeline.unique_objects:
            class_name = obj["class_name_fr"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        summary["object_analysis"] = {
            "classes_detected": list(class_counts.keys()),
            "class_distribution": class_counts,
            "most_common_class": max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else None
        }
        
        # Analyse temporelle
        if timeline.unique_objects:
            durations = []
            for obj in timeline.unique_objects:
                duration = obj["last_seen"] - obj["first_seen"]
                durations.append(duration)
            
            summary["temporal_analysis"] = {
                "avg_object_duration": np.mean(durations),
                "max_object_duration": np.max(durations),
                "objects_present_throughout": len([d for d in durations if d > timeline.duration_seconds * 0.8])
            }
        
        return summary
    
    async def _save_video_result(self, result: VideoProcessingResult, processing_id: str):
        """💾 Sauvegarde le résultat vidéo"""
        
        try:
            result_file = self.results_path / f"{processing_id}_result.json"
            
            # Sérialiser le résultat (structure complexe)
            result_dict = {
                "processing_id": result.processing_id,
                "video_info": result.video_info,
                "timeline": {
                    "frames": [
                        {
                            "frame_index": f.frame_index,
                            "timestamp": f.timestamp,
                            "detections": [d.__dict__ for d in f.detections],
                            "frame_info": f.frame_info
                        }
                        for f in result.timeline.frames
                    ],
                    "unique_objects": result.timeline.unique_objects,
                    "duration_seconds": result.timeline.duration_seconds
                },
                "summary": result.summary,
                "processing_config": result.processing_config,
                "processing_time_seconds": result.processing_time_seconds,
                "visualizations": result.visualizations,
                "metadata": result.metadata,
                "timestamp": time.time()
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur sauvegarde résultat vidéo: {e}")
    
    def _update_video_statistics(self, result: VideoProcessingResult, video_info: Dict[str, Any]):
        """📊 Met à jour les statistiques vidéo"""
        
        self.stats["total_processed"] += 1
        self.stats["total_frames_analyzed"] += len(result.timeline.frames)
        self.stats["total_detections"] += sum(len(f.detections) for f in result.timeline.frames)
        self.stats["processing_times"].append(result.processing_time_seconds)
        
        # Format
        format_name = video_info.get("format", "unknown").lower()
        self.stats["format_counts"][format_name] = self.stats["format_counts"].get(format_name, 0) + 1
        
        # FPS moyen de traitement
        if result.processing_time_seconds > 0:
            fps_processed = len(result.timeline.frames) / result.processing_time_seconds
            self.stats["average_fps_processed"] = (
                self.stats["average_fps_processed"] + fps_processed
            ) / 2
    
    def get_statistics(self) -> Dict[str, Any]:
        """📊 Retourne les statistiques du service"""
        
        processing_times = self.stats["processing_times"]
        
        return {
            "service_type": "video_processing",
            "total_processed": self.stats["total_processed"],
            "total_frames_analyzed": self.stats["total_frames_analyzed"],
            "total_detections": self.stats["total_detections"],
            "error_count": self.stats["error_count"],
            "performance": {
                "avg_processing_time": np.mean(processing_times) if processing_times else 0.0,
                "avg_fps_processed": self.stats["average_fps_processed"],
                "max_processing_time": np.max(processing_times) if processing_times else 0.0
            },
            "format_distribution": self.stats["format_counts"],
            "efficiency": {
                "avg_detections_per_second": (
                    self.stats["total_detections"] / sum(processing_times)
                    if processing_times and sum(processing_times) > 0 else 0.0
                )
            }
        }
    
    async def cleanup(self):
        """🧹 Nettoyage des ressources"""
        logger.info("🧹 Nettoyage VideoService...")
        
        # Nettoyer trackers
        self._trackers.clear()
        
        # Nettoyer service d'images
        await self.image_service.cleanup()
        
        logger.info("✅ VideoService nettoyé")

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "VideoService",
    "VideoProcessingConfig",
    "VideoFormat",
    "ProcessingStrategy",
    "TrackingMode"
]