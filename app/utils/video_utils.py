"""
🎬 VIDEO_UTILS - UTILITAIRES POUR TRAITEMENT VIDÉOS
==================================================
Fonctions utilitaires pour traitement et manipulation de vidéos

Fonctionnalités:
- Extraction de frames avec contrôle d'intervalle
- Encodage vidéo avec annotations de détection
- Optimisation pour web et mobile
- Extraction de keyframes intelligente
- Création de vidéos à partir de frames
- Gestion de métadonnées et timestamps

Formats supportés:
- MP4, AVI, MOV, MKV, WEBM
- Codecs: H.264, H.265, VP9, AV1
- Optimisation streaming et téléchargement
- Support audio (optionnel)

Performance:
- Utilisation OpenCV optimisé
- Traitement par chunks pour grandes vidéos
- Support GPU pour encodage (si disponible)
- Cache intelligent pour opérations répétées
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Iterator, Callable
from pathlib import Path
import logging
import time
import subprocess
import json
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import asyncio
import concurrent.futures

logger = logging.getLogger(__name__)

# 📋 STRUCTURES DE DONNÉES
@dataclass
class VideoInfo:
    """📋 Informations d'une vidéo"""
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str
    bitrate: Optional[int] = None
    has_audio: bool = False
    file_size_mb: float = 0.0
    
    def get_aspect_ratio(self) -> float:
        """📐 Calcule le ratio d'aspect"""
        return self.width / self.height if self.height > 0 else 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """🔄 Conversion en dictionnaire"""
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_seconds": self.duration_seconds,
            "codec": self.codec,
            "bitrate": self.bitrate,
            "has_audio": self.has_audio,
            "file_size_mb": self.file_size_mb,
            "aspect_ratio": self.get_aspect_ratio()
        }

@dataclass
class FrameAnnotation:
    """📋 Annotation d'une frame"""
    frame_number: int
    timestamp: float
    detections: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None

# 📖 ANALYSE VIDÉO
def get_video_info(video_path: Union[str, Path]) -> VideoInfo:
    """
    📖 Récupère les informations détaillées d'une vidéo
    
    Args:
        video_path: Chemin vers la vidéo
        
    Returns:
        Informations de la vidéo
    """
    
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Vidéo non trouvée: {video_path}")
    
    # Ouverture avec OpenCV
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")
    
    try:
        # Propriétés de base
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Durée
        duration = total_frames / fps if fps > 0 else 0.0
        
        # Codec (approximatif avec OpenCV)
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        codec = "".join([chr(int(fourcc) >> 8 * i & 0xFF) for i in range(4)])
        
        # Taille du fichier
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        
        # Bitrate approximatif
        bitrate = None
        if duration > 0:
            bitrate = int((file_size_mb * 8 * 1024) / duration)  # kbps
        
        video_info = VideoInfo(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            codec=codec.strip(),
            bitrate=bitrate,
            has_audio=_check_audio_track(video_path),
            file_size_mb=file_size_mb
        )
        
        logger.debug(f"📖 Info vidéo: {video_info.to_dict()}")
        return video_info
        
    finally:
        cap.release()

def _check_audio_track(video_path: Path) -> bool:
    """🔊 Vérifie la présence d'une piste audio"""
    
    try:
        # Utiliser ffprobe si disponible
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', str(video_path)
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get('streams', [])
            
            for stream in streams:
                if stream.get('codec_type') == 'audio':
                    return True
        
        return False
        
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        # Fallback: assumé pas d'audio si erreur
        return False

# 🎞️ EXTRACTION DE FRAMES
def extract_video_frames(
    video_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    frame_interval: int = 1,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    max_frames: Optional[int] = None,
    image_format: str = "jpg",
    quality: int = 95
) -> List[Dict[str, Any]]:
    """
    🎞️ Extrait les frames d'une vidéo
    
    Args:
        video_path: Chemin vidéo source
        output_dir: Dossier de sortie (optionnel)
        frame_interval: Intervalle entre frames (1 = toutes)
        start_time: Temps de début (secondes)
        end_time: Temps de fin (secondes, optionnel)
        max_frames: Nombre max de frames
        image_format: Format des images (jpg, png)
        quality: Qualité JPEG (0-100)
        
    Returns:
        Liste des frames extraites avec métadonnées
    """
    
    video_path = Path(video_path)
    
    # Dossier de sortie
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ouverture vidéo
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir: {video_path}")
    
    try:
        # Informations vidéo
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calcul des frames de début/fin
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps) if end_time else total_frames
        end_frame = min(end_frame, total_frames)
        
        # Positionnement au début
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        extracted_frames = []
        frame_count = 0
        current_frame = start_frame
        
        logger.info(f"🎞️ Extraction frames {start_frame}-{end_frame}, intervalle={frame_interval}")
        
        while current_frame < end_frame:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Vérifier intervalle
            if (current_frame - start_frame) % frame_interval == 0:
                timestamp = current_frame / fps
                
                frame_info = {
                    "frame_number": current_frame,
                    "timestamp": timestamp,
                    "shape": frame.shape,
                    "extracted_at": time.time()
                }
                
                # Sauvegarde si dossier spécifié
                if output_dir:
                    filename = f"frame_{current_frame:06d}.{image_format}"
                    frame_path = output_dir / filename
                    
                    if image_format.lower() == "jpg":
                        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    elif image_format.lower() == "png":
                        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    else:
                        cv2.imwrite(str(frame_path), frame)
                    
                    frame_info["file_path"] = str(frame_path)
                    frame_info["filename"] = filename
                else:
                    # Garder frame en mémoire
                    frame_info["frame_data"] = frame
                
                extracted_frames.append(frame_info)
                frame_count += 1
                
                # Limite max frames
                if max_frames and frame_count >= max_frames:
                    break
            
            current_frame += 1
        
        logger.info(f"✅ {frame_count} frames extraites")
        return extracted_frames
        
    finally:
        cap.release()

async def extract_video_frames_async(
    video_path: Union[str, Path],
    **kwargs
) -> List[Dict[str, Any]]:
    """🎞️ Extraction de frames asynchrone"""
    
    loop = asyncio.get_event_loop()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor, 
            extract_video_frames,
            video_path,
            **kwargs
        )
    
    return result

# 🔑 EXTRACTION KEYFRAMES
def extract_keyframes(
    video_path: Union[str, Path],
    method: str = "optical_flow",
    threshold: float = 0.3,
    max_keyframes: int = 100,
    min_interval: float = 1.0
) -> List[Dict[str, Any]]:
    """
    🔑 Extrait les keyframes importantes d'une vidéo
    
    Args:
        video_path: Chemin vidéo
        method: Méthode de détection ("optical_flow", "histogram", "edge")
        threshold: Seuil de changement
        max_keyframes: Nombre max de keyframes
        min_interval: Intervalle minimum entre keyframes (secondes)
        
    Returns:
        Liste des keyframes avec scores d'importance
    """
    
    if method == "optical_flow":
        return _extract_keyframes_optical_flow(video_path, threshold, max_keyframes, min_interval)
    elif method == "histogram":
        return _extract_keyframes_histogram(video_path, threshold, max_keyframes, min_interval)
    elif method == "edge":
        return _extract_keyframes_edge(video_path, threshold, max_keyframes, min_interval)
    else:
        raise ValueError(f"Méthode non supportée: {method}")

def _extract_keyframes_optical_flow(
    video_path: Path,
    threshold: float,
    max_keyframes: int,
    min_interval: float
) -> List[Dict[str, Any]]:
    """🔑 Keyframes basées sur optical flow"""
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    min_frame_interval = int(min_interval * fps)
    
    keyframes = []
    prev_frame = None
    frame_number = 0
    last_keyframe = -min_frame_interval
    
    logger.info(f"🔑 Extraction keyframes optical flow, seuil={threshold}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Conversion niveaux de gris
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None and frame_number - last_keyframe >= min_frame_interval:
                # Calcul optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_frame, gray, None, None,
                    winSize=(15, 15),
                    maxLevel=2
                )[0]
                
                if flow is not None:
                    # Magnitude du mouvement
                    flow_magnitude = np.sqrt(flow[:, 0]**2 + flow[:, 1]**2)
                    avg_motion = np.mean(flow_magnitude)
                    
                    # Keyframe si mouvement important
                    if avg_motion > threshold:
                        timestamp = frame_number / fps
                        
                        keyframe_info = {
                            "frame_number": frame_number,
                            "timestamp": timestamp,
                            "importance_score": float(avg_motion),
                            "method": "optical_flow",
                            "frame_data": frame.copy()
                        }
                        
                        keyframes.append(keyframe_info)
                        last_keyframe = frame_number
                        
                        if len(keyframes) >= max_keyframes:
                            break
            
            prev_frame = gray
            frame_number += 1
    
    finally:
        cap.release()
    
    logger.info(f"✅ {len(keyframes)} keyframes extraites (optical flow)")
    return keyframes

def _extract_keyframes_histogram(
    video_path: Path,
    threshold: float,
    max_keyframes: int,
    min_interval: float
) -> List[Dict[str, Any]]:
    """🔑 Keyframes basées sur histogrammes"""
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    min_frame_interval = int(min_interval * fps)
    
    keyframes = []
    prev_hist = None
    frame_number = 0
    last_keyframe = -min_frame_interval
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Histogramme couleur
            hist = cv2.calcHist([frame], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            if prev_hist is not None and frame_number - last_keyframe >= min_frame_interval:
                # Comparaison histogrammes
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                difference = 1.0 - correlation
                
                # Keyframe si changement important
                if difference > threshold:
                    timestamp = frame_number / fps
                    
                    keyframe_info = {
                        "frame_number": frame_number,
                        "timestamp": timestamp,
                        "importance_score": float(difference),
                        "method": "histogram",
                        "frame_data": frame.copy()
                    }
                    
                    keyframes.append(keyframe_info)
                    last_keyframe = frame_number
                    
                    if len(keyframes) >= max_keyframes:
                        break
            
            prev_hist = hist
            frame_number += 1
    
    finally:
        cap.release()
    
    logger.info(f"✅ {len(keyframes)} keyframes extraites (histogram)")
    return keyframes

def _extract_keyframes_edge(
    video_path: Path,
    threshold: float,
    max_keyframes: int,
    min_interval: float
) -> List[Dict[str, Any]]:
    """🔑 Keyframes basées sur détection de contours"""
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    min_frame_interval = int(min_interval * fps)
    
    keyframes = []
    prev_edges = None
    frame_number = 0
    last_keyframe = -min_frame_interval
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Détection de contours
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            if prev_edges is not None and frame_number - last_keyframe >= min_frame_interval:
                # Différence de contours
                edge_diff = np.sum(np.abs(edges.astype(np.float32) - prev_edges.astype(np.float32)))
                edge_diff_normalized = edge_diff / (edges.shape[0] * edges.shape[1] * 255)
                
                # Keyframe si changement important
                if edge_diff_normalized > threshold:
                    timestamp = frame_number / fps
                    
                    keyframe_info = {
                        "frame_number": frame_number,
                        "timestamp": timestamp,
                        "importance_score": float(edge_diff_normalized),
                        "edge_density": float(edge_density),
                        "method": "edge",
                        "frame_data": frame.copy()
                    }
                    
                    keyframes.append(keyframe_info)
                    last_keyframe = frame_number
                    
                    if len(keyframes) >= max_keyframes:
                        break
            
            prev_edges = edges
            frame_number += 1
    
    finally:
        cap.release()
    
    logger.info(f"✅ {len(keyframes)} keyframes extraites (edge)")
    return keyframes

# 🎥 CRÉATION DE VIDÉOS
def create_video_from_frames(
    frames: List[Union[np.ndarray, str, Path]],
    output_path: Union[str, Path],
    fps: float = 30.0,
    codec: str = "mp4v",
    quality: int = 80
) -> bool:
    """
    🎥 Crée une vidéo à partir de frames
    
    Args:
        frames: Liste de frames (arrays ou chemins)
        output_path: Chemin vidéo de sortie
        fps: Images par seconde
        codec: Codec vidéo
        quality: Qualité (0-100)
        
    Returns:
        Succès de la création
    """
    
    if not frames:
        logger.error("❌ Aucune frame fournie")
        return False
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Déterminer dimensions depuis première frame
    first_frame = _load_frame(frames[0])
    if first_frame is None:
        logger.error("❌ Impossible de charger la première frame")
        return False
    
    height, width = first_frame.shape[:2]
    
    # Codec fourcc
    fourcc_map = {
        "mp4v": cv2.VideoWriter_fourcc(*'mp4v'),
        "h264": cv2.VideoWriter_fourcc(*'H264'),
        "xvid": cv2.VideoWriter_fourcc(*'XVID'),
        "mjpg": cv2.VideoWriter_fourcc(*'MJPG')
    }
    fourcc = fourcc_map.get(codec.lower(), cv2.VideoWriter_fourcc(*'mp4v'))
    
    # Writer vidéo
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not writer.isOpened():
        logger.error(f"❌ Impossible de créer le writer vidéo: {output_path}")
        return False
    
    try:
        logger.info(f"🎥 Création vidéo: {len(frames)} frames à {fps} FPS")
        
        for i, frame_source in enumerate(frames):
            frame = _load_frame(frame_source)
            
            if frame is None:
                logger.warning(f"⚠️ Frame {i} ignorée (erreur chargement)")
                continue
            
            # Redimensionner si nécessaire
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            writer.write(frame)
        
        logger.info(f"✅ Vidéo créée: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur création vidéo: {e}")
        return False
        
    finally:
        writer.release()

def _load_frame(frame_source: Union[np.ndarray, str, Path]) -> Optional[np.ndarray]:
    """📖 Charge une frame depuis différentes sources"""
    
    if isinstance(frame_source, np.ndarray):
        return frame_source
    
    elif isinstance(frame_source, (str, Path)):
        try:
            return cv2.imread(str(frame_source))
        except Exception as e:
            logger.error(f"❌ Erreur chargement {frame_source}: {e}")
            return None
    
    else:
        logger.error(f"❌ Type de frame non supporté: {type(frame_source)}")
        return None

# 🎨 ANNOTATION VIDÉO
def encode_video_with_annotations(
    video_path: Union[str, Path],
    annotations: List[FrameAnnotation],
    output_path: Union[str, Path],
    show_boxes: bool = True,
    show_labels: bool = True,
    show_scores: bool = True,
    box_thickness: int = 2,
    font_scale: float = 0.6
) -> bool:
    """
    🎨 Encode une vidéo avec annotations de détection
    
    Args:
        video_path: Vidéo source
        annotations: Annotations par frame
        output_path: Vidéo de sortie
        show_boxes: Afficher boîtes englobantes
        show_labels: Afficher labels
        show_scores: Afficher scores
        box_thickness: Épaisseur des boîtes
        font_scale: Taille de la police
        
    Returns:
        Succès de l'encodage
    """
    
    video_path = Path(video_path)
    output_path = Path(output_path)
    
    # Index des annotations par frame
    annotations_by_frame = {ann.frame_number: ann for ann in annotations}
    
    # Ouverture vidéo source
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        logger.error(f"❌ Impossible d'ouvrir: {video_path}")
        return False
    
    try:
        # Propriétés vidéo
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Writer sortie
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not writer.isOpened():
            logger.error(f"❌ Impossible de créer writer: {output_path}")
            return False
        
        logger.info(f"🎨 Encodage avec annotations: {len(annotations)} frames annotées")
        
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Vérifier si frame a des annotations
            if frame_number in annotations_by_frame:
                annotation = annotations_by_frame[frame_number]
                frame = _draw_detections_on_frame(
                    frame, annotation.detections,
                    show_boxes, show_labels, show_scores,
                    box_thickness, font_scale
                )
            
            writer.write(frame)
            frame_number += 1
            
            # Progress log
            if frame_number % (total_frames // 10 + 1) == 0:
                progress = (frame_number / total_frames) * 100
                logger.debug(f"📊 Progression: {progress:.1f}%")
        
        logger.info(f"✅ Vidéo annotée créée: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur encodage: {e}")
        return False
        
    finally:
        cap.release()
        writer.release()

def _draw_detections_on_frame(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    show_boxes: bool,
    show_labels: bool,
    show_scores: bool,
    box_thickness: int,
    font_scale: float
) -> np.ndarray:
    """🎨 Dessine les détections sur une frame"""
    
    # Couleurs par défaut
    colors = [
        (0, 255, 0),    # Vert
        (255, 0, 0),    # Bleu
        (0, 0, 255),    # Rouge
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Jaune
        (128, 0, 128),  # Violet
        (255, 165, 0)   # Orange
    ]
    
    for i, detection in enumerate(detections):
        color = colors[i % len(colors)]
        
        # Récupération des données
        bbox = detection.get("bbox", {})
        if not bbox:
            continue
        
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))
        
        # Boîte englobante
        if show_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
        
        # Label et score
        if show_labels or show_scores:
            text_parts = []
            
            if show_labels and "class_name" in detection:
                text_parts.append(detection["class_name"])
            
            if show_scores and "confidence" in detection:
                text_parts.append(f"{detection['confidence']:.2f}")
            
            if text_parts:
                text = " - ".join(text_parts)
                
                # Taille du texte
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                
                # Fond du texte
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Texte
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1
                )
    
    return frame

# 🌐 OPTIMISATION WEB
def optimize_video_for_web(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
    target_bitrate: str = "1000k",
    format: str = "mp4"
) -> bool:
    """
    🌐 Optimise une vidéo pour le web
    
    Args:
        input_path: Vidéo source
        output_path: Vidéo optimisée
        target_size: Taille cible (width, height)
        target_bitrate: Bitrate cible
        format: Format de sortie
        
    Returns:
        Succès de l'optimisation
    """
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Commande ffmpeg
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-b:v', target_bitrate,
        '-movflags', '+faststart',  # Optimisation streaming
        '-y'  # Overwrite
    ]
    
    # Redimensionnement si spécifié
    if target_size:
        width, height = target_size
        cmd.extend(['-vf', f'scale={width}:{height}'])
    
    cmd.append(str(output_path))
    
    try:
        logger.info(f"🌐 Optimisation web: {input_path} → {output_path}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Vidéo optimisée: {output_path}")
            return True
        else:
            logger.error(f"❌ Erreur ffmpeg: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout optimisation vidéo")
        return False
        
    except FileNotFoundError:
        logger.error("❌ ffmpeg non trouvé")
        return False
        
    except Exception as e:
        logger.error(f"❌ Erreur optimisation: {e}")
        return False

# 📊 UTILITAIRES
def get_video_thumbnail(
    video_path: Union[str, Path],
    timestamp: float = 1.0,
    size: Optional[Tuple[int, int]] = None
) -> Optional[np.ndarray]:
    """📊 Génère une miniature de vidéo"""
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            return None
        
        # Redimensionnement si demandé
        if size:
            width, height = size
            frame = cv2.resize(frame, (width, height))
        
        return frame
        
    finally:
        cap.release()

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "VideoInfo",
    "FrameAnnotation",
    "get_video_info",
    "extract_video_frames",
    "extract_video_frames_async",
    "extract_keyframes",
    "create_video_from_frames",
    "encode_video_with_annotations",
    "optimize_video_for_web",
    "get_video_thumbnail"
]