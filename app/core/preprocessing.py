"""
🔄 PREPROCESSING - PRÉTRAITEMENT IMAGES ET VIDÉOS
===============================================
Module de prétraitement optimisé pour la détection d'objets perdus

Fonctionnalités:
- Prétraitement adaptatif selon source (webcam, upload, batch)
- Optimisation performance (3 modes: FAST, BALANCED, QUALITY)
- Support multi-formats (JPEG, PNG, WEBP, MP4, AVI, etc.)
- Transformations avancées (augmentation, normalisation)
- Pipeline configurable par modèle
- Cache intelligent des transformations

Modes de prétraitement:
- FAST: Optimisé streaming (latence minimale)
- BALANCED: Équilibre qualité/vitesse (usage général)
- QUALITY: Qualité maximale (précision optimale)

Architecture:
- ImagePreprocessor: Images statiques et frames
- VideoPreprocessor: Vidéos complètes
- PreprocessingPipeline: Pipeline configurable
- Optimisations GPU/CPU automatiques
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
import io

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import cv2

# Imports internes
from app.config.config import get_settings

logger = logging.getLogger(__name__)

# 📋 ÉNUMÉRATIONS ET TYPES
class PreprocessingMode(str, Enum):
    """🎛️ Modes de prétraitement"""
    ULTRA_FAST = "ultra_fast"    # Streaming temps réel
    FAST = "fast"                # Rapide
    BALANCED = "balanced"        # Équilibré
    QUALITY = "quality"          # Qualité maximale
    BATCH = "batch"              # Optimisé batch

class ImageFormat(str, Enum):
    """🖼️ Formats d'image supportés"""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    BMP = "bmp"
    TIFF = "tiff"

class ResizeMethod(str, Enum):
    """📐 Méthodes de redimensionnement"""
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    NEAREST = "nearest"
    LANCZOS = "lanczos"
    AREA = "area"

@dataclass
class PreprocessingConfig:
    """⚙️ Configuration de prétraitement"""
    # Dimensions
    input_size: Tuple[int, int] = (640, 640)
    maintain_aspect_ratio: bool = True
    resize_method: ResizeMethod = ResizeMethod.BILINEAR
    
    # Normalisation
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    normalize: bool = True
    
    # Optimisations
    mode: PreprocessingMode = PreprocessingMode.BALANCED
    use_gpu: bool = True
    half_precision: bool = False
    
    # Augmentations (pour training/validation)
    enable_augmentation: bool = False
    brightness_factor: float = 0.1
    contrast_factor: float = 0.1
    saturation_factor: float = 0.1
    
    # Qualité
    jpeg_quality: int = 95
    png_compress_level: int = 6

# 🖼️ PRÉPROCESSEUR D'IMAGES
class ImagePreprocessor:
    """🖼️ Préprocesseur spécialisé pour images statiques et frames"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.settings = get_settings()
        
        # Cache des transformations
        self._transform_cache: Dict[str, transforms.Compose] = {}
        
        # Statistiques
        self.processed_count = 0
        self.total_processing_time = 0.0
        
        logger.info(f"🖼️ ImagePreprocessor initialisé sur {self.device}")
    
    def preprocess_fast(
        self, 
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        model_config: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """⚡ Prétraitement rapide pour streaming"""
        
        config = PreprocessingConfig(
            input_size=tuple(model_config.get("input_size", (640, 640))),
            mean=tuple(model_config.get("mean", (0.485, 0.456, 0.406))),
            std=tuple(model_config.get("std", (0.229, 0.224, 0.225))),
            mode=PreprocessingMode.FAST,
            resize_method=ResizeMethod.BILINEAR,
            maintain_aspect_ratio=False  # Plus rapide
        )
        
        return self._preprocess_with_config(image, config)
    
    def preprocess_balanced(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor], 
        model_config: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """⚖️ Prétraitement équilibré"""
        
        config = PreprocessingConfig(
            input_size=tuple(model_config.get("input_size", (640, 640))),
            mean=tuple(model_config.get("mean", (0.485, 0.456, 0.406))),
            std=tuple(model_config.get("std", (0.229, 0.224, 0.225))),
            mode=PreprocessingMode.BALANCED,
            resize_method=ResizeMethod.BILINEAR,
            maintain_aspect_ratio=True
        )
        
        return self._preprocess_with_config(image, config)
    
    def preprocess_quality(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        model_config: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """💎 Prétraitement qualité maximale"""
        
        config = PreprocessingConfig(
            input_size=tuple(model_config.get("input_size", (640, 640))),
            mean=tuple(model_config.get("mean", (0.485, 0.456, 0.406))),
            std=tuple(model_config.get("std", (0.229, 0.224, 0.225))),
            mode=PreprocessingMode.QUALITY,
            resize_method=ResizeMethod.BICUBIC,
            maintain_aspect_ratio=True,
            enable_augmentation=False
        )
        
        return self._preprocess_with_config(image, config)
    
    def _preprocess_with_config(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        config: PreprocessingConfig
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """🔄 Prétraitement avec configuration spécifique"""
        
        start_time = time.time()
        
        try:
            # 1. Conversion vers PIL Image
            pil_image, original_size = self._to_pil_image(image)
            
            # 2. Transformations de base
            processed_image = self._apply_basic_transforms(pil_image, config)
            
            # 3. Conversion en tensor
            tensor = self._to_tensor(processed_image, config)
            
            # 4. Normalisation
            if config.normalize:
                tensor = self._normalize_tensor(tensor, config)
            
            # 5. Ajout dimension batch si nécessaire
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            
            # Statistiques
            processing_time = time.time() - start_time
            self.processed_count += 1
            self.total_processing_time += processing_time
            
            return tensor, original_size
            
        except Exception as e:
            logger.error(f"❌ Erreur prétraitement: {e}")
            raise
    
    def _to_pil_image(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Tuple[Image.Image, Tuple[int, int]]:
        """🔄 Conversion vers PIL Image"""
        
        if isinstance(image, Image.Image):
            original_size = image.size
            return image.convert("RGB"), original_size
        
        elif isinstance(image, np.ndarray):
            # NumPy array
            if image.ndim == 3:
                if image.shape[2] == 3:  # RGB
                    pil_image = Image.fromarray(image, mode="RGB")
                elif image.shape[2] == 4:  # RGBA
                    pil_image = Image.fromarray(image, mode="RGBA").convert("RGB")
                else:
                    raise ValueError(f"Format numpy non supporté: {image.shape}")
            elif image.ndim == 2:  # Grayscale
                pil_image = Image.fromarray(image, mode="L").convert("RGB")
            else:
                raise ValueError(f"Dimensions numpy non supportées: {image.ndim}")
            
            original_size = (image.shape[1], image.shape[0])  # width, height
            return pil_image, original_size
        
        elif isinstance(image, torch.Tensor):
            # Tensor PyTorch
            if image.dim() == 4:  # Batch
                image = image.squeeze(0)
            
            if image.dim() == 3:
                # Channel first → Channel last
                if image.shape[0] in [1, 3, 4]:
                    image = image.permute(1, 2, 0)
                
                # Conversion numpy puis PIL
                np_image = image.cpu().numpy()
                if np_image.dtype != np.uint8:
                    np_image = (np_image * 255).astype(np.uint8)
                
                pil_image = Image.fromarray(np_image)
                original_size = (image.shape[1], image.shape[0])
                return pil_image.convert("RGB"), original_size
            
            else:
                raise ValueError(f"Dimensions tensor non supportées: {image.shape}")
        
        else:
            raise TypeError(f"Type d'image non supporté: {type(image)}")
    
    def _apply_basic_transforms(self, image: Image.Image, config: PreprocessingConfig) -> Image.Image:
        """🔄 Applique les transformations de base"""
        
        # Redimensionnement
        if config.maintain_aspect_ratio:
            image = self._resize_with_aspect_ratio(image, config.input_size, config.resize_method)
        else:
            image = self._resize_direct(image, config.input_size, config.resize_method)
        
        # Augmentations si activées
        if config.enable_augmentation:
            image = self._apply_augmentations(image, config)
        
        return image
    
    def _resize_with_aspect_ratio(
        self, 
        image: Image.Image, 
        target_size: Tuple[int, int],
        resize_method: ResizeMethod
    ) -> Image.Image:
        """📐 Redimensionnement avec conservation du ratio"""
        
        target_width, target_height = target_size
        original_width, original_height = image.size
        
        # Calcul du ratio
        ratio = min(target_width / original_width, target_height / original_height)
        
        # Nouvelles dimensions
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # Redimensionnement
        resample_map = {
            ResizeMethod.BILINEAR: Image.BILINEAR,
            ResizeMethod.BICUBIC: Image.BICUBIC,
            ResizeMethod.NEAREST: Image.NEAREST,
            ResizeMethod.LANCZOS: Image.LANCZOS
        }
        
        resized = image.resize((new_width, new_height), resample_map[resize_method])
        
        # Padding pour atteindre la taille cible
        if new_width != target_width or new_height != target_height:
            # Créer image avec padding
            padded = Image.new("RGB", target_size, (114, 114, 114))  # Gris foncé
            
            # Centrer l'image
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            padded.paste(resized, (x_offset, y_offset))
            return padded
        
        return resized
    
    def _resize_direct(
        self,
        image: Image.Image,
        target_size: Tuple[int, int], 
        resize_method: ResizeMethod
    ) -> Image.Image:
        """📐 Redimensionnement direct (peut déformer)"""
        
        resample_map = {
            ResizeMethod.BILINEAR: Image.BILINEAR,
            ResizeMethod.BICUBIC: Image.BICUBIC,
            ResizeMethod.NEAREST: Image.NEAREST,
            ResizeMethod.LANCZOS: Image.LANCZOS
        }
        
        return image.resize(target_size, resample_map[resize_method])
    
    def _apply_augmentations(self, image: Image.Image, config: PreprocessingConfig) -> Image.Image:
        """🎨 Applique les augmentations d'image"""
        
        # Luminosité
        if config.brightness_factor > 0:
            brightness_factor = 1.0 + np.random.uniform(-config.brightness_factor, config.brightness_factor)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)
        
        # Contraste
        if config.contrast_factor > 0:
            contrast_factor = 1.0 + np.random.uniform(-config.contrast_factor, config.contrast_factor)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)
        
        # Saturation
        if config.saturation_factor > 0:
            saturation_factor = 1.0 + np.random.uniform(-config.saturation_factor, config.saturation_factor)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation_factor)
        
        return image
    
    def _to_tensor(self, image: Image.Image, config: PreprocessingConfig) -> torch.Tensor:
        """🔄 Conversion PIL → Tensor"""
        
        # Conversion standard PIL → Tensor
        tensor = TF.to_tensor(image)
        
        # Déplacer vers device si nécessaire
        if config.use_gpu and self.device.type != "cpu":
            tensor = tensor.to(self.device)
        
        # Half precision si activée
        if config.half_precision and self.device.type == "cuda":
            tensor = tensor.half()
        
        return tensor
    
    def _normalize_tensor(self, tensor: torch.Tensor, config: PreprocessingConfig) -> torch.Tensor:
        """📏 Normalisation du tensor"""
        
        mean = torch.tensor(config.mean, device=tensor.device, dtype=tensor.dtype)
        std = torch.tensor(config.std, device=tensor.device, dtype=tensor.dtype)
        
        # Reshape pour broadcasting
        if tensor.dim() == 4:  # Batch
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1)
        else:  # Single image
            mean = mean.view(3, 1, 1)
            std = std.view(3, 1, 1)
        
        return (tensor - mean) / std
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """📊 Statistiques de performance"""
        
        avg_time = self.total_processing_time / max(1, self.processed_count)
        
        return {
            "processed_count": self.processed_count,
            "total_time": self.total_processing_time,
            "average_time_ms": avg_time * 1000,
            "throughput_fps": 1.0 / avg_time if avg_time > 0 else 0.0,
            "device": str(self.device)
        }

# 🎬 PRÉPROCESSEUR VIDÉO
class VideoPreprocessor:
    """🎬 Préprocesseur spécialisé pour vidéos"""
    
    def __init__(self, image_preprocessor: ImagePreprocessor = None):
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()
        self.settings = get_settings()
        
        # Statistiques
        self.videos_processed = 0
        self.frames_processed = 0
        self.total_processing_time = 0.0
        
        logger.info("🎬 VideoPreprocessor initialisé")
    
    async def extract_frames(
        self,
        video_path: Union[str, Path],
        frame_interval: int = 1,
        max_frames: Optional[int] = None,
        model_config: Dict[str, Any] = None
    ) -> List[Tuple[torch.Tensor, float]]:
        """🎞️ Extrait et prétraite les frames d'une vidéo"""
        
        start_time = time.time()
        frames_data = []
        
        try:
            # Ouverture vidéo
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")
            
            # Informations vidéo
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"🎬 Extraction frames: {total_frames} frames à {fps} FPS")
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Intervalle de frames
                if frame_count % frame_interval == 0:
                    # Conversion BGR → RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Prétraitement de la frame
                    processed_frame, _ = self.image_preprocessor.preprocess_balanced(
                        frame_rgb, model_config or {}
                    )
                    
                    # Timestamp
                    timestamp = frame_count / fps
                    
                    frames_data.append((processed_frame, timestamp))
                    extracted_count += 1
                    
                    # Limite max frames
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
            
            cap.release()
            
            # Statistiques
            processing_time = time.time() - start_time
            self.videos_processed += 1
            self.frames_processed += extracted_count
            self.total_processing_time += processing_time
            
            logger.info(
                f"✅ {extracted_count} frames extraites en {processing_time:.2f}s "
                f"({extracted_count/processing_time:.1f} FPS)"
            )
            
            return frames_data
            
        except Exception as e:
            logger.error(f"❌ Erreur extraction frames: {e}")
            raise
    
    async def preprocess_video_batch(
        self,
        video_frames: List[np.ndarray],
        model_config: Dict[str, Any],
        batch_size: int = 8
    ) -> List[torch.Tensor]:
        """📦 Prétraitement batch de frames vidéo"""
        
        processed_frames = []
        
        # Traitement par batches
        for i in range(0, len(video_frames), batch_size):
            batch_frames = video_frames[i:i + batch_size]
            
            # Traitement parallèle du batch
            batch_tasks = [
                asyncio.to_thread(
                    self.image_preprocessor.preprocess_balanced,
                    frame,
                    model_config
                )
                for frame in batch_frames
            ]
            
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Extraction des tensors (ignorer original_size)
            batch_tensors = [result[0] for result in batch_results]
            processed_frames.extend(batch_tensors)
        
        return processed_frames
    
    def get_video_info(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """📋 Récupère les informations d'une vidéo"""
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                return {"error": f"Impossible d'ouvrir: {video_path}"}
            
            info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
                "codec": int(cap.get(cv2.CAP_PROP_FOURCC))
            }
            
            cap.release()
            return info
            
        except Exception as e:
            return {"error": str(e)}

# 🔧 PIPELINE CONFIGURABLE
class PreprocessingPipeline:
    """🔧 Pipeline de prétraitement configurable"""
    
    def __init__(self):
        self.image_preprocessor = ImagePreprocessor()
        self.video_preprocessor = VideoPreprocessor(self.image_preprocessor)
        self.pipelines: Dict[str, Callable] = {}
        
        # Enregistrer pipelines par défaut
        self._register_default_pipelines()
        
        logger.info("🔧 PreprocessingPipeline initialisé")
    
    def _register_default_pipelines(self):
        """📋 Enregistre les pipelines par défaut"""
        
        self.pipelines.update({
            "image_fast": self.image_preprocessor.preprocess_fast,
            "image_balanced": self.image_preprocessor.preprocess_balanced,
            "image_quality": self.image_preprocessor.preprocess_quality,
            "video_extract": self.video_preprocessor.extract_frames,
            "video_batch": self.video_preprocessor.preprocess_video_batch
        })
    
    def register_pipeline(self, name: str, pipeline_func: Callable):
        """📝 Enregistre un pipeline personnalisé"""
        self.pipelines[name] = pipeline_func
        logger.info(f"📝 Pipeline enregistré: {name}")
    
    def get_pipeline(self, name: str) -> Optional[Callable]:
        """🔍 Récupère un pipeline par nom"""
        return self.pipelines.get(name)
    
    def list_pipelines(self) -> List[str]:
        """📋 Liste tous les pipelines disponibles"""
        return list(self.pipelines.keys())
    
    async def process_with_pipeline(
        self,
        pipeline_name: str,
        data: Any,
        **kwargs
    ) -> Any:
        """🔄 Exécute un pipeline spécifique"""
        
        pipeline = self.get_pipeline(pipeline_name)
        
        if not pipeline:
            raise ValueError(f"Pipeline inconnu: {pipeline_name}")
        
        try:
            # Exécution du pipeline
            if asyncio.iscoroutinefunction(pipeline):
                return await pipeline(data, **kwargs)
            else:
                return pipeline(data, **kwargs)
                
        except Exception as e:
            logger.error(f"❌ Erreur pipeline {pipeline_name}: {e}")
            raise
    
    def get_global_stats(self) -> Dict[str, Any]:
        """📊 Statistiques globales"""
        
        return {
            "image_processor": self.image_preprocessor.get_performance_stats(),
            "video_processor": {
                "videos_processed": self.video_preprocessor.videos_processed,
                "frames_processed": self.video_preprocessor.frames_processed,
                "total_time": self.video_preprocessor.total_processing_time
            },
            "available_pipelines": self.list_pipelines()
        }

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "ImagePreprocessor",
    "VideoPreprocessor",
    "PreprocessingPipeline",
    "PreprocessingConfig",
    "PreprocessingMode",
    "ResizeMethod"
]