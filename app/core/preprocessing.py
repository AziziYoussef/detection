"""
🔄 PREPROCESSING - PRÉTRAITEMENT INTELLIGENT D'IMAGES
===================================================
Pipeline de prétraitement adaptatif pour optimiser les performances

Fonctionnalités:
- Redimensionnement intelligent avec préservation aspect ratio
- Normalisation selon les modèles (ImageNet standards)
- Augmentation de données pour robustesse
- Optimisations selon le mode de détection
- Support formats multiples (PIL, OpenCV, numpy, torch)
- Pipeline optimisé GPU/CPU
"""

import logging
from typing import Tuple, Union, Optional, List, Any, Dict
from enum import Enum
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance
import cv2

# Imports internes
from app.schemas.detection import DetectionMode
from app.models.model import ModelConfig

logger = logging.getLogger(__name__)

class ResizeMethod(str, Enum):
    """📏 Méthodes de redimensionnement"""
    LETTERBOX = "letterbox"      # Avec padding pour préserver ratio
    STRETCH = "stretch"          # Étirement direct
    CROP = "crop"               # Crop central
    ADAPTIVE = "adaptive"        # Adaptatif selon contenu

class ImageFormat(str, Enum):
    """🖼️ Formats d'image supportés"""
    PIL = "pil"
    OPENCV = "opencv" 
    NUMPY = "numpy"
    TORCH = "torch"

class PreprocessingConfig:
    """⚙️ Configuration du prétraitement"""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        resize_method: ResizeMethod = ResizeMethod.LETTERBOX,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        pad_color: Tuple[int, int, int] = (114, 114, 114),
        quality_enhancement: bool = False,
        device: str = "cpu"
    ):
        self.target_size = target_size
        self.resize_method = resize_method
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.pad_color = pad_color
        self.quality_enhancement = quality_enhancement
        self.device = device

class ImagePreprocessor:
    """🔄 Préprocesseur principal d'images"""
    
    def __init__(self):
        # Configurations par mode de détection
        self.mode_configs = {
            DetectionMode.ULTRA_FAST: PreprocessingConfig(
                target_size=(320, 320),
                resize_method=ResizeMethod.STRETCH,
                quality_enhancement=False
            ),
            DetectionMode.FAST: PreprocessingConfig(
                target_size=(416, 416),
                resize_method=ResizeMethod.LETTERBOX,
                quality_enhancement=False
            ),
            DetectionMode.BALANCED: PreprocessingConfig(
                target_size=(640, 640),
                resize_method=ResizeMethod.LETTERBOX,
                quality_enhancement=False
            ),
            DetectionMode.QUALITY: PreprocessingConfig(
                target_size=(800, 800),
                resize_method=ResizeMethod.LETTERBOX,
                quality_enhancement=True
            ),
            DetectionMode.BATCH: PreprocessingConfig(
                target_size=(640, 640),
                resize_method=ResizeMethod.LETTERBOX,
                quality_enhancement=False
            )
        }
        
        # Cache des transformations
        self._transform_cache: Dict[str, transforms.Compose] = {}
        
        # Statistiques
        self.processing_times: List[float] = []
        self.total_processed = 0
        
        logger.info("🔄 ImagePreprocessor initialisé")
    
    def preprocess_fast(
        self, 
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        model_config: ModelConfig
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """⚡ Prétraitement rapide pour streaming"""
        return self._preprocess_with_mode(image, model_config, DetectionMode.ULTRA_FAST)
    
    def preprocess_balanced(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        model_config: ModelConfig
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """⚖️ Prétraitement équilibré"""
        return self._preprocess_with_mode(image, model_config, DetectionMode.BALANCED)
    
    def preprocess_quality(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        model_config: ModelConfig
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """🎯 Prétraitement haute qualité"""
        return self._preprocess_with_mode(image, model_config, DetectionMode.QUALITY)
    
    def _preprocess_with_mode(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        model_config: ModelConfig,
        mode: DetectionMode
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """🔄 Prétraitement avec mode spécifique"""
        
        start_time = time.time()
        
        try:
            # 1. Détection du format et conversion
            pil_image, original_size = self._to_pil_image(image)
            
            # 2. Configuration selon le mode
            config = self.mode_configs[mode]
            config.target_size = model_config.input_size
            
            # 3. Amélioration qualité si demandée
            if config.quality_enhancement:
                pil_image = self._enhance_image_quality(pil_image)
            
            # 4. Redimensionnement
            processed_image, scale_info = self._resize_image(pil_image, config)
            
            # 5. Conversion en tensor et normalisation
            tensor_image = self._to_tensor_and_normalize(processed_image, config)
            
            # 6. Ajout dimension batch
            if tensor_image.dim() == 3:
                tensor_image = tensor_image.unsqueeze(0)
            
            # 7. Métriques
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.total_processed += 1
            
            logger.debug(f"🔄 Image prétraitée en {processing_time*1000:.1f}ms")
            
            return tensor_image, original_size
            
        except Exception as e:
            logger.error(f"❌ Erreur prétraitement: {e}")
            raise
    
    def _to_pil_image(
        self, 
        image: Union[Image.Image, np.ndarray, torch.Tensor]
    ) -> Tuple[Image.Image, Tuple[int, int]]:
        """🖼️ Convertit vers PIL Image"""
        
        if isinstance(image, Image.Image):
            original_size = image.size  # (width, height)
            return image.convert('RGB'), original_size
            
        elif isinstance(image, np.ndarray):
            # OpenCV BGR → RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                if image.dtype == np.uint8 and np.max(image) <= 255:
                    # Supposer BGR d'OpenCV
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
            else:
                image_rgb = image
            
            # Conversion PIL
            if image_rgb.dtype != np.uint8:
                image_rgb = (image_rgb * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image_rgb)
            original_size = pil_image.size
            return pil_image, original_size
            
        elif isinstance(image, torch.Tensor):
            # Tensor → numpy → PIL
            if image.dim() == 4:  # Batch dimension
                image = image.squeeze(0)
            
            if image.dim() == 3:  # CHW → HWC
                image = image.permute(1, 2, 0)
            
            # Dénormalisation si nécessaire
            if image.dtype == torch.float32 and image.max() <= 1.0:
                image = (image * 255).clamp(0, 255).byte()
            
            numpy_image = image.cpu().numpy()
            pil_image = Image.fromarray(numpy_image)
            original_size = pil_image.size
            return pil_image, original_size
            
        else:
            raise ValueError(f"Format d'image non supporté: {type(image)}")
    
    def _enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """✨ Améliore la qualité d'image"""
        
        try:
            # Auto-contraste léger
            image = ImageOps.autocontrast(image, cutoff=1)
            
            # Netteté légère
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Saturation légère
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.05)
            
            return image
            
        except Exception as e:
            logger.debug(f"⚠️ Erreur amélioration qualité: {e}")
            return image
    
    def _resize_image(
        self, 
        image: Image.Image, 
        config: PreprocessingConfig
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """📏 Redimensionne l'image selon la méthode"""
        
        original_width, original_height = image.size
        target_width, target_height = config.target_size
        
        if config.resize_method == ResizeMethod.LETTERBOX:
            return self._letterbox_resize(image, config.target_size, config.pad_color)
            
        elif config.resize_method == ResizeMethod.STRETCH:
            resized = image.resize(config.target_size, Image.LANCZOS)
            scale_info = {
                "method": "stretch",
                "scale_x": target_width / original_width,
                "scale_y": target_height / original_height
            }
            return resized, scale_info
            
        elif config.resize_method == ResizeMethod.CROP:
            return self._center_crop_resize(image, config.target_size)
            
        elif config.resize_method == ResizeMethod.ADAPTIVE:
            return self._adaptive_resize(image, config.target_size)
            
        else:
            # Fallback vers letterbox
            return self._letterbox_resize(image, config.target_size, config.pad_color)
    
    def _letterbox_resize(
        self, 
        image: Image.Image, 
        target_size: Tuple[int, int],
        pad_color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """📦 Redimensionnement letterbox (préserve ratio)"""
        
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        # Calcul du ratio optimal
        scale = min(target_width / original_width, target_height / original_height)
        
        # Nouvelles dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Redimensionnement
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Padding pour atteindre la taille cible
        padded = Image.new('RGB', target_size, pad_color)
        
        # Centrage
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        
        padded.paste(resized, (paste_x, paste_y))
        
        scale_info = {
            "method": "letterbox",
            "scale": scale,
            "new_size": (new_width, new_height),
            "padding": (paste_x, paste_y),
            "original_size": (original_width, original_height)
        }
        
        return padded, scale_info
    
    def _center_crop_resize(
        self, 
        image: Image.Image, 
        target_size: Tuple[int, int]
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """✂️ Redimensionnement avec crop central"""
        
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        # Calcul pour crop carré centré
        min_dim = min(original_width, original_height)
        
        # Coordonnées de crop
        left = (original_width - min_dim) // 2
        top = (original_height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        
        # Crop et resize
        cropped = image.crop((left, top, right, bottom))
        resized = cropped.resize(target_size, Image.LANCZOS)
        
        scale_info = {
            "method": "center_crop",
            "crop_box": (left, top, right, bottom),
            "scale": target_width / min_dim
        }
        
        return resized, scale_info
    
    def _adaptive_resize(
        self, 
        image: Image.Image, 
        target_size: Tuple[int, int]
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """🤖 Redimensionnement adaptatif intelligent"""
        
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        # Analyse du contenu pour choisir la méthode
        aspect_ratio_original = original_width / original_height
        aspect_ratio_target = target_width / target_height
        
        ratio_diff = abs(aspect_ratio_original - aspect_ratio_target)
        
        if ratio_diff < 0.2:  # Ratios similaires
            # Étirement acceptable
            resized = image.resize(target_size, Image.LANCZOS)
            scale_info = {"method": "adaptive_stretch"}
        else:
            # Letterbox pour préserver le contenu
            resized, scale_info = self._letterbox_resize(image, target_size)
            scale_info["method"] = "adaptive_letterbox"
        
        return resized, scale_info
    
    def _to_tensor_and_normalize(
        self, 
        image: Image.Image, 
        config: PreprocessingConfig
    ) -> torch.Tensor:
        """🔢 Conversion tensor et normalisation"""
        
        # Conversion en tensor
        tensor = transforms.ToTensor()(image)
        
        # Normalisation
        if config.normalize:
            normalize_transform = transforms.Normalize(
                mean=config.mean,
                std=config.std
            )
            tensor = normalize_transform(tensor)
        
        return tensor
    
    def get_preprocessing_pipeline(
        self, 
        model_config: ModelConfig,
        mode: DetectionMode = DetectionMode.BALANCED
    ) -> transforms.Compose:
        """🔧 Récupère un pipeline de transformation torchvision"""
        
        cache_key = f"{model_config.input_size}_{mode.value}"
        
        if cache_key in self._transform_cache:
            return self._transform_cache[cache_key]
        
        config = self.mode_configs[mode]
        config.target_size = model_config.input_size
        
        # Construction du pipeline
        transform_list = []
        
        # Redimensionnement
        if config.resize_method == ResizeMethod.LETTERBOX:
            transform_list.append(
                transforms.Resize(config.target_size, interpolation=transforms.InterpolationMode.BILINEAR)
            )
        else:
            transform_list.append(
                transforms.Resize(config.target_size)
            )
        
        # Conversion tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalisation
        if config.normalize:
            transform_list.append(
                transforms.Normalize(mean=config.mean, std=config.std)
            )
        
        # Pipeline complet
        pipeline = transforms.Compose(transform_list)
        self._transform_cache[cache_key] = pipeline
        
        return pipeline
    
    def preprocess_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, torch.Tensor]],
        model_config: ModelConfig,
        mode: DetectionMode = DetectionMode.BATCH
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """📦 Prétraitement batch optimisé"""
        
        start_time = time.time()
        
        try:
            tensors = []
            original_sizes = []
            
            for image in images:
                tensor, original_size = self._preprocess_with_mode(image, model_config, mode)
                tensors.append(tensor.squeeze(0))  # Enlever batch dim
                original_sizes.append(original_size)
            
            # Stack en batch
            batch_tensor = torch.stack(tensors, dim=0)
            
            processing_time = time.time() - start_time
            logger.info(f"📦 Batch de {len(images)} images prétraité en {processing_time*1000:.1f}ms")
            
            return batch_tensor, original_sizes
            
        except Exception as e:
            logger.error(f"❌ Erreur prétraitement batch: {e}")
            raise
    
    def denormalize_tensor(
        self, 
        tensor: torch.Tensor,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> torch.Tensor:
        """🔄 Dénormalise un tensor pour visualisation"""
        
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
        
        # Dénormalisation
        denormalized = tensor * std + mean
        
        # Clamp pour s'assurer que les valeurs sont dans [0, 1]
        denormalized = torch.clamp(denormalized, 0, 1)
        
        return denormalized
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """📊 Statistiques de performance du préprocesseur"""
        
        if not self.processing_times:
            return {
                "total_processed": 0,
                "average_time_ms": 0.0,
                "min_time_ms": 0.0,
                "max_time_ms": 0.0
            }
        
        avg_time = np.mean(self.processing_times)
        min_time = np.min(self.processing_times)
        max_time = np.max(self.processing_times)
        
        return {
            "total_processed": self.total_processed,
            "average_time_ms": avg_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "throughput_images_per_second": 1.0 / avg_time if avg_time > 0 else 0
        }
    
    def clear_cache(self):
        """🧹 Vide le cache des transformations"""
        self._transform_cache.clear()
        logger.debug("🧹 Cache de transformations vidé")

class PreprocessingPipeline:
    """🔗 Pipeline de prétraitement complet"""
    
    def __init__(self, preprocessor: ImagePreprocessor):
        self.preprocessor = preprocessor
        self.batch_processors = {}
    
    async def process_single_image(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        model_config: ModelConfig,
        mode: DetectionMode = DetectionMode.BALANCED
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """🔄 Traite une image unique"""
        return self.preprocessor._preprocess_with_mode(image, model_config, mode)
    
    async def process_batch_images(
        self,
        images: List[Union[Image.Image, np.ndarray, torch.Tensor]],
        model_config: ModelConfig,
        mode: DetectionMode = DetectionMode.BATCH
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """📦 Traite un batch d'images"""
        return self.preprocessor.preprocess_batch(images, model_config, mode)

# === EXPORTS ===
__all__ = [
    "ImagePreprocessor",
    "PreprocessingPipeline",
    "PreprocessingConfig",
    "ResizeMethod",
    "ImageFormat"
]