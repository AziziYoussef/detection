"""
📸 IMAGE SERVICE - SERVICE DE TRAITEMENT D'IMAGES STATIQUES
===========================================================
Service spécialisé pour la détection d'objets perdus dans des images statiques

Fonctionnalités:
- Support multi-formats (JPG, PNG, BMP, TIFF, WebP)
- Validation et optimisation automatique des images
- Détection avec sélection automatique du meilleur modèle
- Post-traitement et filtrage des résultats
- Sauvegarde et cache des résultats
- Génération de visualisations annotées
- Extraction de métadonnées EXIF
- Redimensionnement intelligent

Formats supportés:
- Images: JPEG, PNG, BMP, TIFF, WebP, GIF (première frame)
- Taille max: 50MB par image
- Résolution max: 8192x8192 pixels
- Couleurs: RGB, RGBA, Grayscale
"""

import asyncio
import time
import logging
import hashlib
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ExifTags
import cv2
import torch

# Imports internes
from app.schemas.detection import DetectionResult, ImageProcessingResult
from app.config.config import get_settings
from .model_service import ModelService, PerformanceProfile

logger = logging.getLogger(__name__)

# 📋 ÉNUMÉRATIONS ET TYPES
class ImageFormat(str, Enum):
    """🖼️ Formats d'images supportés"""
    JPEG = "jpeg"
    PNG = "png"
    BMP = "bmp"
    TIFF = "tiff"
    WEBP = "webp"
    GIF = "gif"

class ImageQuality(str, Enum):
    """📊 Niveaux de qualité de traitement"""
    FAST = "fast"           # Traitement rapide, qualité réduite
    BALANCED = "balanced"   # Équilibre vitesse/qualité
    HIGH = "high"          # Haute qualité, traitement plus lent
    MAXIMUM = "maximum"    # Qualité maximale, très lent

class ResizeStrategy(str, Enum):
    """📏 Stratégies de redimensionnement"""
    CROP_CENTER = "crop_center"       # Crop au centre
    RESIZE_ASPECT = "resize_aspect"   # Redimensionner en gardant ratio
    RESIZE_FILL = "resize_fill"       # Redimensionner et remplir
    PAD_ASPECT = "pad_aspect"         # Padding pour garder ratio

@dataclass
class ImageProcessingConfig:
    """⚙️ Configuration du traitement d'images"""
    
    # 🔧 Paramètres de base
    max_image_size_mb: float = 50.0
    max_resolution: Tuple[int, int] = (8192, 8192)
    supported_formats: List[str] = field(default_factory=lambda: ["jpeg", "jpg", "png", "bmp", "tiff", "webp"])
    
    # 📊 Qualité de traitement
    quality: ImageQuality = ImageQuality.BALANCED
    resize_strategy: ResizeStrategy = ResizeStrategy.RESIZE_ASPECT
    target_size: Optional[Tuple[int, int]] = None  # Auto si None
    
    # 🎯 Paramètres de détection
    confidence_threshold: float = 0.5
    auto_model_selection: bool = True
    preferred_model: Optional[str] = None
    
    # 🖼️ Prétraitement
    auto_enhance: bool = True
    normalize_brightness: bool = True
    denoise: bool = False
    sharpen: bool = False
    
    # 💾 Sauvegarde et cache
    save_results: bool = True
    save_annotated_images: bool = True
    cache_results: bool = True
    cache_ttl_seconds: int = 3600  # 1 heure
    
    # 📊 Métadonnées
    extract_exif: bool = True
    preserve_original: bool = True
    
    def __post_init__(self):
        """Validation de la configuration"""
        if self.confidence_threshold < 0.1 or self.confidence_threshold > 0.9:
            raise ValueError("confidence_threshold doit être entre 0.1 et 0.9")

# 📸 SERVICE PRINCIPAL
class ImageService:
    """📸 Service de traitement d'images statiques"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.settings = get_settings()
        
        # Configuration par défaut
        self.config = ImageProcessingConfig()
        
        # Chemins de stockage
        self.uploads_path = Path(self.settings.UPLOADS_PATH) / "images"
        self.results_path = Path(self.settings.RESULTS_PATH) / "images"
        self.cache_path = Path(self.settings.CACHE_PATH) / "images"
        
        # Créer les répertoires
        for path in [self.uploads_path, self.results_path, self.cache_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Cache des résultats
        self._results_cache: Dict[str, Tuple[ImageProcessingResult, float]] = {}
        
        # Statistiques
        self.stats = {
            "total_processed": 0,
            "total_detections": 0,
            "processing_times": [],
            "format_counts": {},
            "error_count": 0,
            "cache_hits": 0
        }
        
        logger.info("📸 ImageService initialisé")
    
    async def initialize(self):
        """🚀 Initialise le service d'images"""
        logger.info("🚀 Initialisation ImageService...")
        
        # Vérifier les dépendances
        try:
            from PIL import Image
            logger.debug("✅ PIL/Pillow disponible")
        except ImportError:
            logger.error("❌ PIL/Pillow requis pour ImageService")
            raise
        
        try:
            import cv2
            logger.debug("✅ OpenCV disponible")
        except ImportError:
            logger.warning("⚠️ OpenCV non disponible - fonctionnalités limitées")
        
        logger.info("✅ ImageService initialisé")
    
    async def process_image(
        self,
        image_input: Union[Path, str, bytes, np.ndarray, Image.Image],
        config: Optional[ImageProcessingConfig] = None,
        processing_id: Optional[str] = None
    ) -> ImageProcessingResult:
        """
        📸 Traite une image et détecte les objets perdus
        
        Args:
            image_input: Image sous différents formats
            config: Configuration de traitement personnalisée
            processing_id: ID unique pour ce traitement
            
        Returns:
            Résultat complet du traitement
        """
        
        start_time = time.time()
        processing_id = processing_id or str(uuid.uuid4())
        config = config or self.config
        
        logger.info(f"📸 Début traitement image: {processing_id}")
        
        try:
            # 1. Chargement et validation de l'image
            image_info = await self._load_and_validate_image(image_input, processing_id)
            
            # 2. Vérification du cache
            if config.cache_results:
                cached_result = self._get_cached_result(image_info["hash"])
                if cached_result:
                    self.stats["cache_hits"] += 1
                    logger.debug(f"🎯 Cache hit pour {processing_id}")
                    return cached_result
            
            # 3. Prétraitement de l'image
            processed_image = await self._preprocess_image(
                image_info["image"], config, processing_id
            )
            
            # 4. Sélection du modèle optimal
            model_name = await self._select_optimal_model(
                processed_image, config, image_info
            )
            
            # 5. Détection d'objets
            detections = await self.model_service.detect_objects(
                image=processed_image,
                model_name=model_name,
                confidence_threshold=config.confidence_threshold,
                detection_id=processing_id
            )
            
            # 6. Post-traitement des résultats
            filtered_detections = await self._postprocess_detections(
                detections, processed_image, config
            )
            
            # 7. Génération des visualisations
            visualizations = await self._generate_visualizations(
                image_info["image"], filtered_detections, config, processing_id
            )
            
            # 8. Création du résultat final
            processing_time = time.time() - start_time
            
            result = ImageProcessingResult(
                processing_id=processing_id,
                detections=filtered_detections,
                image_info=image_info,
                processing_config=config.__dict__,
                model_used=model_name,
                processing_time_seconds=processing_time,
                visualizations=visualizations,
                metadata={
                    "total_detections": len(filtered_detections),
                    "confidence_avg": np.mean([d.confidence for d in filtered_detections]) if filtered_detections else 0.0,
                    "classes_detected": list(set(d.class_name for d in filtered_detections)),
                    "processing_quality": config.quality.value
                }
            )
            
            # 9. Sauvegarde et cache
            if config.save_results:
                await self._save_result(result, processing_id)
            
            if config.cache_results:
                self._cache_result(image_info["hash"], result)
            
            # 10. Mise à jour des statistiques
            self._update_statistics(result, image_info)
            
            logger.info(
                f"✅ Image traitée: {processing_id} - "
                f"{len(filtered_detections)} objets en {processing_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            self.stats["error_count"] += 1
            logger.error(f"❌ Erreur traitement image {processing_id}: {e}")
            
            # Retourner un résultat d'erreur
            return ImageProcessingResult(
                processing_id=processing_id,
                detections=[],
                image_info={"error": str(e)},
                processing_config=config.__dict__ if config else {},
                model_used="none",
                processing_time_seconds=time.time() - start_time,
                visualizations={},
                metadata={"error": str(e)}
            )
    
    async def process_multiple_images(
        self,
        image_inputs: List[Union[Path, str, bytes]],
        config: Optional[ImageProcessingConfig] = None,
        batch_id: Optional[str] = None
    ) -> List[ImageProcessingResult]:
        """
        📸 Traite plusieurs images en parallèle
        
        Args:
            image_inputs: Liste d'images à traiter
            config: Configuration commune
            batch_id: ID du batch
            
        Returns:
            Liste des résultats de traitement
        """
        
        batch_id = batch_id or str(uuid.uuid4())
        config = config or self.config
        
        logger.info(f"📸 Traitement batch: {batch_id} - {len(image_inputs)} images")
        
        # Traitement en parallèle avec limite de concurrence
        semaphore = asyncio.Semaphore(5)  # Max 5 images simultanées
        
        async def process_single(image_input, index):
            async with semaphore:
                processing_id = f"{batch_id}_img_{index}"
                return await self.process_image(image_input, config, processing_id)
        
        # Lancer tous les traitements
        tasks = [
            process_single(img, i) 
            for i, img in enumerate(image_inputs)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Gérer les exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Erreur image {i}: {result}")
                # Créer un résultat d'erreur
                error_result = ImageProcessingResult(
                    processing_id=f"{batch_id}_img_{i}",
                    detections=[],
                    image_info={"error": str(result)},
                    processing_config=config.__dict__,
                    model_used="none",
                    processing_time_seconds=0.0,
                    visualizations={},
                    metadata={"error": str(result)}
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        logger.info(f"✅ Batch terminé: {batch_id}")
        return processed_results
    
    async def _load_and_validate_image(
        self, 
        image_input: Union[Path, str, bytes, np.ndarray, Image.Image],
        processing_id: str
    ) -> Dict[str, Any]:
        """🔍 Charge et valide une image"""
        
        image_info = {
            "processing_id": processing_id,
            "format": None,
            "size": None,
            "mode": None,
            "has_alpha": False,
            "exif": {},
            "hash": None,
            "file_size_mb": 0.0,
            "image": None
        }
        
        try:
            # Chargement selon le type d'entrée
            if isinstance(image_input, (str, Path)):
                # Fichier
                file_path = Path(image_input)
                if not file_path.exists():
                    raise FileNotFoundError(f"Image non trouvée: {file_path}")
                
                image_info["file_size_mb"] = file_path.stat().st_size / (1024 * 1024)
                if image_info["file_size_mb"] > self.config.max_image_size_mb:
                    raise ValueError(f"Image trop grande: {image_info['file_size_mb']:.1f}MB")
                
                image = Image.open(file_path)
                
            elif isinstance(image_input, bytes):
                # Données binaires
                image_info["file_size_mb"] = len(image_input) / (1024 * 1024)
                if image_info["file_size_mb"] > self.config.max_image_size_mb:
                    raise ValueError(f"Image trop grande: {image_info['file_size_mb']:.1f}MB")
                
                from io import BytesIO
                image = Image.open(BytesIO(image_input))
                
            elif isinstance(image_input, np.ndarray):
                # Array NumPy
                if image_input.dtype != np.uint8:
                    image_input = (image_input * 255).astype(np.uint8)
                
                if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                    # RGB
                    image = Image.fromarray(image_input, 'RGB')
                elif len(image_input.shape) == 2:
                    # Grayscale
                    image = Image.fromarray(image_input, 'L')
                else:
                    raise ValueError(f"Format array non supporté: {image_input.shape}")
                    
            elif isinstance(image_input, Image.Image):
                # Image PIL directe
                image = image_input.copy()
                
            else:
                raise ValueError(f"Type d'entrée non supporté: {type(image_input)}")
            
            # Validation de l'image
            if image.size[0] > self.config.max_resolution[0] or image.size[1] > self.config.max_resolution[1]:
                raise ValueError(f"Résolution trop élevée: {image.size}")
            
            # Extraction des informations
            image_info.update({
                "format": image.format or "unknown",
                "size": image.size,
                "mode": image.mode,
                "has_alpha": image.mode in ('RGBA', 'LA', 'PA'),
                "image": image
            })
            
            # Extraction EXIF si demandé
            if self.config.extract_exif and hasattr(image, '_getexif'):
                try:
                    exif_data = image._getexif()
                    if exif_data:
                        exif_readable = {}
                        for tag_id, value in exif_data.items():
                            tag = ExifTags.TAGS.get(tag_id, tag_id)
                            exif_readable[tag] = str(value)[:100]  # Limiter la taille
                        image_info["exif"] = exif_readable
                except Exception:
                    pass  # EXIF non critique
            
            # Calcul du hash pour cache
            image_array = np.array(image)
            image_info["hash"] = hashlib.md5(image_array.tobytes()).hexdigest()
            
            return image_info
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement image {processing_id}: {e}")
            raise
    
    async def _preprocess_image(
        self,
        image: Image.Image,
        config: ImageProcessingConfig,
        processing_id: str
    ) -> Image.Image:
        """🔄 Prétraite l'image pour optimiser la détection"""
        
        processed = image.copy()
        
        try:
            # 1. Conversion en RGB si nécessaire
            if processed.mode != 'RGB':
                if processed.mode == 'RGBA':
                    # Créer un fond blanc pour transparence
                    background = Image.new('RGB', processed.size, (255, 255, 255))
                    background.paste(processed, mask=processed.split()[-1])
                    processed = background
                else:
                    processed = processed.convert('RGB')
            
            # 2. Redimensionnement si nécessaire
            if config.target_size:
                processed = await self._resize_image(processed, config.target_size, config.resize_strategy)
            elif max(processed.size) > 1280:  # Auto-resize pour gros images
                ratio = 1280 / max(processed.size)
                new_size = (int(processed.size[0] * ratio), int(processed.size[1] * ratio))
                processed = processed.resize(new_size, Image.Resampling.LANCZOS)
            
            # 3. Améliorations si activées
            if config.auto_enhance:
                processed = await self._enhance_image(processed, config)
            
            return processed
            
        except Exception as e:
            logger.error(f"❌ Erreur prétraitement {processing_id}: {e}")
            return image  # Retourner l'original en cas d'erreur
    
    async def _resize_image(
        self,
        image: Image.Image,
        target_size: Tuple[int, int],
        strategy: ResizeStrategy
    ) -> Image.Image:
        """📏 Redimensionne l'image selon la stratégie"""
        
        if strategy == ResizeStrategy.CROP_CENTER:
            # Crop au centre
            return self._crop_center(image, target_size)
            
        elif strategy == ResizeStrategy.RESIZE_ASPECT:
            # Redimensionner en gardant les proportions
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
            return image
            
        elif strategy == ResizeStrategy.RESIZE_FILL:
            # Redimensionner pour remplir exactement
            return image.resize(target_size, Image.Resampling.LANCZOS)
            
        elif strategy == ResizeStrategy.PAD_ASPECT:
            # Padding pour garder les proportions
            return self._pad_to_size(image, target_size)
        
        return image
    
    def _crop_center(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """✂️ Crop l'image au centre"""
        width, height = image.size
        target_width, target_height = target_size
        
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        return image.crop((left, top, right, bottom))
    
    def _pad_to_size(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """📦 Ajoute du padding pour atteindre la taille cible"""
        # Redimensionner en gardant les proportions
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Créer une nouvelle image avec fond noir
        padded = Image.new('RGB', target_size, (0, 0, 0))
        
        # Centrer l'image
        x = (target_size[0] - image.size[0]) // 2
        y = (target_size[1] - image.size[1]) // 2
        padded.paste(image, (x, y))
        
        return padded
    
    async def _enhance_image(
        self,
        image: Image.Image,
        config: ImageProcessingConfig
    ) -> Image.Image:
        """✨ Améliore l'image pour une meilleure détection"""
        
        enhanced = image.copy()
        
        try:
            from PIL import ImageEnhance, ImageFilter
            
            # Normalisation de la luminosité
            if config.normalize_brightness:
                # Calculer la luminosité moyenne
                grayscale = enhanced.convert('L')
                mean_brightness = np.array(grayscale).mean()
                
                # Ajuster si trop sombre ou trop clair
                if mean_brightness < 100:
                    enhancer = ImageEnhance.Brightness(enhanced)
                    enhanced = enhancer.enhance(1.2)
                elif mean_brightness > 200:
                    enhancer = ImageEnhance.Brightness(enhanced)
                    enhanced = enhancer.enhance(0.9)
            
            # Débruitage
            if config.denoise:
                enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
            
            # Accentuation
            if config.sharpen:
                enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur amélioration image: {e}")
            return image
    
    async def _select_optimal_model(
        self,
        image: Image.Image,
        config: ImageProcessingConfig,
        image_info: Dict[str, Any]
    ) -> str:
        """🎯 Sélectionne le modèle optimal pour cette image"""
        
        if not config.auto_model_selection and config.preferred_model:
            return config.preferred_model
        
        # Critères de sélection
        width, height = image.size
        total_pixels = width * height
        
        # Profil de performance selon la qualité demandée
        if config.quality == ImageQuality.FAST:
            performance_profile = PerformanceProfile.SPEED
        elif config.quality == ImageQuality.HIGH:
            performance_profile = PerformanceProfile.ACCURACY
        elif config.quality == ImageQuality.MAXIMUM:
            performance_profile = PerformanceProfile.ACCURACY
        else:  # BALANCED
            performance_profile = PerformanceProfile.BALANCED
        
        # Sélection automatique
        if total_pixels > 1280 * 1280:  # Grande image
            task_type = "accuracy" if config.quality in [ImageQuality.HIGH, ImageQuality.MAXIMUM] else "general"
        else:
            task_type = "general"
        
        model_name = await self.model_service.get_best_model_for_task(
            task_type=task_type,
            performance_profile=performance_profile,
            image_size=(width, height)
        )
        
        logger.debug(f"🎯 Modèle sélectionné: {model_name} pour image {width}x{height}")
        return model_name
    
    async def _postprocess_detections(
        self,
        detections: List[DetectionResult],
        image: Image.Image,
        config: ImageProcessingConfig
    ) -> List[DetectionResult]:
        """🔄 Post-traite les détections"""
        
        if not detections:
            return []
        
        # Filtrage par confiance
        filtered = [d for d in detections if d.confidence >= config.confidence_threshold]
        
        # Filtrage par taille minimale (éviter les micro-détections)
        min_area = 100  # pixels²
        filtered = [
            d for d in filtered 
            if (d.bbox.x2 - d.bbox.x1) * (d.bbox.y2 - d.bbox.y1) >= min_area
        ]
        
        # Tri par confiance décroissante
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        return filtered
    
    async def _generate_visualizations(
        self,
        original_image: Image.Image,
        detections: List[DetectionResult],
        config: ImageProcessingConfig,
        processing_id: str
    ) -> Dict[str, str]:
        """🎨 Génère les visualisations annotées"""
        
        visualizations = {}
        
        if not config.save_annotated_images or not detections:
            return visualizations
        
        try:
            # Image annotée avec boîtes
            annotated = await self._draw_detections(original_image.copy(), detections)
            
            # Sauvegarde
            annotated_path = self.results_path / f"{processing_id}_annotated.jpg"
            annotated.save(annotated_path, "JPEG", quality=90)
            visualizations["annotated"] = str(annotated_path)
            
            # Image avec statistiques
            if len(detections) > 0:
                stats_image = await self._draw_statistics(original_image.copy(), detections)
                stats_path = self.results_path / f"{processing_id}_stats.jpg"
                stats_image.save(stats_path, "JPEG", quality=90)
                visualizations["statistics"] = str(stats_path)
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur génération visualisations: {e}")
        
        return visualizations
    
    async def _draw_detections(
        self,
        image: Image.Image,
        detections: List[DetectionResult]
    ) -> Image.Image:
        """🎨 Dessine les détections sur l'image"""
        
        draw = ImageDraw.Draw(image)
        
        # Essayer de charger une police
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Couleurs pour les classes
        from storage.models.config_epoch_30 import CHAMPION_CLASS_COLORS
        
        for detection in detections:
            bbox = detection.bbox
            
            # Couleur selon la classe
            color = CHAMPION_CLASS_COLORS.get(detection.class_id, (255, 255, 255))
            
            # Boîte de détection
            draw.rectangle(
                [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                outline=color,
                width=3
            )
            
            # Label avec confiance
            label = f"{detection.class_name_fr} ({detection.confidence:.2f})"
            
            # Fond pour le texte
            text_bbox = draw.textbbox((bbox.x1, bbox.y1 - 25), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            
            # Texte
            draw.text(
                (bbox.x1, bbox.y1 - 25),
                label,
                fill=(0, 0, 0) if sum(color) > 400 else (255, 255, 255),
                font=font
            )
        
        return image
    
    async def _draw_statistics(
        self,
        image: Image.Image,
        detections: List[DetectionResult]
    ) -> Image.Image:
        """📊 Dessine les statistiques sur l'image"""
        
        # Créer une zone de stats en bas
        width, height = image.size
        stats_height = 100
        
        # Nouvelle image avec espace pour stats
        stats_image = Image.new('RGB', (width, height + stats_height), (50, 50, 50))
        stats_image.paste(image, (0, 0))
        
        draw = ImageDraw.Draw(stats_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # Statistiques
        total_objects = len(detections)
        avg_confidence = np.mean([d.confidence for d in detections])
        classes_found = list(set(d.class_name_fr for d in detections))
        
        # Texte des statistiques
        stats_text = [
            f"🎯 Objets détectés: {total_objects}",
            f"📊 Confiance moyenne: {avg_confidence:.2f}",
            f"🏷️ Classes: {', '.join(classes_found[:3])}{'...' if len(classes_found) > 3 else ''}"
        ]
        
        # Dessiner les statistiques
        y_offset = height + 10
        for line in stats_text:
            draw.text((10, y_offset), line, fill=(255, 255, 255), font=font)
            y_offset += 25
        
        return stats_image
    
    def _get_cached_result(self, image_hash: str) -> Optional[ImageProcessingResult]:
        """🔍 Récupère un résultat du cache"""
        
        if image_hash not in self._results_cache:
            return None
        
        result, timestamp = self._results_cache[image_hash]
        
        # Vérifier TTL
        if time.time() - timestamp > self.config.cache_ttl_seconds:
            del self._results_cache[image_hash]
            return None
        
        return result
    
    def _cache_result(self, image_hash: str, result: ImageProcessingResult):
        """💾 Met en cache un résultat"""
        
        # Nettoyage périodique du cache
        if len(self._results_cache) > 1000:
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in self._results_cache.items()
                if current_time - timestamp > self.config.cache_ttl_seconds
            ]
            for key in expired_keys:
                del self._results_cache[key]
        
        self._results_cache[image_hash] = (result, time.time())
    
    async def _save_result(self, result: ImageProcessingResult, processing_id: str):
        """💾 Sauvegarde le résultat"""
        
        try:
            result_file = self.results_path / f"{processing_id}_result.json"
            
            # Sérialiser le résultat (sans l'image)
            result_dict = {
                "processing_id": result.processing_id,
                "detections": [d.__dict__ for d in result.detections],
                "processing_config": result.processing_config,
                "model_used": result.model_used,
                "processing_time_seconds": result.processing_time_seconds,
                "visualizations": result.visualizations,
                "metadata": result.metadata,
                "timestamp": time.time()
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur sauvegarde résultat: {e}")
    
    def _update_statistics(self, result: ImageProcessingResult, image_info: Dict[str, Any]):
        """📊 Met à jour les statistiques"""
        
        self.stats["total_processed"] += 1
        self.stats["total_detections"] += len(result.detections)
        self.stats["processing_times"].append(result.processing_time_seconds)
        
        # Garder seulement les 1000 derniers temps
        if len(self.stats["processing_times"]) > 1000:
            self.stats["processing_times"] = self.stats["processing_times"][-1000:]
        
        # Compter les formats
        format_name = image_info.get("format", "unknown").lower()
        self.stats["format_counts"][format_name] = self.stats["format_counts"].get(format_name, 0) + 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """📊 Retourne les statistiques du service"""
        
        processing_times = self.stats["processing_times"]
        
        return {
            "service_type": "image_processing",
            "total_processed": self.stats["total_processed"],
            "total_detections": self.stats["total_detections"],
            "error_count": self.stats["error_count"],
            "cache_hits": self.stats["cache_hits"],
            "performance": {
                "avg_processing_time": np.mean(processing_times) if processing_times else 0.0,
                "min_processing_time": np.min(processing_times) if processing_times else 0.0,
                "max_processing_time": np.max(processing_times) if processing_times else 0.0,
                "p95_processing_time": np.percentile(processing_times, 95) if len(processing_times) > 20 else 0.0
            },
            "format_distribution": self.stats["format_counts"],
            "cache_size": len(self._results_cache),
            "uptime_hours": (time.time() - getattr(self, '_start_time', time.time())) / 3600
        }
    
    async def cleanup(self):
        """🧹 Nettoyage des ressources"""
        logger.info("🧹 Nettoyage ImageService...")
        
        # Nettoyer le cache
        self._results_cache.clear()
        
        # Nettoyer les fichiers temporaires anciens
        try:
            import glob
            temp_files = glob.glob(str(self.cache_path / "*"))
            current_time = time.time()
            
            for file_path in temp_files:
                file_age = current_time - Path(file_path).stat().st_mtime
                if file_age > 3600:  # Plus de 1 heure
                    Path(file_path).unlink()
        except Exception as e:
            logger.warning(f"⚠️ Erreur nettoyage fichiers temporaires: {e}")
        
        logger.info("✅ ImageService nettoyé")

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "ImageService",
    "ImageProcessingConfig",
    "ImageFormat",
    "ImageQuality", 
    "ResizeStrategy"
]