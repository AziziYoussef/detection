"""
📸 IMAGE SERVICE - SERVICE DE DÉTECTION D'IMAGES STATIQUES
========================================================
Service spécialisé pour le traitement d'images statiques uploadées

Fonctionnalités:
- Upload et validation d'images
- Détection d'objets perdus sur images
- Optimisation pour traitement unique
- Cache intelligent des résultats
- Support formats multiples (JPG, PNG, WebP, etc.)
- Génération de rapports détaillés
- Export des résultats annotés
"""

import asyncio
import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import uuid
from datetime import datetime

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# Imports internes
from app.core.model_manager import ModelManager
from app.core.detector import ObjectDetector, DetectionConfig
from app.core.preprocessing import ImagePreprocessor
from app.core.postprocessing import ResultPostprocessor
from app.schemas.detection import (
    DetectionResult, DetectionRequest, DetectionResponse,
    LostObjectState, DetectionMode, ModelType
)
from app.config.config import Settings

logger = logging.getLogger(__name__)

class ImageValidationError(Exception):
    """❌ Erreur de validation d'image"""
    pass

class ImageProcessor:
    """🖼️ Processeur d'images avec validation et optimisation"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.max_image_size = settings.MAX_IMAGE_SIZE
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    async def validate_and_load_image(
        self, 
        image_data: Union[bytes, str, Image.Image],
        max_size: Optional[int] = None
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """✅ Valide et charge une image"""
        
        max_size = max_size or self.max_image_size
        
        try:
            # Chargement selon le type
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, str):
                # Chemin de fichier ou base64
                if image_data.startswith('data:image'):
                    # Base64 data URL
                    image = self._load_from_base64(image_data)
                else:
                    # Chemin de fichier
                    image_path = Path(image_data)
                    if not image_path.exists():
                        raise ImageValidationError(f"Fichier non trouvé: {image_path}")
                    image = Image.open(image_path)
            elif isinstance(image_data, Image.Image):
                image = image_data
            else:
                raise ImageValidationError(f"Type d'image non supporté: {type(image_data)}")
            
            # Validation du format
            if image.format and image.format.lower() not in ['jpeg', 'png', 'bmp', 'tiff', 'webp']:
                logger.warning(f"⚠️ Format d'image inhabituel: {image.format}")
            
            # Conversion en RGB si nécessaire
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
            
            # Validation de la taille
            width, height = image.size
            if width * height > max_size * max_size:
                # Redimensionnement si trop grand
                image = self._resize_if_needed(image, max_size)
                logger.info(f"📏 Image redimensionnée: {width}x{height} → {image.size}")
            
            # Informations sur l'image
            image_info = {
                "original_size": (width, height),
                "final_size": image.size,
                "mode": image.mode,
                "format": image.format,
                "has_transparency": image.mode in ['RGBA', 'LA'] or 'transparency' in image.info
            }
            
            return image, image_info
            
        except Exception as e:
            logger.error(f"❌ Erreur validation image: {e}")
            raise ImageValidationError(f"Image invalide: {e}")
    
    def _load_from_base64(self, data_url: str) -> Image.Image:
        """📄 Charge image depuis data URL base64"""
        import base64
        import io
        
        try:
            # Extraction des données base64
            header, data = data_url.split(',', 1)
            image_data = base64.b64decode(data)
            
            return Image.open(io.BytesIO(image_data))
            
        except Exception as e:
            raise ImageValidationError(f"Data URL base64 invalide: {e}")
    
    def _resize_if_needed(self, image: Image.Image, max_size: int) -> Image.Image:
        """📏 Redimensionne l'image si nécessaire"""
        
        width, height = image.size
        
        if max(width, height) <= max_size:
            return image
        
        # Calcul des nouvelles dimensions en préservant le ratio
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

class ResultsExporter:
    """📤 Exportateur de résultats avec annotations"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.output_dir = settings.TEMP_DIR / "results"
        self.output_dir.mkdir(exist_ok=True)
        
        # Configuration des annotations
        self.colors = {
            'person': (0, 255, 0),      # Vert
            'backpack': (255, 0, 0),    # Rouge
            'suitcase': (255, 0, 0),    # Rouge
            'handbag': (255, 0, 0),     # Rouge
            'cell phone': (0, 0, 255),  # Bleu
            'laptop': (0, 0, 255),      # Bleu
            'default': (128, 128, 128)  # Gris
        }
        
        # Status colors pour objets perdus
        self.status_colors = {
            'normal': (0, 255, 0),      # Vert
            'suspect': (255, 165, 0),   # Orange
            'lost': (255, 0, 0),        # Rouge
            'critical': (139, 0, 0)     # Rouge foncé
        }
    
    async def export_annotated_image(
        self,
        original_image: Image.Image,
        detections: List[DetectionResult],
        lost_objects: List[LostObjectState],
        request_id: str = None
    ) -> str:
        """📸 Exporte une image annotée avec les détections"""
        
        # Copie de l'image pour annotation
        annotated_image = original_image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Chargement de la police
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = font
        
        # Annoter les détections normales
        for detection in detections:
            self._draw_detection(draw, detection, font, font_small)
        
        # Annoter les objets perdus (par-dessus)
        for lost_object in lost_objects:
            self._draw_lost_object(draw, lost_object, font, font_small)
        
        # Sauvegarde
        filename = f"annotated_{request_id or uuid.uuid4().hex}.jpg"
        output_path = self.output_dir / filename
        annotated_image.save(output_path, "JPEG", quality=90)
        
        logger.info(f"📸 Image annotée sauvegardée: {output_path}")
        return str(output_path)
    
    def _draw_detection(
        self, 
        draw: ImageDraw.Draw, 
        detection: DetectionResult,
        font: ImageFont.ImageFont,
        font_small: ImageFont.ImageFont
    ):
        """🖊️ Dessine une détection sur l'image"""
        
        bbox = detection.bbox
        color = self.colors.get(detection.class_name, self.colors['default'])
        
        # Rectangle
        draw.rectangle(
            [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
            outline=color,
            width=2
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
            fill="white",
            font=font
        )
    
    def _draw_lost_object(
        self,
        draw: ImageDraw.Draw,
        lost_object: LostObjectState,
        font: ImageFont.ImageFont,
        font_small: ImageFont.ImageFont
    ):
        """🚨 Dessine un objet perdu avec statut"""
        
        detection = lost_object.detection_result
        bbox = detection.bbox
        status_color = self.status_colors.get(lost_object.status.value, self.status_colors['lost'])
        
        # Rectangle épais pour objet perdu
        draw.rectangle(
            [bbox.x1 - 2, bbox.y1 - 2, bbox.x2 + 2, bbox.y2 + 2],
            outline=status_color,
            width=4
        )
        
        # Label spécial objet perdu
        status_text = {
            'suspect': '⚠️ SUSPECT',
            'lost': '🚨 PERDU',
            'critical': '🔥 CRITIQUE'
        }.get(lost_object.status.value, '❓ INCONNU')
        
        duration_text = f"{lost_object.stationary_duration}s"
        
        # Fond pour le statut
        status_bbox = draw.textbbox((bbox.x1, bbox.y2 + 5), status_text, font=font)
        draw.rectangle(status_bbox, fill=status_color)
        
        # Texte statut
        draw.text(
            (bbox.x1, bbox.y2 + 5),
            status_text,
            fill="white",
            font=font
        )
        
        # Durée
        draw.text(
            (bbox.x2 - 50, bbox.y2 + 5),
            duration_text,
            fill=status_color,
            font=font_small
        )

class ImageService:
    """📸 Service principal pour détection d'images statiques"""
    
    def __init__(self, model_manager: ModelManager, settings: Settings):
        self.model_manager = model_manager
        self.settings = settings
        
        # Composants de traitement
        self.image_processor = ImageProcessor(settings)
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = ResultPostprocessor(settings)
        self.exporter = ResultsExporter(settings)
        
        # Cache des résultats
        self._results_cache: Dict[str, Tuple[DetectionResponse, float]] = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Statistiques
        self.total_processed = 0
        self.processing_times: List[float] = []
        self.cache_hits = 0
        
        logger.info("📸 ImageService initialisé")
    
    async def detect_objects_in_image(
        self,
        image_data: Union[bytes, str, Image.Image],
        request: DetectionRequest,
        request_id: Optional[str] = None
    ) -> DetectionResponse:
        """
        🎯 Détecte les objets dans une image statique
        
        Args:
            image_data: Données de l'image (bytes, chemin, PIL Image)
            request: Paramètres de détection
            request_id: ID unique de la requête
            
        Returns:
            Résultats de détection avec objets perdus
        """
        
        start_time = time.time()
        request_id = request_id or str(uuid.uuid4())
        
        try:
            # 1. Validation et chargement de l'image
            image, image_info = await self.image_processor.validate_and_load_image(image_data)
            
            # 2. Vérification du cache
            if self._should_use_cache(image, request):
                cache_key = self._generate_cache_key(image, request)
                cached_result = self._get_cached_result(cache_key)
                
                if cached_result:
                    self.cache_hits += 1
                    logger.debug(f"📸 Cache hit pour requête {request_id}")
                    return cached_result
            
            # 3. Configuration du détecteur
            detector_config = DetectionConfig(
                confidence_threshold=request.confidence_threshold,
                nms_threshold=request.nms_threshold,
                max_detections=request.max_detections,
                detection_mode=request.detection_mode,
                model_type=request.model_name,
                half_precision=self.settings.HALF_PRECISION
            )
            
            # 4. Création du détecteur
            detector = ObjectDetector(self.model_manager, detector_config)
            
            # 5. Détection
            detections = await detector.detect_objects(
                image=image,
                model_name=request.model_name.value,
                confidence_threshold=request.confidence_threshold,
                detection_id=request_id
            )
            
            # 6. Analyse objets perdus
            lost_objects = []
            if request.enable_lost_detection:
                _, lost_objects = self.postprocessor.process_detections(
                    detections, 
                    enable_lost_detection=True
                )
            
            # 7. Export image annotée si demandé
            annotated_image_path = None
            if request.return_cropped_objects:
                annotated_image_path = await self.exporter.export_annotated_image(
                    image, detections, lost_objects, request_id
                )
            
            # 8. Construction de la réponse
            processing_time = (time.time() - start_time) * 1000
            
            response = DetectionResponse(
                success=True,
                detections=detections,
                processing_time_ms=processing_time,
                model_used=request.model_name.value,
                image_info=image_info,
                total_objects=len(detections),
                lost_objects_count=len(lost_objects),
                request_id=request_id
            )
            
            # 9. Mise à jour des statistiques
            self.total_processed += 1
            self.processing_times.append(processing_time / 1000)
            
            # 10. Mise en cache si approprié
            if self._should_cache_result(request):
                cache_key = self._generate_cache_key(image, request)
                self._cache_result(cache_key, response)
            
            logger.info(
                f"📸 Image traitée: {len(detections)} objets, "
                f"{len(lost_objects)} perdus en {processing_time:.1f}ms"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement image {request_id}: {e}")
            
            # Réponse d'erreur
            return DetectionResponse(
                success=False,
                detections=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used=request.model_name.value,
                request_id=request_id
            )
    
    async def detect_objects_batch(
        self,
        images_data: List[Union[bytes, str, Image.Image]],
        request: DetectionRequest,
        batch_id: Optional[str] = None
    ) -> List[DetectionResponse]:
        """📦 Détection sur un lot d'images"""
        
        batch_id = batch_id or str(uuid.uuid4())
        logger.info(f"📦 Traitement batch {batch_id}: {len(images_data)} images")
        
        # Traitement parallèle
        tasks = []
        for i, image_data in enumerate(images_data):
            request_id = f"{batch_id}_{i}"
            task = self.detect_objects_in_image(image_data, request, request_id)
            tasks.append(task)
        
        # Attendre tous les résultats
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrer les exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Erreur image {i} du batch {batch_id}: {result}")
                # Créer une réponse d'erreur
                error_response = DetectionResponse(
                    success=False,
                    detections=[],
                    processing_time_ms=0,
                    model_used=request.model_name.value,
                    request_id=f"{batch_id}_{i}"
                )
                valid_results.append(error_response)
            else:
                valid_results.append(result)
        
        logger.info(f"📦 Batch {batch_id} terminé: {len(valid_results)} résultats")
        return valid_results
    
    def _should_use_cache(self, image: Image.Image, request: DetectionRequest) -> bool:
        """🤔 Détermine si utiliser le cache"""
        
        # Pas de cache pour les requêtes avec tracking
        if request.enable_tracking or request.enable_lost_detection:
            return False
        
        # Cache seulement pour images de taille raisonnable
        if image.size[0] * image.size[1] > 1920 * 1080:
            return False
        
        return True
    
    def _generate_cache_key(self, image: Image.Image, request: DetectionRequest) -> str:
        """🔑 Génère une clé de cache"""
        
        # Hash basé sur l'image et les paramètres
        image_bytes = image.tobytes()
        image_hash = hashlib.md5(image_bytes).hexdigest()[:16]
        
        params_str = f"{request.model_name.value}_{request.confidence_threshold}_{request.detection_mode.value}"
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        
        return f"{image_hash}_{params_hash}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[DetectionResponse]:
        """📥 Récupère un résultat du cache"""
        
        if cache_key not in self._results_cache:
            return None
        
        result, timestamp = self._results_cache[cache_key]
        
        # Vérifier TTL
        if time.time() - timestamp > self._cache_ttl:
            del self._results_cache[cache_key]
            return None
        
        return result
    
    def _should_cache_result(self, request: DetectionRequest) -> bool:
        """🤔 Détermine si mettre en cache le résultat"""
        
        # Pas de cache pour les requêtes avec tracking
        if request.enable_tracking or request.enable_lost_detection:
            return False
        
        return True
    
    def _cache_result(self, cache_key: str, result: DetectionResponse):
        """💾 Met en cache un résultat"""
        
        # Nettoyage périodique du cache
        if len(self._results_cache) > 100:
            self._cleanup_cache()
        
        self._results_cache[cache_key] = (result, time.time())
    
    def _cleanup_cache(self):
        """🧹 Nettoie le cache expiré"""
        
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._results_cache.items()
            if current_time - timestamp > self._cache_ttl
        ]
        
        for key in expired_keys:
            del self._results_cache[key]
        
        logger.debug(f"🧹 Cache nettoyé: {len(expired_keys)} entrées supprimées")
    
    async def get_supported_formats(self) -> List[str]:
        """📋 Retourne les formats d'images supportés"""
        return list(self.image_processor.supported_formats)
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """📊 Statistiques du service"""
        
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            "total_processed": self.total_processed,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.total_processed),
            "average_processing_time_ms": avg_processing_time * 1000,
            "cache_size": len(self._results_cache),
            "supported_formats": list(self.image_processor.supported_formats)
        }
    
    async def cleanup(self):
        """🧹 Nettoyage des ressources"""
        self._results_cache.clear()
        logger.info("🧹 ImageService nettoyé")

# === EXPORTS ===
__all__ = [
    "ImageService",
    "ImageProcessor",
    "ResultsExporter",
    "ImageValidationError"
]