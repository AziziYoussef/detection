"""
🎯 DETECTOR - CLASSE PRINCIPALE DE DÉTECTION D'OBJETS PERDUS
===========================================================
Classe centrale pour la détection d'objets perdus utilisant vos modèles PyTorch

Fonctionnalités:
- Détection multi-classes (28 objets perdus)
- Support modèles multiples (Epoch 30, Extended, Fast)
- Optimisation GPU/CPU automatique
- Modes de détection (FAST, BALANCED, QUALITY)
- Pipeline de traitement optimisé
- Cache intelligent des résultats

Modèles supportés:
- Epoch 30: Champion (F1=49.86%, Précision=60.73%)
- Extended: 28 classes d'objets perdus
- Fast: Optimisé pour temps réel (streaming)

Architecture:
- Prétraitement adaptatif selon la source
- Inférence optimisée PyTorch
- Post-traitement avec filtrage intelligent
- Gestion mémoire GPU optimisée
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from enum import Enum
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2

# Imports internes
from app.schemas.detection import DetectionResult, BoundingBox
from app.config.config import get_settings
from .preprocessing import ImagePreprocessor, PreprocessingPipeline
from .postprocessing import ResultPostprocessor, DetectionFilter

logger = logging.getLogger(__name__)

# 📋 ÉNUMÉRATIONS ET TYPES
class DetectionMode(str, Enum):
    """🎛️ Modes de détection"""
    ULTRA_FAST = "ultra_fast"    # Optimisé streaming (latence minimale)
    FAST = "fast"                # Rapide (équilibre vitesse/qualité)
    BALANCED = "balanced"        # Équilibré (usage général)
    QUALITY = "quality"          # Qualité maximale (précision)
    BATCH = "batch"              # Optimisé traitement batch

class ModelType(str, Enum):
    """🤖 Types de modèles"""
    EPOCH_30 = "epoch_30"        # Champion model
    EXTENDED = "extended"        # Extended classes
    FAST = "fast"                # Streaming optimized
    MOBILE = "mobile"            # Mobile/Edge optimized

@dataclass
class DetectionConfig:
    """⚙️ Configuration de détection"""
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100
    min_box_area: int = 25
    detection_mode: DetectionMode = DetectionMode.BALANCED
    model_type: ModelType = ModelType.EPOCH_30
    device: str = "auto"  # auto, cpu, cuda
    half_precision: bool = False  # FP16 pour optimisation GPU
    
    def __post_init__(self):
        """Validation de la configuration"""
        if not 0.1 <= self.confidence_threshold <= 0.9:
            raise ValueError("confidence_threshold doit être entre 0.1 et 0.9")
        if not 0.1 <= self.nms_threshold <= 0.9:
            raise ValueError("nms_threshold doit être entre 0.1 et 0.9")

# 🧠 MOTEUR DE DÉTECTION
class DetectionEngine:
    """🧠 Moteur principal de détection"""
    
    def __init__(self, model: torch.nn.Module, device: torch.device, config: DetectionConfig):
        self.model = model
        self.device = device
        self.config = config
        self.class_names = get_settings().DETECTION_CLASSES
        self.class_names_fr = get_settings().CLASSES_FR_NAMES
        
        # Optimisations
        self.model.eval()
        if config.half_precision and device.type == "cuda":
            self.model = self.model.half()
        
        # Statistiques
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.last_inference_time = 0.0
        
        logger.info(f"🧠 DetectionEngine initialisé sur {device}")
    
    @torch.no_grad()
    def detect(self, tensor_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """🎯 Effectue la détection sur un tensor preprocessé"""
        
        start_time = time.time()
        
        try:
            # Déplacer sur device
            tensor_input = tensor_input.to(self.device)
            
            # Conversion FP16 si activée
            if self.config.half_precision and self.device.type == "cuda":
                tensor_input = tensor_input.half()
            
            # Inférence
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                predictions = self.model(tensor_input)
            
            # Extraction des résultats selon l'architecture du modèle
            if isinstance(predictions, (list, tuple)):
                # Format YOLOv5/YOLOv8 standard
                boxes, scores, classes = self._extract_yolo_predictions(predictions[0])
            elif isinstance(predictions, dict):
                # Format Detectron2 ou custom
                boxes = predictions.get("boxes", torch.empty(0, 4))
                scores = predictions.get("scores", torch.empty(0))
                classes = predictions.get("labels", torch.empty(0))
            else:
                # Format tensor direct
                boxes, scores, classes = self._extract_tensor_predictions(predictions)
            
            # Mise à jour statistiques
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            self.last_inference_time = inference_time
            
            return boxes, scores, classes
            
        except Exception as e:
            logger.error(f"❌ Erreur inférence: {e}")
            # Retourner tensors vides en cas d'erreur
            empty_boxes = torch.empty(0, 4, device=self.device)
            empty_scores = torch.empty(0, device=self.device)
            empty_classes = torch.empty(0, device=self.device, dtype=torch.long)
            return empty_boxes, empty_scores, empty_classes
    
    def _extract_yolo_predictions(self, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """🎯 Extrait les prédictions format YOLO"""
        
        # Format YOLO: [batch, anchors, 5+num_classes]
        # 5 = x, y, w, h, objectness
        if predictions.dim() == 3:
            predictions = predictions[0]  # Prendre le premier batch
        
        # Séparation des composants
        boxes = predictions[:, :4]  # x, y, w, h
        objectness = predictions[:, 4]  # confiance objectness
        class_probs = predictions[:, 5:]  # probabilités classes
        
        # Calcul des scores finaux
        scores = objectness.unsqueeze(1) * class_probs
        max_scores, classes = torch.max(scores, dim=1)
        
        # Filtrage par seuil
        mask = max_scores > self.config.confidence_threshold
        boxes = boxes[mask]
        scores = max_scores[mask]
        classes = classes[mask]
        
        return boxes, scores, classes
    
    def _extract_tensor_predictions(self, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """🎯 Extrait les prédictions format tensor générique"""
        
        if predictions.size(-1) >= 6:  # x, y, w, h, conf, class
            boxes = predictions[:, :4]
            scores = predictions[:, 4]
            classes = predictions[:, 5].long()
            
            # Filtrage par seuil
            mask = scores > self.config.confidence_threshold
            return boxes[mask], scores[mask], classes[mask]
        
        # Format non reconnu, retourner vide
        device = predictions.device
        return (torch.empty(0, 4, device=device),
                torch.empty(0, device=device),
                torch.empty(0, device=device, dtype=torch.long))
    
    def get_average_inference_time(self) -> float:
        """⏱️ Retourne le temps d'inférence moyen"""
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_time / self.inference_count
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """📊 Retourne les statistiques de performance"""
        return {
            "inference_count": self.inference_count,
            "total_inference_time": self.total_inference_time,
            "average_inference_time": self.get_average_inference_time(),
            "last_inference_time": self.last_inference_time,
            "fps_estimate": 1.0 / self.last_inference_time if self.last_inference_time > 0 else 0.0,
            "device": str(self.device),
            "half_precision": self.config.half_precision
        }

# 🎯 DÉTECTEUR PRINCIPAL
class ObjectDetector:
    """🎯 Détecteur principal d'objets perdus"""
    
    def __init__(self, model_manager, config: DetectionConfig = None):
        """
        Initialise le détecteur
        
        Args:
            model_manager: Instance de ModelManager
            config: Configuration de détection
        """
        self.model_manager = model_manager
        self.config = config or DetectionConfig()
        self.device = self._determine_device()
        
        # Composants de traitement
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = ResultPostprocessor()
        self.detection_filter = DetectionFilter()
        
        # Moteurs de détection par modèle (cache)
        self._detection_engines: Dict[str, DetectionEngine] = {}
        
        # Statistiques globales
        self.total_detections = 0
        self.processing_times = []
        
        logger.info(f"🎯 ObjectDetector initialisé sur {self.device}")
    
    def _determine_device(self) -> torch.device:
        """🔧 Détermine le device optimal"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")  # Apple Silicon
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)
    
    async def detect_objects(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        model_name: str = None,
        confidence_threshold: float = None,
        detection_id: str = None
    ) -> List[DetectionResult]:
        """
        🎯 Détecte les objets perdus dans une image
        
        Args:
            image: Image à analyser (PIL, numpy, tensor)
            model_name: Nom du modèle à utiliser
            confidence_threshold: Seuil de confiance personnalisé
            detection_id: ID unique pour cette détection
            
        Returns:
            Liste des objets détectés
        """
        
        start_time = time.time()
        model_name = model_name or self.config.model_type.value
        confidence_threshold = confidence_threshold or self.config.confidence_threshold
        
        logger.debug(f"🎯 Début détection - Modèle: {model_name}, Seuil: {confidence_threshold}")
        
        try:
            # 1. Récupération du moteur de détection
            engine = await self._get_detection_engine(model_name, confidence_threshold)
            
            # 2. Prétraitement de l'image
            processed_input, original_size = await self._preprocess_image(image, model_name)
            
            # 3. Détection
            boxes, scores, classes = engine.detect(processed_input)
            
            # 4. Post-traitement
            detections = await self._postprocess_results(
                boxes, scores, classes, original_size, model_name, detection_id
            )
            
            # 5. Filtrage final
            filtered_detections = self.detection_filter.filter_detections(
                detections, confidence_threshold, self.config.max_detections
            )
            
            # 6. Statistiques
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.total_detections += len(filtered_detections)
            
            logger.debug(
                f"✅ Détection terminée - {len(filtered_detections)} objets "
                f"en {processing_time*1000:.1f}ms"
            )
            
            return filtered_detections
            
        except Exception as e:
            logger.error(f"❌ Erreur détection: {e}", exc_info=True)
            return []
    
    async def detect_objects_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        model_name: str = None,
        confidence_threshold: float = None,
        batch_size: int = 8
    ) -> List[List[DetectionResult]]:
        """
        🎯 Détection batch sur plusieurs images
        
        Args:
            images: Liste d'images à analyser
            model_name: Nom du modèle
            confidence_threshold: Seuil de confiance
            batch_size: Taille des batches
            
        Returns:
            Liste de listes de détections (une par image)
        """
        
        all_results = []
        
        # Traitement par batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Traitement parallèle du batch
            batch_tasks = [
                self.detect_objects(
                    image=img,
                    model_name=model_name,
                    confidence_threshold=confidence_threshold,
                    detection_id=f"batch_{i}_{j}"
                )
                for j, img in enumerate(batch_images)
            ]
            
            batch_results = await asyncio.gather(*batch_tasks)
            all_results.extend(batch_results)
        
        return all_results
    
    async def _get_detection_engine(self, model_name: str, confidence_threshold: float) -> DetectionEngine:
        """🧠 Récupère ou crée un moteur de détection"""
        
        cache_key = f"{model_name}_{confidence_threshold}"
        
        if cache_key not in self._detection_engines:
            # Charger le modèle
            model = await self.model_manager.get_model(model_name)
            
            # Créer la configuration pour ce moteur
            engine_config = DetectionConfig(
                confidence_threshold=confidence_threshold,
                nms_threshold=self.config.nms_threshold,
                max_detections=self.config.max_detections,
                detection_mode=self.config.detection_mode,
                device=str(self.device),
                half_precision=self.config.half_precision
            )
            
            # Créer le moteur
            engine = DetectionEngine(model, self.device, engine_config)
            self._detection_engines[cache_key] = engine
            
            logger.info(f"🧠 Nouveau moteur créé: {cache_key}")
        
        return self._detection_engines[cache_key]
    
    async def _preprocess_image(
        self, 
        image: Union[Image.Image, np.ndarray, torch.Tensor], 
        model_name: str
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """🔄 Prétraite l'image pour le modèle"""
        
        # Récupération de la configuration du modèle
        model_config = await self.model_manager.get_model_config(model_name)
        
        # Prétraitement selon le mode de détection
        if self.config.detection_mode == DetectionMode.ULTRA_FAST:
            processed = self.preprocessor.preprocess_fast(image, model_config)
        elif self.config.detection_mode == DetectionMode.QUALITY:
            processed = self.preprocessor.preprocess_quality(image, model_config)
        else:
            processed = self.preprocessor.preprocess_balanced(image, model_config)
        
        return processed
    
    async def _postprocess_results(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor, 
        classes: torch.Tensor,
        original_size: Tuple[int, int],
        model_name: str,
        detection_id: str = None
    ) -> List[DetectionResult]:
        """🔄 Post-traite les résultats de détection"""
        
        if len(boxes) == 0:
            return []
        
        # Application NMS
        keep_indices = self._apply_nms(boxes, scores, self.config.nms_threshold)
        
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        classes = classes[keep_indices]
        
        # Conversion en DetectionResult
        detections = []
        
        for i in range(len(boxes)):
            box = boxes[i]
            score = scores[i].item()
            class_id = classes[i].item()
            
            # Conversion des coordonnées au format original
            bbox = self._convert_bbox_to_original(box, original_size)
            
            # Création du résultat
            detection = DetectionResult(
                bbox=bbox,
                confidence=score,
                class_id=int(class_id),
                class_name=self._get_class_name(class_id),
                class_name_fr=self._get_class_name_fr(class_id),
                model_name=model_name,
                detection_id=detection_id or f"det_{int(time.time() * 1000)}_{i}"
            )
            
            detections.append(detection)
        
        return detections
    
    def _apply_nms(self, boxes: torch.Tensor, scores: torch.Tensor, nms_threshold: float) -> torch.Tensor:
        """🎯 Applique Non-Maximum Suppression"""
        
        try:
            from torchvision.ops import nms
            
            # Conversion format xyxy si nécessaire
            if boxes.size(1) == 4:
                # Supposer format xywh → xyxy
                boxes_xyxy = boxes.clone()
                boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]  # x + w
                boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]  # y + h
            else:
                boxes_xyxy = boxes
            
            keep = nms(boxes_xyxy, scores, nms_threshold)
            return keep
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur NMS, retour sans filtrage: {e}")
            return torch.arange(len(boxes))
    
    def _convert_bbox_to_original(self, box: torch.Tensor, original_size: Tuple[int, int]) -> BoundingBox:
        """📐 Convertit bbox vers taille originale"""
        
        orig_width, orig_height = original_size
        
        # Supposer format normalisé [0,1] ou format pixel
        if box.max() <= 1.0:
            # Format normalisé
            x1 = int(box[0].item() * orig_width)
            y1 = int(box[1].item() * orig_height)
            x2 = int(box[2].item() * orig_width)
            y2 = int(box[3].item() * orig_height)
        else:
            # Format pixel direct
            x1, y1, x2, y2 = box.int().tolist()
        
        return BoundingBox(
            x1=max(0, x1),
            y1=max(0, y1),
            x2=min(orig_width, x2),
            y2=min(orig_height, y2)
        )
    
    def _get_class_name(self, class_id: int) -> str:
        """🏷️ Récupère le nom de classe anglais"""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"unknown_{class_id}"
    
    def _get_class_name_fr(self, class_id: int) -> str:
        """🏷️ Récupère le nom de classe français"""
        if 0 <= class_id < len(self.class_names_fr):
            return self.class_names_fr[class_id]
        return f"inconnu_{class_id}"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """📊 Récupère les métriques de performance"""
        
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        
        engine_stats = {}
        for cache_key, engine in self._detection_engines.items():
            engine_stats[cache_key] = engine.get_performance_stats()
        
        return {
            "total_detections": self.total_detections,
            "total_processing_calls": len(self.processing_times),
            "average_processing_time": avg_processing_time,
            "processing_times_history": self.processing_times[-100:],  # Dernières 100
            "device": str(self.device),
            "config": {
                "confidence_threshold": self.config.confidence_threshold,
                "nms_threshold": self.config.nms_threshold,
                "detection_mode": self.config.detection_mode.value,
                "half_precision": self.config.half_precision
            },
            "engines": engine_stats
        }
    
    def update_config(self, **kwargs):
        """⚙️ Met à jour la configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"⚙️ Config mise à jour: {key}={value}")
        
        # Invalider le cache des moteurs si nécessaire
        if any(key in ["confidence_threshold", "nms_threshold", "detection_mode"] for key in kwargs):
            self._detection_engines.clear()
            logger.info("🔄 Cache des moteurs invalidé")
    
    async def cleanup(self):
        """🧹 Nettoyage des ressources"""
        self._detection_engines.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("🧹 ObjectDetector nettoyé")

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "ObjectDetector",
    "DetectionEngine", 
    "DetectionConfig",
    "DetectionMode",
    "ModelType"
]