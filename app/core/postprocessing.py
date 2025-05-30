"""
🔄 POSTPROCESSING - POST-TRAITEMENT DES RÉSULTATS DE DÉTECTION
============================================================
Module de post-traitement intelligent pour optimiser les résultats de détection

Fonctionnalités:
- Filtrage avancé des détections (confiance, taille, pertinence)
- Agrégation intelligente de résultats multiples
- Analyse statistique des détections
- Génération de rapports et visualisations
- Optimisation selon le contexte (streaming vs batch)
- Cache des résultats pour réutilisation

Composants:
- ResultPostprocessor: Post-traitement principal
- DetectionFilter: Filtrage intelligent
- ResultAggregator: Agrégation multi-sources
- StatisticsGenerator: Analyses statistiques

Architecture:
- Pipeline configurable de post-traitement
- Algorithmes adaptatifs selon le contexte
- Optimisations pour différents cas d'usage
- Intégration avec visualisation et rapports
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import statistics
import json

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Imports internes
from app.schemas.detection import DetectionResult, BoundingBox
from app.config.config import get_settings

logger = logging.getLogger(__name__)

# 📋 ÉNUMÉRATIONS ET TYPES
class FilteringMode(str, Enum):
    """🔍 Modes de filtrage"""
    STRICT = "strict"           # Filtrage strict (haute précision)
    BALANCED = "balanced"       # Équilibré (usage général)
    PERMISSIVE = "permissive"   # Permissif (capture maximale)
    ADAPTIVE = "adaptive"       # Adaptatif selon contexte

class AggregationMethod(str, Enum):
    """📊 Méthodes d'agrégation"""
    SIMPLE = "simple"           # Agrégation simple
    WEIGHTED = "weighted"       # Pondérée par confiance
    CONSENSUS = "consensus"     # Consensus multi-modèles
    TEMPORAL = "temporal"       # Agrégation temporelle (vidéo)

class ContextType(str, Enum):
    """🎭 Types de contexte"""
    STREAMING = "streaming"     # Streaming temps réel
    BATCH = "batch"            # Traitement batch
    SINGLE_IMAGE = "single"     # Image unique
    VIDEO = "video"            # Vidéo complète

@dataclass
class PostprocessingConfig:
    """⚙️ Configuration post-traitement"""
    # Filtrage
    min_confidence: float = 0.5
    max_detections: int = 100
    min_box_area: int = 25
    max_box_area: int = 100000
    filtering_mode: FilteringMode = FilteringMode.BALANCED
    
    # Agrégation
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED
    temporal_window: float = 1.0  # secondes pour agrégation temporelle
    
    # Contexte
    context_type: ContextType = ContextType.SINGLE_IMAGE
    
    # NMS avancé
    advanced_nms: bool = True
    nms_threshold: float = 0.4
    soft_nms: bool = False
    
    # Analyse
    generate_statistics: bool = True
    track_performance: bool = True

@dataclass
class DetectionStatistics:
    """📊 Statistiques de détection"""
    total_detections: int = 0
    detections_by_class: Dict[str, int] = field(default_factory=dict)
    confidence_distribution: List[float] = field(default_factory=list)
    box_size_distribution: List[float] = field(default_factory=list)
    
    # Performance
    processing_time_ms: float = 0.0
    average_confidence: float = 0.0
    max_confidence: float = 0.0
    min_confidence: float = 1.0
    
    # Analyse qualité
    quality_score: float = 0.0
    reliability_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire"""
        return {
            "total_detections": self.total_detections,
            "detections_by_class": self.detections_by_class,
            "average_confidence": self.average_confidence,
            "max_confidence": self.max_confidence,
            "min_confidence": self.min_confidence,
            "processing_time_ms": self.processing_time_ms,
            "quality_score": self.quality_score,
            "reliability_score": self.reliability_score,
            "confidence_stats": {
                "mean": statistics.mean(self.confidence_distribution) if self.confidence_distribution else 0.0,
                "median": statistics.median(self.confidence_distribution) if self.confidence_distribution else 0.0,
                "std": statistics.stdev(self.confidence_distribution) if len(self.confidence_distribution) > 1 else 0.0
            }
        }

# 🔍 FILTRE DE DÉTECTIONS
class DetectionFilter:
    """🔍 Filtre intelligent pour les détections"""
    
    def __init__(self):
        self.settings = get_settings()
        self.filtered_count = 0
        self.total_processed = 0
        
        logger.info("🔍 DetectionFilter initialisé")
    
    def filter_detections(
        self,
        detections: List[DetectionResult],
        confidence_threshold: float = 0.5,
        max_detections: int = 100,
        config: PostprocessingConfig = None
    ) -> List[DetectionResult]:
        """🔍 Filtre les détections selon les critères"""
        
        config = config or PostprocessingConfig(
            min_confidence=confidence_threshold,
            max_detections=max_detections
        )
        
        self.total_processed += len(detections)
        
        # Filtrage par confiance
        filtered = self._filter_by_confidence(detections, config.min_confidence)
        
        # Filtrage par taille de boîte
        filtered = self._filter_by_box_size(filtered, config.min_box_area, config.max_box_area)
        
        # Filtrage par classe (si liste noire/blanche définie)
        filtered = self._filter_by_class(filtered)
        
        # NMS avancé si activé
        if config.advanced_nms:
            filtered = self._apply_advanced_nms(filtered, config.nms_threshold, config.soft_nms)
        
        # Limitation nombre max
        filtered = self._limit_max_detections(filtered, config.max_detections)
        
        # Tri par confiance décroissante
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        self.filtered_count += len(detections) - len(filtered)
        
        logger.debug(f"🔍 Filtrage: {len(detections)} → {len(filtered)} détections")
        
        return filtered
    
    def _filter_by_confidence(self, detections: List[DetectionResult], min_confidence: float) -> List[DetectionResult]:
        """🎯 Filtrage par seuil de confiance"""
        return [det for det in detections if det.confidence >= min_confidence]
    
    def _filter_by_box_size(
        self, 
        detections: List[DetectionResult], 
        min_area: int, 
        max_area: int
    ) -> List[DetectionResult]:
        """📐 Filtrage par taille de boîte"""
        
        filtered = []
        for det in detections:
            area = det.bbox.get_area()
            if min_area <= area <= max_area:
                filtered.append(det)
        
        return filtered
    
    def _filter_by_class(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """🏷️ Filtrage par classe"""
        
        # Classes interdites (si configurées)
        blacklisted_classes = getattr(self.settings, 'BLACKLISTED_CLASSES', [])
        
        if blacklisted_classes:
            return [det for det in detections if det.class_name not in blacklisted_classes]
        
        return detections
    
    def _apply_advanced_nms(
        self, 
        detections: List[DetectionResult], 
        nms_threshold: float,
        soft_nms: bool = False
    ) -> List[DetectionResult]:
        """🎯 NMS avancé avec support Soft-NMS"""
        
        if len(detections) <= 1:
            return detections
        
        # Grouper par classe
        by_class = defaultdict(list)
        for det in detections:
            by_class[det.class_name].append(det)
        
        final_detections = []
        
        # Appliquer NMS par classe
        for class_name, class_detections in by_class.items():
            if soft_nms:
                filtered_class = self._soft_nms(class_detections, nms_threshold)
            else:
                filtered_class = self._standard_nms(class_detections, nms_threshold)
            
            final_detections.extend(filtered_class)
        
        return final_detections
    
    def _standard_nms(self, detections: List[DetectionResult], threshold: float) -> List[DetectionResult]:
        """🎯 NMS standard"""
        
        if len(detections) <= 1:
            return detections
        
        # Tri par confiance décroissante
        sorted_dets = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        remaining = sorted_dets.copy()
        
        while remaining:
            # Prendre le meilleur
            current = remaining.pop(0)
            keep.append(current)
            
            # Supprimer ceux qui se chevauchent trop
            remaining = [
                det for det in remaining 
                if self._calculate_iou(current.bbox, det.bbox) < threshold
            ]
        
        return keep
    
    def _soft_nms(self, detections: List[DetectionResult], threshold: float) -> List[DetectionResult]:
        """🎯 Soft-NMS (diminue confiance au lieu de supprimer)"""
        
        if len(detections) <= 1:
            return detections
        
        # Copie pour modification
        dets = [det for det in detections]
        
        for i, det_i in enumerate(dets):
            for j, det_j in enumerate(dets):
                if i != j:
                    iou = self._calculate_iou(det_i.bbox, det_j.bbox)
                    
                    if iou > threshold:
                        # Diminuer confiance selon IoU
                        decay_factor = np.exp(-(iou ** 2) / 0.5)
                        det_j.confidence *= decay_factor
        
        # Filtrer les détections avec confiance trop faible
        return [det for det in dets if det.confidence > 0.01]
    
    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """📐 Calcule l'IoU entre deux boîtes"""
        
        # Intersection
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Union
        area1 = bbox1.get_area()
        area2 = bbox2.get_area()
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _limit_max_detections(self, detections: List[DetectionResult], max_count: int) -> List[DetectionResult]:
        """🔢 Limite le nombre maximum de détections"""
        
        if len(detections) <= max_count:
            return detections
        
        # Garder les meilleures par confiance
        sorted_dets = sorted(detections, key=lambda x: x.confidence, reverse=True)
        return sorted_dets[:max_count]
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """📊 Statistiques de filtrage"""
        
        filter_rate = self.filtered_count / max(1, self.total_processed)
        
        return {
            "total_processed": self.total_processed,
            "total_filtered": self.filtered_count,
            "filter_rate": filter_rate,
            "kept_rate": 1.0 - filter_rate
        }

# 📊 AGRÉGATEUR DE RÉSULTATS
class ResultAggregator:
    """📊 Agrégateur intelligent de résultats multiples"""
    
    def __init__(self):
        self.aggregation_count = 0
        
        logger.info("📊 ResultAggregator initialisé")
    
    def aggregate_detections(
        self,
        detection_lists: List[List[DetectionResult]],
        method: AggregationMethod = AggregationMethod.WEIGHTED,
        weights: Optional[List[float]] = None
    ) -> List[DetectionResult]:
        """📊 Agrège plusieurs listes de détections"""
        
        if not detection_lists:
            return []
        
        if len(detection_lists) == 1:
            return detection_lists[0]
        
        self.aggregation_count += 1
        
        if method == AggregationMethod.SIMPLE:
            return self._simple_aggregation(detection_lists)
        elif method == AggregationMethod.WEIGHTED:
            return self._weighted_aggregation(detection_lists, weights)
        elif method == AggregationMethod.CONSENSUS:
            return self._consensus_aggregation(detection_lists)
        elif method == AggregationMethod.TEMPORAL:
            return self._temporal_aggregation(detection_lists)
        else:
            return self._simple_aggregation(detection_lists)
    
    def _simple_aggregation(self, detection_lists: List[List[DetectionResult]]) -> List[DetectionResult]:
        """📊 Agrégation simple (union)"""
        
        all_detections = []
        for det_list in detection_lists:
            all_detections.extend(det_list)
        
        return all_detections
    
    def _weighted_aggregation(
        self, 
        detection_lists: List[List[DetectionResult]], 
        weights: Optional[List[float]] = None
    ) -> List[DetectionResult]:
        """⚖️ Agrégation pondérée"""
        
        if weights is None:
            weights = [1.0] * len(detection_lists)
        
        if len(weights) != len(detection_lists):
            weights = [1.0] * len(detection_lists)
        
        # Normaliser les poids
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Pondérer les confiances
        weighted_detections = []
        
        for i, (det_list, weight) in enumerate(zip(detection_lists, weights)):
            for det in det_list:
                # Créer copie avec confiance pondérée
                weighted_det = DetectionResult(
                    bbox=det.bbox,
                    confidence=det.confidence * weight,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    class_name_fr=det.class_name_fr,
                    model_name=det.model_name,
                    detection_id=f"{det.detection_id}_w{i}"
                )
                weighted_detections.append(weighted_det)
        
        return weighted_detections
    
    def _consensus_aggregation(self, detection_lists: List[List[DetectionResult]]) -> List[DetectionResult]:
        """🤝 Agrégation par consensus"""
        
        # Regrouper détections similaires
        groups = self._group_similar_detections(detection_lists)
        
        consensus_detections = []
        
        for group in groups:
            if len(group) >= len(detection_lists) // 2:  # Majorité
                # Créer détection consensus
                consensus_det = self._create_consensus_detection(group)
                consensus_detections.append(consensus_det)
        
        return consensus_detections
    
    def _temporal_aggregation(self, detection_lists: List[List[DetectionResult]]) -> List[DetectionResult]:
        """⏰ Agrégation temporelle (pour vidéos)"""
        
        # Pistes de détections temporelles
        tracks = self._create_temporal_tracks(detection_lists)
        
        # Détections stables dans le temps
        stable_detections = []
        
        for track in tracks:
            if len(track) >= 3:  # Minimum 3 détections
                # Moyenner les détections de la piste
                averaged_det = self._average_track_detections(track)
                stable_detections.append(averaged_det)
        
        return stable_detections
    
    def _group_similar_detections(self, detection_lists: List[List[DetectionResult]]) -> List[List[DetectionResult]]:
        """👥 Groupe les détections similaires"""
        
        all_detections = []
        for det_list in detection_lists:
            all_detections.extend(det_list)
        
        groups = []
        processed = set()
        
        for i, det1 in enumerate(all_detections):
            if i in processed:
                continue
            
            group = [det1]
            processed.add(i)
            
            for j, det2 in enumerate(all_detections[i+1:], i+1):
                if j in processed:
                    continue
                
                # Vérifier similarité
                if self._are_detections_similar(det1, det2):
                    group.append(det2)
                    processed.add(j)
            
            groups.append(group)
        
        return groups
    
    def _are_detections_similar(self, det1: DetectionResult, det2: DetectionResult) -> bool:
        """🔍 Vérifie si deux détections sont similaires"""
        
        # Même classe
        if det1.class_name != det2.class_name:
            return False
        
        # IoU suffisant
        iou = self._calculate_iou(det1.bbox, det2.bbox)
        return iou > 0.5
    
    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """📐 Calcule l'IoU (même que DetectionFilter)"""
        
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = bbox1.get_area() + bbox2.get_area() - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _create_consensus_detection(self, group: List[DetectionResult]) -> DetectionResult:
        """🤝 Crée une détection consensus"""
        
        # Moyenner les boîtes englobantes
        avg_x1 = sum(det.bbox.x1 for det in group) / len(group)
        avg_y1 = sum(det.bbox.y1 for det in group) / len(group)
        avg_x2 = sum(det.bbox.x2 for det in group) / len(group)
        avg_y2 = sum(det.bbox.y2 for det in group) / len(group)
        
        avg_bbox = BoundingBox(
            x1=int(avg_x1),
            y1=int(avg_y1),
            x2=int(avg_x2),
            y2=int(avg_y2)
        )
        
        # Confiance moyenne pondérée
        avg_confidence = sum(det.confidence for det in group) / len(group)
        
        # Prendre infos du premier
        base_det = group[0]
        
        return DetectionResult(
            bbox=avg_bbox,
            confidence=avg_confidence,
            class_id=base_det.class_id,
            class_name=base_det.class_name,
            class_name_fr=base_det.class_name_fr,
            model_name=f"consensus_{len(group)}",
            detection_id=f"consensus_{int(time.time() * 1000)}"
        )
    
    def _create_temporal_tracks(self, detection_lists: List[List[DetectionResult]]) -> List[List[DetectionResult]]:
        """⏰ Crée des pistes temporelles"""
        
        tracks = []
        
        # Algorithme simple de tracking
        for frame_detections in detection_lists:
            for det in frame_detections:
                # Chercher piste existante
                best_track = None
                best_iou = 0.0
                
                for track in tracks:
                    if track and track[-1].class_name == det.class_name:
                        iou = self._calculate_iou(track[-1].bbox, det.bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_track = track
                
                # Ajouter à piste existante ou créer nouvelle
                if best_track and best_iou > 0.3:
                    best_track.append(det)
                else:
                    tracks.append([det])
        
        return tracks
    
    def _average_track_detections(self, track: List[DetectionResult]) -> DetectionResult:
        """📊 Moyenne les détections d'une piste"""
        
        # Moyenner positions et confiances
        avg_x1 = sum(det.bbox.x1 for det in track) / len(track)
        avg_y1 = sum(det.bbox.y1 for det in track) / len(track)
        avg_x2 = sum(det.bbox.x2 for det in track) / len(track)
        avg_y2 = sum(det.bbox.y2 for det in track) / len(track)
        
        avg_bbox = BoundingBox(
            x1=int(avg_x1),
            y1=int(avg_y1),
            x2=int(avg_x2),
            y2=int(avg_y2)
        )
        
        avg_confidence = sum(det.confidence for det in track) / len(track)
        
        base_det = track[0]
        
        return DetectionResult(
            bbox=avg_bbox,
            confidence=avg_confidence,
            class_id=base_det.class_id,
            class_name=base_det.class_name,
            class_name_fr=base_det.class_name_fr,
            model_name=f"track_{len(track)}",
            detection_id=f"track_{int(time.time() * 1000)}"
        )

# 🔄 POST-PROCESSEUR PRINCIPAL
class ResultPostprocessor:
    """🔄 Post-processeur principal des résultats"""
    
    def __init__(self):
        self.filter = DetectionFilter()
        self.aggregator = ResultAggregator()
        
        # Statistiques
        self.processed_results = 0
        self.total_processing_time = 0.0
        
        logger.info("🔄 ResultPostprocessor initialisé")
    
    async def process_results(
        self,
        detections: List[DetectionResult],
        config: PostprocessingConfig = None
    ) -> Tuple[List[DetectionResult], DetectionStatistics]:
        """🔄 Traitement principal des résultats"""
        
        start_time = time.time()
        config = config or PostprocessingConfig()
        
        try:
            # 1. Filtrage des détections
            filtered_detections = self.filter.filter_detections(
                detections, 
                config.min_confidence,
                config.max_detections,
                config
            )
            
            # 2. Génération des statistiques
            stats = None
            if config.generate_statistics:
                stats = self._generate_statistics(filtered_detections, config)
                stats.processing_time_ms = (time.time() - start_time) * 1000
            
            # 3. Mise à jour statistiques globales
            processing_time = time.time() - start_time
            self.processed_results += 1
            self.total_processing_time += processing_time
            
            return filtered_detections, stats
            
        except Exception as e:
            logger.error(f"❌ Erreur post-traitement: {e}")
            return detections, DetectionStatistics()
    
    async def process_multiple_results(
        self,
        detection_lists: List[List[DetectionResult]],
        aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED,
        config: PostprocessingConfig = None
    ) -> Tuple[List[DetectionResult], DetectionStatistics]:
        """🔄 Traitement de résultats multiples"""
        
        config = config or PostprocessingConfig()
        
        # Agrégation
        aggregated_detections = self.aggregator.aggregate_detections(
            detection_lists, aggregation_method
        )
        
        # Post-traitement standard
        return await self.process_results(aggregated_detections, config)
    
    def _generate_statistics(
        self, 
        detections: List[DetectionResult], 
        config: PostprocessingConfig
    ) -> DetectionStatistics:
        """📊 Génère les statistiques des détections"""
        
        stats = DetectionStatistics()
        
        if not detections:
            return stats
        
        # Statistiques de base
        stats.total_detections = len(detections)
        
        # Par classe
        class_counts = Counter(det.class_name for det in detections)
        stats.detections_by_class = dict(class_counts)
        
        # Distribution confiances
        confidences = [det.confidence for det in detections]
        stats.confidence_distribution = confidences
        stats.average_confidence = statistics.mean(confidences)
        stats.max_confidence = max(confidences)
        stats.min_confidence = min(confidences)
        
        # Distribution tailles
        box_areas = [det.bbox.get_area() for det in detections]
        stats.box_size_distribution = box_areas
        
        # Scores qualité
        stats.quality_score = self._calculate_quality_score(detections)
        stats.reliability_score = self._calculate_reliability_score(detections)
        
        return stats
    
    def _calculate_quality_score(self, detections: List[DetectionResult]) -> float:
        """💎 Calcule un score de qualité des détections"""
        
        if not detections:
            return 0.0
        
        # Facteurs de qualité
        avg_confidence = statistics.mean(det.confidence for det in detections)
        confidence_std = statistics.stdev([det.confidence for det in detections]) if len(detections) > 1 else 0.0
        
        # Diversité des classes
        unique_classes = len(set(det.class_name for det in detections))
        class_diversity = min(1.0, unique_classes / 10)  # Normaliser sur 10 classes max
        
        # Répartition des tailles
        areas = [det.bbox.get_area() for det in detections]
        area_std = statistics.stdev(areas) if len(areas) > 1 else 0.0
        area_diversity = min(1.0, area_std / 10000)  # Normaliser
        
        # Score composite
        quality_score = (
            avg_confidence * 0.5 +
            (1.0 - confidence_std) * 0.2 +
            class_diversity * 0.2 +
            area_diversity * 0.1
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_reliability_score(self, detections: List[DetectionResult]) -> float:
        """🎯 Calcule un score de fiabilité"""
        
        if not detections:
            return 0.0
        
        # Facteurs de fiabilité
        high_confidence_ratio = len([d for d in detections if d.confidence > 0.7]) / len(detections)
        
        # Cohérence des tailles
        areas = [det.bbox.get_area() for det in detections]
        reasonable_size_ratio = len([a for a in areas if 100 <= a <= 50000]) / len(areas)
        
        # Pas de détections dupliquées
        unique_positions = set((det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2) for det in detections)
        uniqueness_ratio = len(unique_positions) / len(detections)
        
        # Score composite
        reliability_score = (
            high_confidence_ratio * 0.4 +
            reasonable_size_ratio * 0.3 +
            uniqueness_ratio * 0.3
        )
        
        return max(0.0, min(1.0, reliability_score))
    
    async def generate_report(
        self, 
        detections: List[DetectionResult],
        stats: DetectionStatistics,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """📋 Génère un rapport détaillé"""
        
        report = {
            "summary": {
                "total_detections": len(detections),
                "unique_classes": len(set(det.class_name for det in detections)),
                "average_confidence": stats.average_confidence,
                "quality_score": stats.quality_score,
                "reliability_score": stats.reliability_score
            },
            "detections_by_class": stats.detections_by_class,
            "confidence_analysis": {
                "distribution": stats.confidence_distribution,
                "mean": stats.average_confidence,
                "max": stats.max_confidence,
                "min": stats.min_confidence,
                "std": statistics.stdev(stats.confidence_distribution) if len(stats.confidence_distribution) > 1 else 0.0
            },
            "detections_detail": [
                {
                    "class": det.class_name,
                    "class_fr": det.class_name_fr,
                    "confidence": det.confidence,
                    "bbox": {
                        "x1": det.bbox.x1,
                        "y1": det.bbox.y1,
                        "x2": det.bbox.x2,
                        "y2": det.bbox.y2,
                        "area": det.bbox.get_area()
                    },
                    "model": det.model_name
                }
                for det in detections
            ],
            "processing_info": {
                "processing_time_ms": stats.processing_time_ms,
                "timestamp": time.time()
            }
        }
        
        # Sauvegarde si chemin fourni
        if output_path:
            async with asyncio.to_thread(output_path.open, 'w') as f:
                await asyncio.to_thread(json.dump, report, f, indent=2)
        
        return report
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """📊 Métriques de performance"""
        
        avg_time = self.total_processing_time / max(1, self.processed_results)
        
        return {
            "processed_results": self.processed_results,
            "total_processing_time": self.total_processing_time,
            "average_processing_time_ms": avg_time * 1000,
            "throughput_results_per_sec": 1.0 / avg_time if avg_time > 0 else 0.0,
            "filter_stats": self.filter.get_filter_stats(),
            "aggregation_count": self.aggregator.aggregation_count
        }

# 📝 INFORMATIONS D'EXPORT
__all__ = [
    "ResultPostprocessor",
    "DetectionFilter",
    "ResultAggregator",
    "PostprocessingConfig",
    "DetectionStatistics",
    "FilteringMode",
    "AggregationMethod",
    "ContextType"
]